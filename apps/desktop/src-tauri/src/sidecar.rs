use std::env;
use std::fs::{self, File};
use std::io::{Read as _, Write as _};
use std::net::TcpListener;
#[cfg(unix)]
use std::os::fd::AsRawFd;
#[cfg(unix)]
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Manager};

const DEFAULT_HOST: &str = "127.0.0.1";
const DEFAULT_PORT: u16 = 9000;
const PORT_SEARCH_LIMIT: u16 = 32;
const DESKTOP_LEASE_TTL_SECS: u64 = 20;
const DESKTOP_LEASE_HEARTBEAT_SECS: u64 = 5;
const SIDECAR_LEASE_WAIT_SECS: u64 = 8;

pub struct SidecarState {
    inner: Mutex<ManagedSidecar>,
}

struct ManagedSidecar {
    child: Option<Child>,
    launch: Option<SidecarLaunchConfig>,
    desktop: Option<DesktopOwnership>,
}

#[derive(Debug, Clone)]
struct SidecarLaunchConfig {
    base_url: String,
    host: String,
    port: u16,
    database_path: PathBuf,
    scratch_dir: PathBuf,
    workspace_default_path: PathBuf,
    log_path: PathBuf,
    instance_id: String,
    desktop_lease_path: PathBuf,
    sidecar_state_path: PathBuf,
}

struct DesktopOwnership {
    instance_id: String,
    lease_path: PathBuf,
    lock_path: PathBuf,
    lock_file: File,
    stop_signal: Arc<AtomicBool>,
    heartbeat_thread: Option<JoinHandle<()>>,
}

#[derive(Debug, Clone)]
struct DesktopRuntimePaths {
    runtime_root: PathBuf,
    logs_dir: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct DesktopLeaseRecord {
    instance_id: String,
    desktop_pid: u32,
    created_at_unix_ms: u64,
    updated_at_unix_ms: u64,
    lease_expires_unix_ms: u64,
}

#[derive(Debug, Clone, Deserialize)]
struct SidecarProcessRecord {
    instance_id: String,
    pid: u32,
    lease_path: String,
}

#[derive(Debug, Serialize)]
pub struct SidecarBootstrapResponse {
    pub base_url: String,
    pub managed_by_desktop: bool,
}

#[derive(Debug, Serialize)]
pub struct SidecarStatusResponse {
    pub running: bool,
    pub managed_by_desktop: bool,
    pub base_url: String,
    pub pid: Option<u32>,
    pub database_path: String,
    pub scratch_dir: String,
    pub workspace_default_path: String,
    pub log_path: String,
}

impl Default for SidecarState {
    fn default() -> Self {
        Self {
            inner: Mutex::new(ManagedSidecar {
                child: None,
                launch: None,
                desktop: None,
            }),
        }
    }
}

impl Drop for SidecarState {
    fn drop(&mut self) {
        if let Ok(mut inner) = self.inner.lock() {
            stop_child_locked(&mut inner);
            release_desktop_ownership_locked(&mut inner);
        }
    }
}

impl SidecarState {
    pub fn initialize(&self, app_handle: &AppHandle) -> Result<(), String> {
        let mut inner = self
            .inner
            .lock()
            .map_err(|_| String::from("desktop sidecar state lock poisoned"))?;
        if inner.desktop.is_some() {
            return Ok(());
        }
        let paths = resolve_runtime_paths(app_handle)?;
        inner.desktop = Some(DesktopOwnership::acquire(
            paths.runtime_root.join("desktop.instance.lock"),
            paths.runtime_root.join("desktop.instance.json"),
        )?);
        Ok(())
    }

    pub fn ensure_started(
        &self,
        app_handle: &AppHandle,
    ) -> Result<SidecarBootstrapResponse, String> {
        let mut inner = self
            .inner
            .lock()
            .map_err(|_| String::from("desktop sidecar state lock poisoned"))?;

        if let Some(child) = inner.child.as_mut() {
            match child.try_wait() {
                Ok(None) => {
                    let launch = inner
                        .launch
                        .as_ref()
                        .ok_or_else(|| String::from("desktop sidecar launch metadata missing"))?;
                    return Ok(SidecarBootstrapResponse {
                        base_url: launch.base_url.clone(),
                        managed_by_desktop: true,
                    });
                }
                Ok(Some(_)) => {
                    inner.child = None;
                }
                Err(error) => {
                    return Err(format!("failed to inspect loomd sidecar: {error}"));
                }
            }
        }

        if inner.desktop.is_none() {
            let paths = resolve_runtime_paths(app_handle)?;
            inner.desktop = Some(DesktopOwnership::acquire(
                paths.runtime_root.join("desktop.instance.lock"),
                paths.runtime_root.join("desktop.instance.json"),
            )?);
        }

        let repo_root = find_repo_root().ok_or_else(|| {
            String::from("unable to locate Loom repo root for desktop sidecar startup")
        })?;
        let desktop = inner
            .desktop
            .as_ref()
            .ok_or_else(|| String::from("desktop ownership lease not initialized"))?;
        let launch = build_launch_config(app_handle, desktop)?;
        wait_for_prior_sidecar_shutdown(&launch)?;

        let child = spawn_sidecar(&repo_root, &launch)?;
        let base_url = launch.base_url.clone();
        inner.child = Some(child);
        inner.launch = Some(launch);

        Ok(SidecarBootstrapResponse {
            base_url,
            managed_by_desktop: true,
        })
    }

    pub fn status(&self) -> Result<SidecarStatusResponse, String> {
        let mut inner = self
            .inner
            .lock()
            .map_err(|_| String::from("desktop sidecar state lock poisoned"))?;
        let mut running = false;
        let mut pid = None;

        if let Some(child) = inner.child.as_mut() {
            match child.try_wait() {
                Ok(None) => {
                    running = true;
                    pid = Some(child.id());
                }
                Ok(Some(_)) => {
                    inner.child = None;
                }
                Err(error) => {
                    return Err(format!("failed to inspect loomd sidecar: {error}"));
                }
            }
        }

        let launch = inner.launch.clone().unwrap_or_else(default_launch_config);

        Ok(SidecarStatusResponse {
            running,
            managed_by_desktop: true,
            base_url: launch.base_url,
            pid,
            database_path: launch.database_path.display().to_string(),
            scratch_dir: launch.scratch_dir.display().to_string(),
            workspace_default_path: launch.workspace_default_path.display().to_string(),
            log_path: launch.log_path.display().to_string(),
        })
    }

    pub fn stop(&self) -> Result<(), String> {
        let mut inner = self
            .inner
            .lock()
            .map_err(|_| String::from("desktop sidecar state lock poisoned"))?;
        stop_child_locked(&mut inner);
        Ok(())
    }

    pub fn shutdown_for_exit(&self) -> Result<(), String> {
        let mut inner = self
            .inner
            .lock()
            .map_err(|_| String::from("desktop sidecar state lock poisoned"))?;
        stop_child_locked(&mut inner);
        release_desktop_ownership_locked(&mut inner);
        Ok(())
    }
}

impl DesktopOwnership {
    fn acquire(lock_path: PathBuf, lease_path: PathBuf) -> Result<Self, String> {
        if let Some(parent) = lock_path.parent() {
            fs::create_dir_all(parent)
                .map_err(|error| format!("failed to create desktop runtime dir: {error}"))?;
        }
        let lock_file = File::options()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(&lock_path)
            .map_err(|error| format!("failed to open desktop lock file: {error}"))?;

        #[cfg(unix)]
        {
            let result =
                unsafe { libc::flock(lock_file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
            if result != 0 {
                return Err(String::from(
                    "Loom Desktop is already running. Close the existing app before launching another instance.",
                ));
            }
        }

        #[cfg(not(unix))]
        {
            // Fallback on non-Unix platforms until native file-lock parity is added.
            if lease_path.exists() {
                return Err(String::from(
                    "Loom Desktop is already running. Close the existing app before launching another instance.",
                ));
            }
        }

        let instance_id = format!("desktop-{}-{}", std::process::id(), now_unix_ms());
        let stop_signal = Arc::new(AtomicBool::new(false));
        write_desktop_lease_record(&lease_path, &instance_id, DESKTOP_LEASE_TTL_SECS)?;
        let thread_signal = Arc::clone(&stop_signal);
        let thread_instance = instance_id.clone();
        let thread_lease_path = lease_path.clone();
        let heartbeat_thread = thread::spawn(move || {
            while !thread_signal.load(Ordering::Relaxed) {
                thread::sleep(Duration::from_secs(DESKTOP_LEASE_HEARTBEAT_SECS));
                if thread_signal.load(Ordering::Relaxed) {
                    break;
                }
                let _ = write_desktop_lease_record(
                    &thread_lease_path,
                    &thread_instance,
                    DESKTOP_LEASE_TTL_SECS,
                );
            }
        });

        Ok(Self {
            instance_id,
            lease_path,
            lock_path,
            lock_file,
            stop_signal,
            heartbeat_thread: Some(heartbeat_thread),
        })
    }

    fn release(mut self) {
        self.stop_signal.store(true, Ordering::Relaxed);
        if let Some(handle) = self.heartbeat_thread.take() {
            let _ = handle.join();
        }
        remove_desktop_lease_if_owned(&self.lease_path, &self.instance_id);

        #[cfg(unix)]
        unsafe {
            libc::flock(self.lock_file.as_raw_fd(), libc::LOCK_UN);
        }

        let _ = fs::remove_file(&self.lock_path);
    }
}

fn stop_child_locked(inner: &mut ManagedSidecar) {
    if let Some(mut child) = inner.child.take() {
        #[cfg(unix)]
        {
            if let Some(pid) = child.id().checked_add(0) {
                unsafe {
                    libc::killpg(pid as libc::pid_t, libc::SIGTERM);
                }
                let deadline = Instant::now() + Duration::from_secs(3);
                loop {
                    match child.try_wait() {
                        Ok(Some(_)) => break,
                        Ok(None) if Instant::now() < deadline => {
                            std::thread::sleep(Duration::from_millis(50));
                        }
                        Ok(None) | Err(_) => {
                            unsafe {
                                libc::killpg(pid as libc::pid_t, libc::SIGKILL);
                            }
                            break;
                        }
                    }
                }
            }
        }
        #[cfg(not(unix))]
        {
            let _ = child.kill();
        }
        let _ = child.wait();
    }

    if let Some(launch) = inner.launch.as_ref() {
        cleanup_sidecar_state(&launch.sidecar_state_path, &launch.instance_id);
    }
}

fn release_desktop_ownership_locked(inner: &mut ManagedSidecar) {
    if let Some(desktop) = inner.desktop.take() {
        desktop.release();
    }
}

fn resolve_runtime_paths(app_handle: &AppHandle) -> Result<DesktopRuntimePaths, String> {
    let app_data_dir = app_handle
        .path()
        .app_data_dir()
        .map_err(|error| format!("failed to resolve desktop app data dir: {error}"))?;
    let runtime_root = app_data_dir.join("runtime");
    let logs_dir = app_data_dir.join("logs");
    fs::create_dir_all(&runtime_root)
        .map_err(|error| format!("failed to create desktop runtime dir: {error}"))?;
    fs::create_dir_all(&logs_dir)
        .map_err(|error| format!("failed to create desktop log dir: {error}"))?;
    Ok(DesktopRuntimePaths {
        runtime_root,
        logs_dir,
    })
}

fn write_desktop_lease_record(
    lease_path: &Path,
    instance_id: &str,
    ttl_secs: u64,
) -> Result<(), String> {
    let now_ms = now_unix_ms();
    let record = DesktopLeaseRecord {
        instance_id: String::from(instance_id),
        desktop_pid: std::process::id(),
        created_at_unix_ms: now_ms,
        updated_at_unix_ms: now_ms,
        lease_expires_unix_ms: now_ms.saturating_add(ttl_secs.saturating_mul(1000)),
    };
    write_json_atomic(lease_path, &record)
}

fn remove_desktop_lease_if_owned(lease_path: &Path, instance_id: &str) {
    if let Some(existing) = read_json_file::<DesktopLeaseRecord>(lease_path) {
        if existing.instance_id != instance_id {
            return;
        }
    }
    let _ = fs::remove_file(lease_path);
}

fn wait_for_prior_sidecar_shutdown(launch: &SidecarLaunchConfig) -> Result<(), String> {
    #[cfg(not(unix))]
    {
        let _ = launch;
        return Ok(());
    }

    #[cfg(unix)]
    {
        let Some(record) = read_json_file::<SidecarProcessRecord>(&launch.sidecar_state_path)
        else {
            return Ok(());
        };
        if record.lease_path != launch.desktop_lease_path.display().to_string() {
            return Ok(());
        }
        let deadline = Instant::now() + Duration::from_secs(SIDECAR_LEASE_WAIT_SECS);
        while Instant::now() < deadline {
            if !process_matches_sidecar(record.pid, &record.instance_id, &record.lease_path) {
                cleanup_sidecar_state(&launch.sidecar_state_path, &record.instance_id);
                return Ok(());
            }
            thread::sleep(Duration::from_millis(200));
        }
        if process_matches_sidecar(record.pid, &record.instance_id, &record.lease_path) {
            terminate_verified_sidecar_process(
                record.pid,
                &record.instance_id,
                &record.lease_path,
            )?;
        }
        cleanup_sidecar_state(&launch.sidecar_state_path, &record.instance_id);
        Ok(())
    }
}

#[cfg(unix)]
fn terminate_verified_sidecar_process(
    pid: u32,
    instance_id: &str,
    lease_path: &str,
) -> Result<(), String> {
    if !process_matches_sidecar(pid, instance_id, lease_path) {
        return Ok(());
    }
    let term_rc = unsafe { libc::kill(pid as libc::pid_t, libc::SIGTERM) };
    if term_rc != 0 {
        return Err(format!("failed to terminate stale loomd sidecar pid {pid}"));
    }
    let deadline = Instant::now() + Duration::from_secs(3);
    while Instant::now() < deadline {
        if !process_matches_sidecar(pid, instance_id, lease_path) {
            return Ok(());
        }
        thread::sleep(Duration::from_millis(100));
    }
    if process_matches_sidecar(pid, instance_id, lease_path) {
        unsafe {
            libc::kill(pid as libc::pid_t, libc::SIGKILL);
        }
    }
    Ok(())
}

#[cfg(unix)]
fn process_matches_sidecar(pid: u32, instance_id: &str, lease_path: &str) -> bool {
    let output = match Command::new("ps")
        .arg("-p")
        .arg(pid.to_string())
        .arg("-o")
        .arg("args=")
        .output()
    {
        Ok(output) => output,
        Err(_) => return false,
    };
    if !output.status.success() {
        return false;
    }
    let args = String::from_utf8_lossy(&output.stdout);
    sidecar_args_match(args.as_ref(), instance_id, lease_path)
}

#[cfg(not(unix))]
fn process_matches_sidecar(_pid: u32, _instance_id: &str, _lease_path: &str) -> bool {
    false
}

fn sidecar_args_match(args: &str, instance_id: &str, lease_path: &str) -> bool {
    args.contains("loomd")
        && args.contains("--desktop-instance-token")
        && args.contains(instance_id)
        && args.contains("--desktop-lease-path")
        && args.contains(lease_path)
}

fn cleanup_sidecar_state(state_path: &Path, instance_id: &str) {
    if let Some(existing) = read_json_file::<SidecarProcessRecord>(state_path) {
        if existing.instance_id != instance_id {
            return;
        }
    }
    let _ = fs::remove_file(state_path);
}

fn spawn_sidecar(repo_root: &Path, launch: &SidecarLaunchConfig) -> Result<Child, String> {
    let stdout = File::options()
        .create(true)
        .append(true)
        .open(&launch.log_path)
        .map_err(|error| format!("failed to open loomd log file: {error}"))?;
    let stderr = stdout
        .try_clone()
        .map_err(|error| format!("failed to clone loomd log file handle: {error}"))?;

    let mut command = Command::new("uv");
    command
        .arg("run")
        .arg("loomd")
        .arg("--host")
        .arg(&launch.host)
        .arg("--port")
        .arg(launch.port.to_string())
        .arg("--database-path")
        .arg(&launch.database_path)
        .arg("--scratch-dir")
        .arg(&launch.scratch_dir)
        .arg("--workspace-default-path")
        .arg(&launch.workspace_default_path)
        .arg("--desktop-instance-token")
        .arg(&launch.instance_id)
        .arg("--desktop-lease-path")
        .arg(&launch.desktop_lease_path)
        .arg("--desktop-sidecar-state-path")
        .arg(&launch.sidecar_state_path)
        .current_dir(repo_root)
        .stdin(Stdio::null())
        .stdout(Stdio::from(stdout))
        .stderr(Stdio::from(stderr));

    #[cfg(unix)]
    unsafe {
        command.pre_exec(|| {
            // Group the child so graceful desktop shutdown can signal the uv/python stack together.
            libc::setpgid(0, 0);
            Ok(())
        });
    }

    command
        .spawn()
        .map_err(|error| format!("failed to start loomd sidecar: {error}"))
}

fn build_launch_config(
    app_handle: &AppHandle,
    desktop: &DesktopOwnership,
) -> Result<SidecarLaunchConfig, String> {
    let paths = resolve_runtime_paths(app_handle)?;
    let port = select_port(DEFAULT_HOST, DEFAULT_PORT, PORT_SEARCH_LIMIT)?;
    let database_path = paths.runtime_root.join("loomd.db");
    let scratch_dir = paths.runtime_root.join("scratch");
    fs::create_dir_all(&scratch_dir)
        .map_err(|error| format!("failed to create desktop scratch dir: {error}"))?;
    let workspace_default_path = home_dir()
        .map(|path| path.join("projects"))
        .unwrap_or_else(|| PathBuf::from("~/projects"));
    let log_path = paths.logs_dir.join("loomd.log");

    Ok(SidecarLaunchConfig {
        base_url: format!("http://{DEFAULT_HOST}:{port}"),
        host: String::from(DEFAULT_HOST),
        port,
        database_path,
        scratch_dir,
        workspace_default_path,
        log_path,
        instance_id: desktop.instance_id.clone(),
        desktop_lease_path: desktop.lease_path.clone(),
        sidecar_state_path: paths.runtime_root.join("loomd.sidecar.json"),
    })
}

fn default_launch_config() -> SidecarLaunchConfig {
    let workspace_default_path = home_dir()
        .map(|path| path.join("projects"))
        .unwrap_or_else(|| PathBuf::from("~/projects"));
    SidecarLaunchConfig {
        base_url: format!("http://{DEFAULT_HOST}:{DEFAULT_PORT}"),
        host: String::from(DEFAULT_HOST),
        port: DEFAULT_PORT,
        database_path: PathBuf::new(),
        scratch_dir: PathBuf::new(),
        workspace_default_path,
        log_path: PathBuf::new(),
        instance_id: String::new(),
        desktop_lease_path: PathBuf::new(),
        sidecar_state_path: PathBuf::new(),
    }
}

fn select_port(host: &str, start_port: u16, attempts: u16) -> Result<u16, String> {
    for offset in 0..attempts {
        let candidate = start_port.saturating_add(offset);
        let address = format!("{host}:{candidate}");
        if let Ok(listener) = TcpListener::bind(&address) {
            drop(listener);
            return Ok(candidate);
        }
    }
    Err(format!(
        "failed to find an available loopback port in range {start_port}-{}",
        start_port.saturating_add(attempts.saturating_sub(1)),
    ))
}

fn write_json_atomic<T: Serialize>(path: &Path, value: &T) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .map_err(|error| format!("failed to prepare runtime ownership dir: {error}"))?;
    }
    let payload = serde_json::to_vec_pretty(value)
        .map_err(|error| format!("failed to serialize ownership metadata: {error}"))?;
    let tmp_path = path.with_extension("tmp");
    let mut file = File::create(&tmp_path)
        .map_err(|error| format!("failed to create temp ownership file: {error}"))?;
    file.write_all(&payload)
        .map_err(|error| format!("failed to write temp ownership file: {error}"))?;
    file.sync_all()
        .map_err(|error| format!("failed to flush temp ownership file: {error}"))?;
    fs::rename(&tmp_path, path)
        .map_err(|error| format!("failed to finalize ownership file: {error}"))
}

fn read_json_file<T: for<'de> Deserialize<'de>>(path: &Path) -> Option<T> {
    let mut contents = String::new();
    let mut file = File::open(path).ok()?;
    file.read_to_string(&mut contents).ok()?;
    serde_json::from_str(&contents).ok()
}

fn now_unix_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_else(|_| Duration::from_secs(0))
        .as_millis() as u64
}

fn home_dir() -> Option<PathBuf> {
    env::var_os("HOME")
        .map(PathBuf::from)
        .or_else(|| env::var_os("USERPROFILE").map(PathBuf::from))
}

fn find_repo_root() -> Option<PathBuf> {
    let start = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    for candidate in start.ancestors() {
        if repo_marker_exists(candidate) {
            return Some(candidate.to_path_buf());
        }
    }
    None
}

fn repo_marker_exists(candidate: &Path) -> bool {
    candidate.join("pyproject.toml").exists() && candidate.join("src").join("loom").exists()
}

#[cfg(test)]
mod tests {
    use super::{
        read_json_file, sidecar_args_match, write_json_atomic, DesktopLeaseRecord,
        SidecarProcessRecord,
    };

    #[test]
    fn matches_sidecar_process_args_by_token_and_lease() {
        let args = "python -m loomd --desktop-instance-token desktop-123 --desktop-lease-path /tmp/desktop.instance.json";
        assert!(sidecar_args_match(
            args,
            "desktop-123",
            "/tmp/desktop.instance.json"
        ));
        assert!(!sidecar_args_match(
            args,
            "desktop-999",
            "/tmp/desktop.instance.json"
        ));
    }

    #[test]
    fn ownership_json_round_trips() {
        let dir = std::env::temp_dir().join(format!("loom-sidecar-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let lease_path = dir.join("desktop.instance.json");
        let record = DesktopLeaseRecord {
            instance_id: String::from("desktop-123"),
            desktop_pid: 42,
            created_at_unix_ms: 1,
            updated_at_unix_ms: 2,
            lease_expires_unix_ms: 3,
        };
        write_json_atomic(&lease_path, &record).unwrap();
        let loaded: DesktopLeaseRecord = read_json_file(&lease_path).unwrap();
        assert_eq!(loaded.instance_id, "desktop-123");

        let sidecar_path = dir.join("loomd.sidecar.json");
        std::fs::write(
            &sidecar_path,
            r#"{"instance_id":"desktop-123","pid":55,"host":"127.0.0.1","port":9000,"base_url":"http://127.0.0.1:9000","database_path":"/tmp/loomd.db","lease_path":"/tmp/desktop.instance.json","started_at_unix_ms":12}"#,
        )
        .unwrap();
        let sidecar: SidecarProcessRecord = read_json_file(&sidecar_path).unwrap();
        assert_eq!(sidecar.instance_id, "desktop-123");
        assert_eq!(sidecar.pid, 55);
        assert_eq!(sidecar.lease_path, "/tmp/desktop.instance.json");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
