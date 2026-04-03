use std::collections::BTreeMap;
use std::env;
use std::ffi::OsString;
use std::fs::{self, File};
use std::io::{Read as _, Write as _};
use std::net::{SocketAddr, TcpListener, TcpStream};
#[cfg(unix)]
use std::os::fd::AsRawFd;
#[cfg(unix)]
use std::os::unix::process::CommandExt;
use std::path::{Component, Path, PathBuf};
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
const SIDECAR_STARTUP_TIMEOUT_SECS: u64 = 20;
const SIDECAR_READY_POLL_MS: u64 = 250;
const BUNDLED_RUNTIME_MANIFEST_NAME: &str = "loom-desktop-bundle.json";
const BUNDLED_RUNTIME_SCHEMA_VERSION: u32 = 1;

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
    python_cache_dir: PathBuf,
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

#[derive(Debug, Clone, Deserialize, Serialize)]
struct BundledRuntimeManifest {
    schema_version: u32,
    enabled_extras: Vec<String>,
    loom_version: String,
    python_version: String,
    python_request: String,
    python_home_relative_path: String,
    python_executable_relative_path: String,
    environment_root_relative_path: String,
    site_packages_relative_path: String,
    entry_module: String,
    uv_version: String,
}

#[derive(Debug, Clone)]
struct PackagedSidecarRuntime {
    enabled_extras: Vec<String>,
    loom_version: String,
    python_version: String,
    python_request: String,
    uv_version: String,
    python_home: PathBuf,
    python_executable: PathBuf,
    environment_root: PathBuf,
    site_packages: PathBuf,
    entry_module: String,
}

#[derive(Debug, Clone)]
struct DevSidecarRuntime {
    repo_root: PathBuf,
}

#[derive(Debug, Clone)]
enum SidecarRuntime {
    Packaged(PackagedSidecarRuntime),
    Dev(DevSidecarRuntime),
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct SidecarSpawnCommand {
    program: PathBuf,
    args: Vec<String>,
    current_dir: PathBuf,
    env: BTreeMap<String, String>,
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
    pub runtime: Option<SidecarRuntimeMetadataResponse>,
    pub runtime_error: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct SidecarRuntimeMetadataResponse {
    pub mode: String,
    pub enabled_extras: Vec<String>,
    pub loom_version: Option<String>,
    pub python_version: Option<String>,
    pub python_request: Option<String>,
    pub uv_version: Option<String>,
    pub entry_module: Option<String>,
    pub python_home: Option<String>,
    pub python_executable: Option<String>,
    pub environment_root: Option<String>,
    pub site_packages: Option<String>,
    pub repo_root: Option<String>,
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

        let desktop = inner
            .desktop
            .as_ref()
            .ok_or_else(|| String::from("desktop ownership lease not initialized"))?;
        let launch = build_launch_config(app_handle, desktop)?;
        wait_for_prior_sidecar_shutdown(&launch)?;

        let mut child = spawn_sidecar(app_handle, &launch)?;
        if let Err(error) = wait_for_sidecar_ready(&mut child, &launch) {
            terminate_spawned_child(&mut child);
            cleanup_sidecar_state(&launch.sidecar_state_path, &launch.instance_id);
            return Err(error);
        }
        let base_url = launch.base_url.clone();
        inner.child = Some(child);
        inner.launch = Some(launch);

        Ok(SidecarBootstrapResponse {
            base_url,
            managed_by_desktop: true,
        })
    }

    pub fn status(&self, app_handle: &AppHandle) -> Result<SidecarStatusResponse, String> {
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
        let (runtime, runtime_error) = match resolve_sidecar_runtime(app_handle) {
            Ok(runtime) => (Some(runtime.metadata()), None),
            Err(error) => (None, Some(error)),
        };

        Ok(SidecarStatusResponse {
            running,
            managed_by_desktop: true,
            base_url: launch.base_url,
            pid,
            database_path: launch.database_path.display().to_string(),
            scratch_dir: launch.scratch_dir.display().to_string(),
            workspace_default_path: launch.workspace_default_path.display().to_string(),
            log_path: launch.log_path.display().to_string(),
            runtime,
            runtime_error,
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

impl SidecarRuntime {
    fn mode_label(&self) -> &'static str {
        match self {
            Self::Packaged(_) => "packaged",
            Self::Dev(_) => "development",
        }
    }

    fn metadata(&self) -> SidecarRuntimeMetadataResponse {
        match self {
            Self::Packaged(runtime) => SidecarRuntimeMetadataResponse {
                mode: String::from("packaged"),
                enabled_extras: runtime.enabled_extras.clone(),
                loom_version: Some(runtime.loom_version.clone()),
                python_version: Some(runtime.python_version.clone()),
                python_request: Some(runtime.python_request.clone()),
                uv_version: Some(runtime.uv_version.clone()),
                entry_module: Some(runtime.entry_module.clone()),
                python_home: Some(runtime.python_home.display().to_string()),
                python_executable: Some(runtime.python_executable.display().to_string()),
                environment_root: Some(runtime.environment_root.display().to_string()),
                site_packages: Some(runtime.site_packages.display().to_string()),
                repo_root: None,
            },
            Self::Dev(runtime) => SidecarRuntimeMetadataResponse {
                mode: String::from("development"),
                enabled_extras: Vec::new(),
                loom_version: None,
                python_version: None,
                python_request: None,
                uv_version: None,
                entry_module: Some(String::from("loom.daemon.cli")),
                python_home: None,
                python_executable: None,
                environment_root: None,
                site_packages: None,
                repo_root: Some(runtime.repo_root.display().to_string()),
            },
        }
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

fn resolve_sidecar_runtime(app_handle: &AppHandle) -> Result<SidecarRuntime, String> {
    let resource_dir = app_handle
        .path()
        .resource_dir()
        .map_err(|error| format!("failed to resolve bundled desktop resources dir: {error}"));
    match resource_dir {
        Ok(resource_dir) => match resolve_packaged_runtime_from(&resource_dir) {
            Ok(runtime) => Ok(SidecarRuntime::Packaged(runtime)),
            Err(packaged_error) if cfg!(debug_assertions) => resolve_dev_runtime()
                .map(SidecarRuntime::Dev)
                .map_err(|dev_error| {
                    format!(
                        "bundled runtime unavailable ({packaged_error}); development fallback unavailable: {dev_error}"
                    )
                }),
            Err(packaged_error) => Err(format!("missing bundled runtime: {packaged_error}")),
        },
        Err(resource_error) if cfg!(debug_assertions) => resolve_dev_runtime()
            .map(SidecarRuntime::Dev)
            .map_err(|dev_error| {
                format!(
                    "{resource_error}; development fallback unavailable: {dev_error}"
                )
            }),
        Err(resource_error) => Err(resource_error),
    }
}

fn resolve_dev_runtime() -> Result<DevSidecarRuntime, String> {
    let repo_root = find_repo_root().ok_or_else(|| {
        String::from("unable to locate Loom repo root for desktop development startup")
    })?;
    Ok(DevSidecarRuntime { repo_root })
}

fn resolve_packaged_runtime_from(resources_root: &Path) -> Result<PackagedSidecarRuntime, String> {
    let manifest_path = resources_root.join(BUNDLED_RUNTIME_MANIFEST_NAME);
    let manifest_payload = fs::read_to_string(&manifest_path).map_err(|error| {
        format!(
            "expected bundled runtime manifest at {}: {error}",
            manifest_path.display()
        )
    })?;
    let manifest: BundledRuntimeManifest = serde_json::from_str(&manifest_payload)
        .map_err(|error| format!("failed to parse bundled runtime manifest: {error}"))?;
    if manifest.schema_version != BUNDLED_RUNTIME_SCHEMA_VERSION {
        return Err(format!(
            "unsupported bundled runtime manifest schema {} in {}",
            manifest.schema_version,
            manifest_path.display()
        ));
    }

    let python_home = resolve_bundle_relative_path(
        resources_root,
        &manifest.python_home_relative_path,
        "python home",
    )?;
    let python_executable = resolve_bundle_relative_path(
        resources_root,
        &manifest.python_executable_relative_path,
        "python executable",
    )?;
    let environment_root = resolve_bundle_relative_path(
        resources_root,
        &manifest.environment_root_relative_path,
        "environment root",
    )?;
    let site_packages = resolve_bundle_relative_path(
        resources_root,
        &manifest.site_packages_relative_path,
        "site-packages",
    )?;

    assert_existing_dir(&python_home, "bundled Python home")?;
    assert_existing_file(&python_executable, "bundled Python executable")?;
    assert_existing_dir(&environment_root, "bundled Loom environment")?;
    assert_existing_dir(&site_packages, "bundled site-packages")?;

    if manifest.entry_module.trim().is_empty() {
        return Err(format!(
            "bundled runtime manifest entry_module is empty in {}",
            manifest_path.display()
        ));
    }

    Ok(PackagedSidecarRuntime {
        enabled_extras: manifest.enabled_extras,
        loom_version: manifest.loom_version,
        python_version: manifest.python_version,
        python_request: manifest.python_request,
        uv_version: manifest.uv_version,
        python_home,
        python_executable,
        environment_root,
        site_packages,
        entry_module: manifest.entry_module,
    })
}

fn sanitize_bundle_relative_path(raw: &str, label: &str) -> Result<PathBuf, String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Err(format!("bundled runtime manifest {label} path is empty"));
    }
    let path = PathBuf::from(trimmed);
    if path.is_absolute() {
        return Err(format!(
            "bundled runtime manifest {label} path must be relative: {trimmed}"
        ));
    }

    let mut cleaned = PathBuf::new();
    for component in path.components() {
        match component {
            Component::Normal(part) => cleaned.push(part),
            Component::CurDir => {}
            Component::ParentDir => {
                return Err(format!(
                    "bundled runtime manifest {label} path cannot escape the app bundle: {trimmed}"
                ));
            }
            Component::RootDir | Component::Prefix(_) => {
                return Err(format!(
                    "bundled runtime manifest {label} path is invalid: {trimmed}"
                ));
            }
        }
    }

    if cleaned.as_os_str().is_empty() {
        return Err(format!("bundled runtime manifest {label} path is empty"));
    }
    Ok(cleaned)
}

fn resolve_bundle_relative_path(
    resources_root: &Path,
    raw: &str,
    label: &str,
) -> Result<PathBuf, String> {
    Ok(resources_root.join(sanitize_bundle_relative_path(raw, label)?))
}

fn assert_existing_file(path: &Path, label: &str) -> Result<(), String> {
    if !path.exists() || !path.is_file() {
        return Err(format!("{label} is missing at {}", path.display()));
    }
    Ok(())
}

fn assert_existing_dir(path: &Path, label: &str) -> Result<(), String> {
    if !path.exists() || !path.is_dir() {
        return Err(format!("{label} is missing at {}", path.display()));
    }
    Ok(())
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
    (args.contains("loom.daemon.cli") || args.contains("loomd"))
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

fn spawn_sidecar(app_handle: &AppHandle, launch: &SidecarLaunchConfig) -> Result<Child, String> {
    let runtime = resolve_sidecar_runtime(app_handle).map_err(|error| {
        append_launcher_diagnostic(&launch.log_path, &error);
        error
    })?;
    let spawn_command = build_spawn_command(&runtime, launch).map_err(|error| {
        append_launcher_diagnostic(&launch.log_path, &error);
        error
    })?;
    append_launcher_diagnostic(
        &launch.log_path,
        &format!(
            "launching {} loomd sidecar ({})",
            runtime.mode_label(),
            format_spawn_command(&spawn_command)
        ),
    );

    let stdout = File::options()
        .create(true)
        .append(true)
        .open(&launch.log_path)
        .map_err(|error| format!("failed to open loomd log file: {error}"))?;
    let stderr = stdout
        .try_clone()
        .map_err(|error| format!("failed to clone loomd log file handle: {error}"))?;

    let mut command = Command::new(&spawn_command.program);
    command
        .args(&spawn_command.args)
        .current_dir(&spawn_command.current_dir)
        .stdin(Stdio::null())
        .stdout(Stdio::from(stdout))
        .stderr(Stdio::from(stderr));
    for (key, value) in &spawn_command.env {
        command.env(key, value);
    }

    #[cfg(unix)]
    unsafe {
        command.pre_exec(|| {
            libc::setpgid(0, 0);
            Ok(())
        });
    }

    command.spawn().map_err(|error| {
        let label = match runtime {
            SidecarRuntime::Packaged(_) => "bundled loomd sidecar",
            SidecarRuntime::Dev(_) => "development loomd sidecar",
        };
        let message = format!("failed to start {label}: {error}");
        append_launcher_diagnostic(&launch.log_path, &message);
        message
    })
}

fn wait_for_sidecar_ready(child: &mut Child, launch: &SidecarLaunchConfig) -> Result<(), String> {
    let deadline = Instant::now() + Duration::from_secs(SIDECAR_STARTUP_TIMEOUT_SECS);
    let address: SocketAddr = format!("{}:{}", launch.host, launch.port)
        .parse()
        .map_err(|error| format!("invalid loomd startup address {}:{}: {error}", launch.host, launch.port))?;

    while Instant::now() < deadline {
        match child.try_wait() {
            Ok(Some(status)) => {
                let message = format!("bundled loomd exited before startup completed with status {status}");
                append_launcher_diagnostic(&launch.log_path, &message);
                return Err(message);
            }
            Ok(None) => {}
            Err(error) => {
                let message = format!("failed to inspect loomd startup state: {error}");
                append_launcher_diagnostic(&launch.log_path, &message);
                return Err(message);
            }
        }

        if runtime_endpoint_ready(address) {
            append_launcher_diagnostic(
                &launch.log_path,
                &format!("loomd sidecar became ready at {}", launch.base_url),
            );
            return Ok(());
        }

        thread::sleep(Duration::from_millis(SIDECAR_READY_POLL_MS));
    }

    let message = format!(
        "bundled loomd did not become ready within {}s at {}",
        SIDECAR_STARTUP_TIMEOUT_SECS, launch.base_url
    );
    append_launcher_diagnostic(&launch.log_path, &message);
    Err(message)
}

fn runtime_endpoint_ready(address: SocketAddr) -> bool {
    let mut stream = match TcpStream::connect_timeout(&address, Duration::from_millis(500)) {
        Ok(stream) => stream,
        Err(_) => return false,
    };
    let _ = stream.set_read_timeout(Some(Duration::from_millis(500)));
    let _ = stream.set_write_timeout(Some(Duration::from_millis(500)));

    if stream
        .write_all(b"GET /runtime HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n")
        .is_err()
    {
        return false;
    }

    let mut response = [0_u8; 256];
    let bytes = match stream.read(&mut response) {
        Ok(bytes) if bytes > 0 => bytes,
        _ => return false,
    };
    response_is_http_ok(&response[..bytes])
}

fn response_is_http_ok(response: &[u8]) -> bool {
    let text = String::from_utf8_lossy(response);
    text.starts_with("HTTP/1.1 200") || text.starts_with("HTTP/1.0 200")
}

fn terminate_spawned_child(child: &mut Child) {
    #[cfg(unix)]
    {
        let pid = child.id() as libc::pid_t;
        unsafe {
            libc::killpg(pid, libc::SIGTERM);
        }
        let deadline = Instant::now() + Duration::from_secs(3);
        loop {
            match child.try_wait() {
                Ok(Some(_)) => break,
                Ok(None) if Instant::now() < deadline => thread::sleep(Duration::from_millis(50)),
                Ok(None) | Err(_) => {
                    unsafe {
                        libc::killpg(pid, libc::SIGKILL);
                    }
                    break;
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

fn build_spawn_command(
    runtime: &SidecarRuntime,
    launch: &SidecarLaunchConfig,
) -> Result<SidecarSpawnCommand, String> {
    match runtime {
        SidecarRuntime::Packaged(runtime) => build_packaged_spawn_command(runtime, launch),
        SidecarRuntime::Dev(runtime) => Ok(build_dev_spawn_command(runtime, launch)),
    }
}

fn shared_sidecar_args(launch: &SidecarLaunchConfig) -> Vec<String> {
    vec![
        String::from("--host"),
        launch.host.clone(),
        String::from("--port"),
        launch.port.to_string(),
        String::from("--database-path"),
        launch.database_path.display().to_string(),
        String::from("--scratch-dir"),
        launch.scratch_dir.display().to_string(),
        String::from("--workspace-default-path"),
        launch.workspace_default_path.display().to_string(),
        String::from("--desktop-instance-token"),
        launch.instance_id.clone(),
        String::from("--desktop-lease-path"),
        launch.desktop_lease_path.display().to_string(),
        String::from("--desktop-sidecar-state-path"),
        launch.sidecar_state_path.display().to_string(),
    ]
}

fn build_packaged_spawn_command(
    runtime: &PackagedSidecarRuntime,
    launch: &SidecarLaunchConfig,
) -> Result<SidecarSpawnCommand, String> {
    let mut args = vec![String::from("-m"), runtime.entry_module.clone()];
    args.extend(shared_sidecar_args(launch));

    let mut env = BTreeMap::new();
    env.insert(
        String::from("PYTHONHOME"),
        runtime.python_home.display().to_string(),
    );
    env.insert(
        String::from("PYTHONPATH"),
        runtime.site_packages.display().to_string(),
    );
    env.insert(String::from("PYTHONDONTWRITEBYTECODE"), String::from("1"));
    env.insert(String::from("PYTHONNOUSERSITE"), String::from("1"));
    env.insert(String::from("PYTHONUNBUFFERED"), String::from("1"));
    env.insert(
        String::from("PYTHONPYCACHEPREFIX"),
        launch.python_cache_dir.display().to_string(),
    );
    env.insert(
        String::from("PATH"),
        build_path_env(&[
            runtime.environment_root.join("bin"),
            runtime.python_home.join("bin"),
        ])?,
    );

    Ok(SidecarSpawnCommand {
        program: runtime.python_executable.clone(),
        args,
        current_dir: launch.scratch_dir.clone(),
        env,
    })
}

fn build_dev_spawn_command(
    runtime: &DevSidecarRuntime,
    launch: &SidecarLaunchConfig,
) -> SidecarSpawnCommand {
    let mut args = vec![
        String::from("run"),
        String::from("python"),
        String::from("-m"),
        String::from("loom.daemon.cli"),
    ];
    args.extend(shared_sidecar_args(launch));

    let mut env = BTreeMap::new();
    env.insert(String::from("PYTHONUNBUFFERED"), String::from("1"));

    SidecarSpawnCommand {
        program: PathBuf::from("uv"),
        args,
        current_dir: runtime.repo_root.clone(),
        env,
    }
}

fn build_path_env(prefixes: &[PathBuf]) -> Result<String, String> {
    let mut entries: Vec<PathBuf> = prefixes.to_vec();
    if let Some(existing) = env::var_os("PATH") {
        entries.extend(env::split_paths(&existing));
    }
    let joined: OsString =
        env::join_paths(entries).map_err(|error| format!("failed to construct PATH: {error}"))?;
    Ok(joined.to_string_lossy().into_owned())
}

fn format_spawn_command(command: &SidecarSpawnCommand) -> String {
    let mut parts = Vec::with_capacity(command.args.len() + 1);
    parts.push(command.program.display().to_string());
    parts.extend(command.args.iter().cloned());
    parts.join(" ")
}

fn append_launcher_diagnostic(log_path: &Path, message: &str) {
    if let Some(parent) = log_path.parent() {
        let _ = fs::create_dir_all(parent);
    }
    if let Ok(mut file) = File::options().create(true).append(true).open(log_path) {
        let _ = writeln!(file, "[loom-desktop] {message}");
    }
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
    let python_cache_dir = paths.runtime_root.join("python-cache");
    fs::create_dir_all(&python_cache_dir)
        .map_err(|error| format!("failed to create desktop Python cache dir: {error}"))?;
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
        python_cache_dir,
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
        python_cache_dir: PathBuf::new(),
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
        build_packaged_spawn_command, read_json_file, resolve_packaged_runtime_from,
        response_is_http_ok, sanitize_bundle_relative_path, sidecar_args_match, write_json_atomic,
        BundledRuntimeManifest, DesktopLeaseRecord, PackagedSidecarRuntime, SidecarLaunchConfig,
        SidecarProcessRecord,
    };
    use std::path::{Path, PathBuf};

    #[test]
    fn matches_sidecar_process_args_by_token_and_lease() {
        let args = "python -m loom.daemon.cli --desktop-instance-token desktop-123 --desktop-lease-path /tmp/desktop.instance.json";
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

    #[test]
    fn resolves_packaged_runtime_from_manifest() {
        let dir = temp_test_dir("loom-packaged-runtime");
        let python_home = dir.join("python");
        let python_bin = python_home.join("bin");
        let python_executable = python_bin.join("python3.11");
        let environment_root = dir.join("loom-env");
        let site_packages = environment_root.join("lib/python3.11/site-packages");
        std::fs::create_dir_all(&python_bin).unwrap();
        std::fs::create_dir_all(&site_packages).unwrap();
        std::fs::write(&python_executable, "").unwrap();

        write_manifest(
            &dir,
            BundledRuntimeManifest {
                schema_version: 1,
                enabled_extras: vec![
                    String::from("browser"),
                    String::from("mcp"),
                    String::from("pdf"),
                    String::from("treesitter"),
                ],
                loom_version: String::from("0.2.2"),
                python_version: String::from("3.11.14"),
                python_request: String::from("3.11.14"),
                python_home_relative_path: String::from("python"),
                python_executable_relative_path: String::from("python/bin/python3.11"),
                environment_root_relative_path: String::from("loom-env"),
                site_packages_relative_path: String::from("loom-env/lib/python3.11/site-packages"),
                entry_module: String::from("loom.daemon.cli"),
                uv_version: String::from("uv 0.10.12"),
            },
        );

        let runtime = resolve_packaged_runtime_from(&dir).unwrap();
        assert_eq!(runtime.enabled_extras, vec!["browser", "mcp", "pdf", "treesitter"]);
        assert_eq!(runtime.loom_version, "0.2.2");
        assert_eq!(runtime.python_version, "3.11.14");
        assert_eq!(runtime.python_request, "3.11.14");
        assert_eq!(runtime.uv_version, "uv 0.10.12");
        assert_eq!(runtime.python_home, python_home);
        assert_eq!(runtime.python_executable, python_executable);
        assert_eq!(runtime.environment_root, environment_root);
        assert_eq!(runtime.site_packages, site_packages);
    }

    #[test]
    fn accepts_http_200_runtime_probe_response() {
        assert!(response_is_http_ok(b"HTTP/1.1 200 OK\r\ncontent-type: application/json\r\n"));
        assert!(response_is_http_ok(b"HTTP/1.0 200 OK\r\n"));
        assert!(!response_is_http_ok(b"HTTP/1.1 503 Service Unavailable\r\n"));
    }

    #[test]
    fn rejects_bundle_manifest_paths_that_escape_resource_root() {
        assert!(sanitize_bundle_relative_path("../python", "python home").is_err());
    }

    #[test]
    fn packaged_spawn_command_uses_bundled_python_and_cache_overrides() {
        let dir = temp_test_dir("loom-packaged-spawn");
        let runtime = PackagedSidecarRuntime {
            enabled_extras: vec![
                String::from("browser"),
                String::from("mcp"),
                String::from("pdf"),
                String::from("treesitter"),
            ],
            loom_version: String::from("0.2.2"),
            python_version: String::from("3.11.14"),
            python_request: String::from("3.11.14"),
            uv_version: String::from("uv 0.10.12"),
            python_home: dir.join("python"),
            python_executable: dir.join("python/bin/python3.11"),
            environment_root: dir.join("loom-env"),
            site_packages: dir.join("loom-env/lib/python3.11/site-packages"),
            entry_module: String::from("loom.daemon.cli"),
        };
        let launch = test_launch_config(&dir);

        let command = build_packaged_spawn_command(&runtime, &launch).unwrap();
        assert_eq!(command.program, runtime.python_executable);
        assert_eq!(command.args[0], "-m");
        assert_eq!(command.args[1], "loom.daemon.cli");
        assert_eq!(
            command.env.get("PYTHONHOME").map(String::as_str),
            Some(runtime.python_home.display().to_string().as_str())
        );
        assert_eq!(
            command.env.get("PYTHONPATH").map(String::as_str),
            Some(runtime.site_packages.display().to_string().as_str())
        );
        assert_eq!(
            command.env.get("PYTHONPYCACHEPREFIX").map(String::as_str),
            Some(launch.python_cache_dir.display().to_string().as_str())
        );
        let path_env = command.env.get("PATH").unwrap();
        assert!(path_env.contains("loom-env/bin"));
        assert!(path_env.contains("python/bin"));
        assert_eq!(command.current_dir, launch.scratch_dir);
    }

    fn write_manifest(dir: &Path, manifest: BundledRuntimeManifest) {
        let path = dir.join("loom-desktop-bundle.json");
        std::fs::write(path, serde_json::to_vec_pretty(&manifest).unwrap()).unwrap();
    }

    fn temp_test_dir(prefix: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("{prefix}-{}", now_suffix()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn now_suffix() -> String {
        format!("{}-{}", std::process::id(), super::now_unix_ms())
    }

    fn test_launch_config(dir: &Path) -> SidecarLaunchConfig {
        SidecarLaunchConfig {
            base_url: String::from("http://127.0.0.1:9000"),
            host: String::from("127.0.0.1"),
            port: 9000,
            database_path: dir.join("runtime/loomd.db"),
            scratch_dir: dir.join("runtime/scratch"),
            workspace_default_path: dir.join("workspace"),
            log_path: dir.join("logs/loomd.log"),
            instance_id: String::from("desktop-123"),
            desktop_lease_path: dir.join("runtime/desktop.instance.json"),
            sidecar_state_path: dir.join("runtime/loomd.sidecar.json"),
            python_cache_dir: dir.join("runtime/python-cache"),
        }
    }
}
