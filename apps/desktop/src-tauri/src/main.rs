#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod sidecar;

use sidecar::{SidecarBootstrapResponse, SidecarState, SidecarStatusResponse};
use std::path::{Component, PathBuf};
use std::process::Command;
use tauri::Manager;

#[derive(serde::Deserialize)]
struct WorkspaceFileImportRequest {
    relative_path: String,
    bytes: Vec<u8>,
    overwrite: Option<bool>,
}

#[tauri::command]
fn desktop_bootstrap(
    app_handle: tauri::AppHandle,
    state: tauri::State<'_, SidecarState>,
) -> Result<SidecarBootstrapResponse, String> {
    state.ensure_started(&app_handle)
}

#[tauri::command]
fn desktop_sidecar_status(
    state: tauri::State<'_, SidecarState>,
) -> Result<SidecarStatusResponse, String> {
    state.status()
}

#[tauri::command]
fn desktop_stop_sidecar(state: tauri::State<'_, SidecarState>) -> Result<(), String> {
    state.stop()
}

#[tauri::command]
fn desktop_create_workspace_directory(path: String) -> Result<String, String> {
    let candidate = PathBuf::from(path.trim());
    if candidate.as_os_str().is_empty() {
        return Err("Workspace path is required.".to_string());
    }
    if candidate.exists() && !candidate.is_dir() {
        return Err(format!("Path is not a directory: {}", candidate.display()));
    }
    std::fs::create_dir_all(&candidate)
        .map_err(|error| format!("Failed to create workspace directory: {error}"))?;
    Ok(candidate.to_string_lossy().into_owned())
}

fn sanitize_workspace_relative_path(relative_path: &str) -> Result<PathBuf, String> {
    let trimmed = relative_path.trim();
    if trimmed.is_empty() {
        return Err("File path is required.".to_string());
    }
    let raw = PathBuf::from(trimmed);
    if raw.is_absolute() {
        return Err("File path must be relative to the workspace.".to_string());
    }

    let mut cleaned = PathBuf::new();
    for component in raw.components() {
        match component {
            Component::Normal(part) => cleaned.push(part),
            Component::CurDir => {}
            Component::ParentDir => {
                return Err("File path cannot escape the workspace.".to_string());
            }
            Component::RootDir | Component::Prefix(_) => {
                return Err("File path must stay inside the workspace.".to_string());
            }
        }
    }

    if cleaned.as_os_str().is_empty() {
        return Err("File path is required.".to_string());
    }
    Ok(cleaned)
}

fn resolve_workspace_target(workspace_path: &str, relative_path: &str) -> Result<PathBuf, String> {
    let workspace_root = PathBuf::from(workspace_path.trim());
    if workspace_root.as_os_str().is_empty() {
        return Err("Workspace path is required.".to_string());
    }
    if !workspace_root.exists() || !workspace_root.is_dir() {
        return Err(format!(
            "Workspace directory is unavailable: {}",
            workspace_root.display()
        ));
    }
    let clean_relative = sanitize_workspace_relative_path(relative_path)?;
    Ok(workspace_root.join(clean_relative))
}

fn open_path_with_system(path: &PathBuf) -> Result<(), String> {
    #[cfg(target_os = "macos")]
    let status = Command::new("open").arg(path).status();

    #[cfg(target_os = "windows")]
    let status = Command::new("cmd")
        .args(["/C", "start", "", &path.to_string_lossy()])
        .status();

    #[cfg(all(unix, not(target_os = "macos")))]
    let status = Command::new("xdg-open").arg(path).status();

    let status = status.map_err(|error| format!("Failed to launch file: {error}"))?;
    if status.success() {
        Ok(())
    } else {
        Err(format!("System open command exited with status: {status}"))
    }
}

fn reveal_path_with_system(path: &PathBuf) -> Result<(), String> {
    #[cfg(target_os = "macos")]
    let status = Command::new("open").args(["-R"]).arg(path).status();

    #[cfg(target_os = "windows")]
    let status = Command::new("explorer")
        .arg(format!("/select,{}", path.to_string_lossy()))
        .status();

    #[cfg(all(unix, not(target_os = "macos")))]
    let status = Command::new("xdg-open")
        .arg(path.parent().unwrap_or(path))
        .status();

    let status = status.map_err(|error| format!("Failed to reveal file: {error}"))?;
    if status.success() {
        Ok(())
    } else {
        Err(format!(
            "System reveal command exited with status: {status}"
        ))
    }
}

#[tauri::command]
fn desktop_create_workspace_file(
    workspace_path: String,
    relative_path: String,
    content: String,
    overwrite: Option<bool>,
) -> Result<String, String> {
    let workspace_root = PathBuf::from(workspace_path.trim());
    if workspace_root.as_os_str().is_empty() {
        return Err("Workspace path is required.".to_string());
    }
    if !workspace_root.exists() || !workspace_root.is_dir() {
        return Err(format!(
            "Workspace directory is unavailable: {}",
            workspace_root.display()
        ));
    }

    let clean_relative = sanitize_workspace_relative_path(&relative_path)?;
    let target_path = workspace_root.join(&clean_relative);
    if let Some(parent) = target_path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|error| format!("Failed to prepare parent directories: {error}"))?;
    }
    if target_path.exists() && !overwrite.unwrap_or(false) {
        return Err(format!("File already exists: {}", clean_relative.display()));
    }

    std::fs::write(&target_path, content)
        .map_err(|error| format!("Failed to write file: {error}"))?;
    Ok(clean_relative.to_string_lossy().into_owned())
}

#[tauri::command]
fn desktop_import_workspace_files(
    workspace_path: String,
    files: Vec<WorkspaceFileImportRequest>,
) -> Result<Vec<String>, String> {
    let workspace_root = PathBuf::from(workspace_path.trim());
    if workspace_root.as_os_str().is_empty() {
        return Err("Workspace path is required.".to_string());
    }
    if !workspace_root.exists() || !workspace_root.is_dir() {
        return Err(format!(
            "Workspace directory is unavailable: {}",
            workspace_root.display()
        ));
    }
    if files.is_empty() {
        return Err("At least one file is required.".to_string());
    }

    let mut written_paths = Vec::with_capacity(files.len());
    for file in files {
        let clean_relative = sanitize_workspace_relative_path(&file.relative_path)?;
        let target_path = workspace_root.join(&clean_relative);
        if let Some(parent) = target_path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|error| format!("Failed to prepare parent directories: {error}"))?;
        }
        if target_path.exists() && !file.overwrite.unwrap_or(false) {
            return Err(format!("File already exists: {}", clean_relative.display()));
        }
        std::fs::write(&target_path, &file.bytes).map_err(|error| {
            format!(
                "Failed to import file {}: {error}",
                clean_relative.display()
            )
        })?;
        written_paths.push(clean_relative.to_string_lossy().into_owned());
    }

    Ok(written_paths)
}

#[tauri::command]
fn desktop_open_workspace_file(
    workspace_path: String,
    relative_path: String,
) -> Result<(), String> {
    let target = resolve_workspace_target(&workspace_path, &relative_path)?;
    if !target.exists() || !target.is_file() {
        return Err(format!(
            "Workspace file is unavailable: {}",
            target.display()
        ));
    }
    open_path_with_system(&target)
}

#[tauri::command]
fn desktop_reveal_workspace_file(
    workspace_path: String,
    relative_path: String,
) -> Result<(), String> {
    let target = resolve_workspace_target(&workspace_path, &relative_path)?;
    if !target.exists() {
        return Err(format!(
            "Workspace path is unavailable: {}",
            target.display()
        ));
    }
    reveal_path_with_system(&target)
}

fn main() {
    let app = tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .manage(SidecarState::default())
        .setup(|app| {
            let state = app.state::<SidecarState>();
            state
                .initialize(app.handle())
                .map_err(std::io::Error::other)?;
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            desktop_bootstrap,
            desktop_sidecar_status,
            desktop_stop_sidecar,
            desktop_create_workspace_directory,
            desktop_create_workspace_file,
            desktop_import_workspace_files,
            desktop_open_workspace_file,
            desktop_reveal_workspace_file
        ])
        .build(tauri::generate_context!())
        .expect("failed to build Loom desktop shell");

    app.run(|app_handle, event| match event {
        tauri::RunEvent::Exit => {
            let state = app_handle.state::<SidecarState>();
            let _ = state.shutdown_for_exit();
        }
        tauri::RunEvent::ExitRequested { .. } => {
            let state = app_handle.state::<SidecarState>();
            let _ = state.shutdown_for_exit();
        }
        _ => {}
    });
}

#[cfg(test)]
mod tests {
    use super::{resolve_workspace_target, sanitize_workspace_relative_path};
    use std::path::PathBuf;

    #[test]
    fn accepts_normal_relative_paths() {
        assert_eq!(
            sanitize_workspace_relative_path("notes/todo.md").unwrap(),
            PathBuf::from("notes").join("todo.md")
        );
        assert_eq!(
            sanitize_workspace_relative_path("./README.md").unwrap(),
            PathBuf::from("README.md")
        );
    }

    #[test]
    fn rejects_empty_or_escaping_paths() {
        assert!(sanitize_workspace_relative_path("").is_err());
        assert!(sanitize_workspace_relative_path("../secret.txt").is_err());
        assert!(sanitize_workspace_relative_path("/tmp/secret.txt").is_err());
    }

    #[test]
    fn resolves_workspace_target_inside_workspace() {
        let temp = std::env::temp_dir().join(format!("loom-desktop-test-{}", std::process::id()));
        std::fs::create_dir_all(&temp).unwrap();
        let resolved =
            resolve_workspace_target(temp.to_string_lossy().as_ref(), "docs/readme.md").unwrap();
        assert_eq!(resolved, temp.join("docs").join("readme.md"));
        let _ = std::fs::remove_dir_all(&temp);
    }
}
