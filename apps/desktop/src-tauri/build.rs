use std::env;
use std::fs;
use std::path::PathBuf;

const STUB_MANIFEST: &str = r#"{
  "schema_version": 1,
  "python_version": "dev-stub",
  "python_home_relative_path": "python",
  "python_executable_relative_path": "python/bin/python3.11",
  "environment_root_relative_path": "loom-env",
  "site_packages_relative_path": "loom-env/lib/python3.11/site-packages",
  "entry_module": "loom.daemon.cli"
}
"#;

fn ensure_debug_resource_stub() {
    let profile = env::var("PROFILE").unwrap_or_default();
    if profile == "release" {
        return;
    }

    let manifest_dir =
        PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR is required"));
    let resources_root = manifest_dir.join("target/macos-bundle-resources");
    let manifest_path = resources_root.join("loom-desktop-bundle.json");
    if manifest_path.exists() {
        return;
    }

    fs::create_dir_all(resources_root.join("python/bin"))
        .expect("failed to create debug stub python resource dir");
    fs::create_dir_all(resources_root.join("loom-env/lib/python3.11/site-packages"))
        .expect("failed to create debug stub environment resource dir");
    fs::write(&manifest_path, STUB_MANIFEST).expect("failed to write debug stub manifest");
}

fn main() {
    println!("cargo:rerun-if-changed=icons/icon.icns");
    println!("cargo:rerun-if-changed=icons/icon.png");
    println!("cargo:rerun-if-changed=icons/source/Icon-iOS-Default-1024x1024@1x.png");
    println!("cargo:rerun-if-changed=icons/source/loom-iOS-Default-1024x1024@2x.png");
    ensure_debug_resource_stub();
    tauri_build::build()
}
