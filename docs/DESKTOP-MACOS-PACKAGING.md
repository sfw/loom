# macOS Desktop Packaging

This document covers the current packaged-product path for Loom Desktop on macOS.

## Build Inputs

- Desktop scope is macOS-only for this packaging path.
- Bundled Python request is `3.11.14`.
- CI pins `uv` to `0.10.12` for the packaged desktop build path.
- The bundled environment is assembled from `uv.lock` plus a locally built Loom wheel.
- The packaged desktop build installs every non-`dev` extra declared in
  `pyproject.toml` (currently `browser`, `mcp`, `pdf`, and `treesitter`).
- Runtime state stays outside the signed bundle under the app-data directory.

## Source Layout

- App bundle icons live in [`apps/desktop/src-tauri/icons`](/Users/sfw/Development/loom/apps/desktop/src-tauri/icons).
- Canonical source icons live in [`apps/desktop/src-tauri/icons/source`](/Users/sfw/Development/loom/apps/desktop/src-tauri/icons/source).
- Refresh generated icons with:

```bash
pnpm --dir apps/desktop icons:sync
```

## Build Commands

Build a packaged macOS app bundle locally:

```bash
pnpm --dir apps/desktop exec tauri build --bundles app --ci --no-sign
```

Build the full local `.app` + `.dmg` package path:

```bash
pnpm --dir apps/desktop package:macos
```

The `beforeBuildCommand` will:

1. regenerate app icons from `src-tauri/icons/source/`
2. build the desktop frontend
3. assemble bundled Python resources under `src-tauri/target/macos-bundle-resources/`

## GitHub Releases

Tagged desktop releases are built by
[`release-desktop.yml`](/Users/sfw/Development/loom/.github/workflows/release-desktop.yml).

- Push a tag in the form `desktop-vX.Y.Z`.
- The tag must match the version in:
  - [`apps/desktop/src-tauri/tauri.conf.json`](/Users/sfw/Development/loom/apps/desktop/src-tauri/tauri.conf.json)
  - [`apps/desktop/src-tauri/Cargo.toml`](/Users/sfw/Development/loom/apps/desktop/src-tauri/Cargo.toml)
  - [`apps/desktop/package.json`](/Users/sfw/Development/loom/apps/desktop/package.json)
- The workflow runs desktop tests, builds a signed `.app`, smoke-tests the
  packaged sidecar, creates a release DMG with
  [`build_macos_desktop_dmg.sh`](/Users/sfw/Development/loom/scripts/build_macos_desktop_dmg.sh),
  notarizes it, and uploads the DMG to the GitHub Release for that tag.

Release secret setup:

- Required signing secrets:
  - `APPLE_CERTIFICATE`
  - `APPLE_CERTIFICATE_PASSWORD`
  - `APPLE_SIGNING_IDENTITY`
- Required notarization secrets, choose one authentication path:
  - App Store Connect API:
    - `APPLE_API_KEY`
    - `APPLE_API_ISSUER`
    - `APPLE_API_KEY_P8`
  - Apple ID:
    - `APPLE_ID`
    - `APPLE_PASSWORD`
    - `APPLE_TEAM_ID`
- Optional if your Apple account spans multiple providers:
  - `APPLE_PROVIDER_SHORT_NAME`

`APPLE_API_KEY_P8` is a repository secret containing the raw `.p8` private key
contents. The workflow writes it to a temporary file and exports
`APPLE_API_KEY_PATH` for Tauri during the build.

## Bundle Layout

Inside `Loom Desktop.app/Contents/Resources/`:

- `python/`: bundled private Python runtime
- `loom-env/`: locked Loom environment and dependencies
- `loom-desktop-bundle.json`: bundle manifest with Loom/Python/uv metadata

Writable runtime state stays outside the bundle:

- `runtime/loomd.db`
- `runtime/scratch/`
- `runtime/python-cache/`
- `runtime/desktop.instance.json`
- `runtime/loomd.sidecar.json`
- `logs/loomd.log`

## Smoke Check

Smoke-test a built bundle without starting the UI shell:

```bash
env UV_CACHE_DIR=.uv-cache \
  uv run python scripts/smoke_macos_desktop_bundle.py \
  --app-bundle "apps/desktop/src-tauri/target/release/bundle/macos/Loom Desktop.app"
```

This verifies that the packaged app bundle contains a runnable bundled runtime and that `loomd` can start from those bundled assets.

## Support Diagnostics

The bundled runtime manifest records:

- enabled non-`dev` extras
- Loom version
- bundled Python version
- requested Python version
- uv version used during assembly
- bundle-relative runtime paths

The desktop Tauri command `desktop_sidecar_status` also exposes runtime metadata and any packaged-runtime resolution error for support/debug tooling.

## Manual Acceptance Checklist

Run this on a macOS machine that does not rely on a source checkout.

- Install the packaged `.app` on a machine without user-installed Python or `uv`.
- Launch the app and confirm the window opens normally.
- Confirm bundled `loomd` starts and `/runtime` becomes healthy.
- Create or open a workspace and confirm workspace registration succeeds.
- Verify conversation and run flows complete against the bundled backend.
- Quit and relaunch the app, then confirm state persisted across restarts.
- Inspect the app-data log and confirm startup failures, if any, are user-diagnosable.

## Signing and Notarization

Signing is not optional release glue for this app shape because the bundle contains an embedded interpreter and Python-native libraries.

Release validation should include:

- `codesign --verify --deep --strict "Loom Desktop.app"`
- notarization submission for the built app or DMG
- a post-signing launch smoke test to confirm the bundled sidecar still starts

CI currently builds with `--no-sign` because signing credentials are release infrastructure, not repository state. Treat signing/notarization as a required release gate for macOS distribution.
