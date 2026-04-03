# macOS Desktop Bundled Python Plan (2026-04-02)

## Objective
Ship the Loom desktop app as a real macOS application that does not require end users to install Python or `uv`, by bundling a private Python runtime plus a locked Loom environment inside the app bundle.

## Decision
For the first desktop packaging wave, Loom should ship:
1. a Tauri macOS app bundle
2. a private embedded Python runtime inside the bundle
3. a bundled Loom Python environment built from the repo's locked dependencies
4. a desktop sidecar launcher that executes the bundled interpreter directly instead of `uv run loomd`

We are explicitly choosing this over a frozen one-file/one-binary Python sidecar for now because it is the lower-risk path from the current architecture to a shippable desktop product.

## Why This Path
The current desktop shell still assumes:
1. a source checkout exists
2. `uv` exists on the host machine
3. `loomd` is started as `uv run loomd`

That is acceptable for development, but not for a desktop product. A bundled Python runtime preserves the existing Python backend with the least architectural churn while removing the user-facing Python dependency.

## Goals
1. A macOS user can install and launch Loom Desktop without separately installing Python or `uv`.
2. The app launches a bundled `loomd` sidecar from inside the app bundle.
3. The bundled sidecar runs against a desktop-owned writable runtime area under app data, not inside the read-only app bundle.
4. Packaging remains reproducible from the monorepo with CI-verifiable build steps.
5. The approach remains compatible with ongoing Python backend development.
6. Logging, crash diagnosis, and local support remain practical.

## Non-Goals
1. Cross-platform desktop packaging in this phase.
2. Rewriting `loomd` in Rust.
3. Freezing the Python backend into a standalone binary in this phase.
4. Sandboxing the backend beyond current local desktop assumptions.
5. Solving optional browser/tooling dependency packaging in one pass unless required for default desktop flows.

## Product Constraints
1. Desktop scope is macOS-only for now.
2. The app should remain local-first and sidecar-based.
3. The sidecar must keep using desktop-owned writable locations for:
   - SQLite database
   - scratch directory
   - logs
   - lease/ownership metadata
4. The shipped desktop package must not depend on a live source tree.

## Current State
Today the desktop app:
1. locates the repo root from `CARGO_MANIFEST_DIR`
2. starts the sidecar with `uv run loomd`
3. assumes `pyproject.toml` and `src/loom` exist near the built app
4. bundles the frontend, but not a Python runtime or Loom environment

This means the current desktop app is still a development shell, not a self-contained product.

## Packaging Model

### App Bundle Layout
Target the following conceptual layout inside `Loom Desktop.app`:
1. `Contents/MacOS/Loom Desktop`
   - Tauri host executable
2. `Contents/Resources/python/`
   - bundled Python runtime
3. `Contents/Resources/loom-env/`
   - installed Loom environment and dependencies
4. `Contents/Resources/loom/`
   - app-owned static metadata/resources only if needed
5. `Contents/Resources/bin/`
   - optional helper launch scripts if direct interpreter invocation proves awkward

The writable runtime remains outside the bundle under the macOS app-data path already used by the desktop sidecar.

### Sidecar Launch Contract
The desktop shell should launch the sidecar using the bundled interpreter and installed package, not `uv`.

Preferred direction:
1. invoke bundled Python directly
2. run the installed `loom.daemon.cli` module or a bundled entrypoint script
3. pass the same runtime overrides already used today:
   - host
   - port
   - database path
   - scratch dir
   - workspace default path
   - desktop ownership token/lease paths

The launcher must not depend on:
1. `PATH` containing Python
2. `PATH` containing `uv`
3. the current working directory being the repo root

## High-Level Design

### 1. Build a Private Desktop Python Environment
At build time we create a relocatable packaged environment for the app bundle.

Desired properties:
1. derived from `uv.lock`
2. reproducible in CI
3. excludes clearly non-desktop extras unless intentionally included
4. can be copied into `Contents/Resources`

We should treat this environment as product build output, not as the developer's local `.venv`.

### 2. Install Loom Into That Environment
The bundled environment should contain:
1. Loom itself
2. required runtime dependencies from `pyproject.toml`
3. desktop-needed optional extras only if necessary

Initial bias:
1. include the base runtime only
2. do not automatically include heavy optional extras such as Playwright unless the desktop product truly requires them on first-class flows

### 3. Launch `loomd` From the Bundle
Replace the current `find_repo_root()` plus `uv run loomd` startup path with resource-relative startup.

The Rust sidecar code should:
1. resolve the app bundle resource directory from Tauri
2. locate bundled Python/interpreter assets there
3. construct a stable command line to launch the daemon
4. preserve the current process-group and lease semantics

### 4. Keep Runtime State Writable and External
The bundled environment is read-only product content.

Runtime state stays under app data:
1. `runtime/loomd.db`
2. `runtime/scratch/`
3. `logs/loomd.log`
4. desktop lease and sidecar state files

No database, cache, log, or temporary state should be written into the app bundle.

### 5. Make Packaging a First-Class CI Artifact
This cannot remain a local manual step.

We need CI/build automation that can:
1. build the frontend
2. build the Tauri shell
3. assemble the bundled Python environment
4. place assets into the app bundle
5. run smoke checks that the bundle can launch the sidecar

## Workstream A: Packaging Design and Bundle Layout

### Problem
We need a stable, inspectable structure for product assets inside the macOS app.

### Plan
1. Define the canonical app-bundle resource layout.
2. Decide whether Loom is invoked via:
   - `python -m loom.daemon.cli`
   - a generated wrapper script
   - an installed console-script entrypoint inside the bundled env
3. Decide whether the Python runtime and site-packages live in:
   - one combined environment directory
   - separate runtime + environment directories
4. Define how the Rust launcher discovers those resources using Tauri path APIs.

### Primary Files
1. `apps/desktop/src-tauri/src/sidecar.rs`
2. `apps/desktop/src-tauri/tauri.conf.json`
3. `apps/desktop/src-tauri/build.rs`
4. new packaging/build scripts under `scripts/` or `apps/desktop/`

### Acceptance
1. The bundle layout is documented and stable.
2. The sidecar launcher can resolve bundled assets without source-tree assumptions.

## Workstream B: Build Pipeline for Bundled Python

### Problem
We need a reproducible way to assemble the private Python runtime and Loom environment during builds.

### Plan
1. Add a build script that:
   - provisions a clean product-build Python environment
   - installs Loom and locked dependencies into it
   - copies the result into the Tauri bundle resources
2. Ensure the build uses `uv.lock` or another lock-respecting path so desktop builds are reproducible.
3. Make the build script explicit about included dependency groups.
4. Avoid relying on the developer's existing `.venv`.
5. Decide whether product packaging runs:
   - entirely from Tauri `beforeBuildCommand`
   - from a dedicated root script that Tauri consumes

### Open Decision
We need to choose one of:
1. build a dedicated product venv from the repo at package time
2. vendor/copy a prepared runtime artifact into the app bundle

Initial recommendation:
1. build a dedicated product environment in CI/local packaging
2. copy only the resulting runtime payload into the final app bundle

### Primary Files
1. `apps/desktop/package.json`
2. `apps/desktop/src-tauri/tauri.conf.json`
3. `pyproject.toml`
4. `uv.lock`
5. new packaging scripts

### Acceptance
1. A clean machine/CI job can assemble the desktop package from repo state alone.
2. The packaging path does not depend on the developer's local `.venv`.

## Workstream C: Rust Sidecar Launcher Refactor

### Problem
The current launcher is development-only.

### Plan
1. Remove the repo-root discovery requirement from the product path.
2. Replace `Command::new("uv") ... "run" ... "loomd"` with bundled-interpreter execution.
3. Preserve:
   - loopback host binding
   - dynamic port selection
   - runtime path overrides
   - desktop ownership leases
   - graceful shutdown semantics
4. Keep a development fallback only if explicitly needed and clearly separated from production behavior.
5. Make startup errors specific and user-readable:
   - missing bundled runtime
   - failed Python launch
   - invalid bundle layout

### Primary Files
1. `apps/desktop/src-tauri/src/sidecar.rs`
2. `apps/desktop/src-tauri/src/main.rs`

### Acceptance
1. A packaged app starts `loomd` without `uv`.
2. No product path depends on source checkout markers.

## Workstream D: Signing, Notarization, and macOS Runtime Hygiene

### Problem
Bundling an interpreter and many Python files changes the code-signing surface materially.

### Plan
1. Validate how bundled Python binaries, shared libraries, and helper files need to be signed.
2. Ensure the final app bundle can be signed/notarized with the embedded runtime included.
3. Ensure runtime writes only touch writable app-data locations, not signed bundle content.
4. Document any entitlements or packaging caveats discovered during signing/notarization work.

### Acceptance
1. The bundled app can be signed successfully.
2. The app can launch the sidecar post-signing without invalidating signatures.

## Workstream E: Testing and Verification

### Problem
We need confidence that the packaged app really behaves like a self-contained product.

### Plan
1. Add unit coverage for bundled-runtime path resolution in Rust where practical.
2. Add a packaging smoke test that verifies the launcher command is constructed from bundled resources.
3. Add CI coverage for desktop build/test paths.
4. Add a manual acceptance checklist for packaged-app verification on macOS:
   - fresh install on a machine without Python/`uv`
   - app first launch
   - sidecar startup
   - workspace registration
   - conversation/run flows
   - app restart with persisted state
5. Add a support-oriented diagnostic path for the desktop app to expose bundled runtime version/build metadata if needed.

### Primary Files
1. desktop Rust test files
2. desktop/frontend CI config
3. packaging scripts
4. release docs

### Acceptance
1. CI validates the desktop surface more fully.
2. A packaged app smoke run proves the bundle can self-start its backend.

## Workstream F: Development Workflow Separation

### Problem
We still need a fast developer workflow even after adding product packaging.

### Plan
1. Preserve a development mode where local Tauri dev can still use repo-local tooling.
2. Keep production packaging behavior separate from development startup behavior.
3. Avoid mixing "dev shell" assumptions into packaged-app code paths.
4. Document the two modes clearly:
   - development mode
   - packaged product mode

### Acceptance
1. Desktop contributors keep a fast local dev loop.
2. Production startup does not accidentally rely on development assumptions.

## Risks

### Risk 1: Bundle size growth
Bundling Python and site-packages will increase app size.

Mitigation:
1. start with required base runtime only
2. avoid including unnecessary extras
3. measure bundle size before adding optional tooling

### Risk 2: Packaging drift from main Python runtime
The desktop bundle could accidentally ship a different dependency set than normal Loom.

Mitigation:
1. derive from the lockfile
2. keep packaging scripted and reproducible
3. add version/build metadata for inspection

### Risk 3: macOS signing complexity
Embedded interpreters and native libs may complicate signing/notarization.

Mitigation:
1. prototype signing early
2. treat signing as part of the implementation track, not release-end glue work

### Risk 4: Hidden source-tree assumptions survive
Some code paths may still assume the repo layout.

Mitigation:
1. audit for `CARGO_MANIFEST_DIR`, `pyproject.toml`, repo-root discovery, and `uv` assumptions
2. add tests for bundled-resource resolution

## Open Questions
1. Should the bundled environment include browser-dependent extras in the first desktop release?
2. Should we invoke `python -m loom.daemon.cli` directly, or ship a tiny wrapper entrypoint inside the bundle?
3. Do we want one shared desktop product environment or a slimmer daemon-only environment?
4. Should packaged desktop builds be produced only on tagged releases, or on every macOS desktop CI run?
5. How much bundle-size increase is acceptable for the first macOS release?

## Recommended Rollout Order
1. Finalize bundle layout and launcher contract.
2. Prototype local macOS packaging with bundled Python assets.
3. Refactor `sidecar.rs` to launch the bundled interpreter.
4. Add signing/notarization validation.
5. Add packaged-app smoke verification.
6. Promote the result into release and CI workflows.

## Exit Criteria
This plan is complete when:
1. Loom Desktop can be installed and launched on macOS without Python or `uv` installed on the machine.
2. The app bundle launches its own `loomd` sidecar from bundled assets.
3. The sidecar uses external writable runtime paths for DB/log/scratch state.
4. The packaged app is buildable reproducibly from the repo.
5. Desktop packaging and launch behavior are covered by CI and release documentation.
