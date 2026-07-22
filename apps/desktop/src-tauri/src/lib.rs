//! Sarva desktop — a native window wrapper around the Sarva web UI.
//!
//! T4 step 2 (current): the Python backend is frozen (`scripts/freeze-server.sh`,
//! PyInstaller `--onefile`) and bundled as a Tauri sidecar
//! (`tauri.conf.json`'s `bundle.externalBin`). `run()` below spawns it on
//! startup and kills it when the window closes, so double-clicking the app
//! is the entire experience — no terminal, no manual `sarva serve` step.
//! `frontendDist`/`devUrl` still point at `http://127.0.0.1:8000`, which is
//! now the sidecar's own default port rather than a developer's manual one.
//!
//! If the sidecar fails to bind that port (e.g. a `sarva serve` from the
//! README quickstart is already running there), the spawn itself still
//! succeeds — the failure surfaces as a log line from the sidecar process,
//! and the UI transparently ends up talking to whichever process actually
//! holds the port. See BUILD-JOURNAL.md for what's verified vs. not yet
//! (cross-platform release bundles, code signing).

use std::sync::Mutex;
use tauri::Manager;
use tauri_plugin_shell::{process::CommandChild, process::CommandEvent, ShellExt};

struct SidecarHandle(Mutex<Option<CommandChild>>);

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .setup(|app| {
            if cfg!(debug_assertions) {
                app.handle().plugin(
                    tauri_plugin_log::Builder::default()
                        .level(log::LevelFilter::Info)
                        .build(),
                )?;
            }

            let (mut rx, child) = app
                .shell()
                .sidecar("sarva-server")
                .expect("sarva-server sidecar not found — run scripts/freeze-server.sh first")
                .args(["serve"])
                .spawn()
                .expect("failed to spawn sarva-server sidecar");

            tauri::async_runtime::spawn(async move {
                while let Some(event) = rx.recv().await {
                    match event {
                        CommandEvent::Stdout(line) => {
                            log::info!("sarva-server: {}", String::from_utf8_lossy(&line));
                        }
                        CommandEvent::Stderr(line) => {
                            log::warn!("sarva-server: {}", String::from_utf8_lossy(&line));
                        }
                        CommandEvent::Terminated(payload) => {
                            log::warn!("sarva-server exited: {:?}", payload.code);
                        }
                        _ => {}
                    }
                }
            });

            app.manage(SidecarHandle(Mutex::new(Some(child))));
            Ok(())
        })
        .on_window_event(|window, event| {
            if let tauri::WindowEvent::CloseRequested { .. } = event {
                let handle = window.app_handle().state::<SidecarHandle>();
                let child = handle.0.lock().unwrap().take();
                if let Some(child) = child {
                    let _ = child.kill();
                }
            }
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
