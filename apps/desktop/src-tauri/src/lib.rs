//! Sarva desktop — a native window wrapper around the Sarva web UI.
//!
//! T4 step 1 (current): `tauri.conf.json`'s `frontendDist`/`devUrl` point
//! directly at a running `sarva serve` backend (default
//! `http://127.0.0.1:8000`) — this app does not bundle the Python engine
//! or start it automatically. That means today it's a native window, not
//! yet the one-click, no-terminal experience the design calls for; run
//! `sarva serve` yourself first, then launch this app.
//!
//! T4 step 2 (not built yet): bundle a Python runtime as a Tauri sidecar
//! and spawn it from `run()` below, so double-clicking the app is the
//! entire install. See BUILD-JOURNAL.md for why this is a separate,
//! larger piece of work rather than rushed into this first slice.

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .setup(|app| {
            if cfg!(debug_assertions) {
                app.handle().plugin(
                    tauri_plugin_log::Builder::default()
                        .level(log::LevelFilter::Info)
                        .build(),
                )?;
            }
            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
