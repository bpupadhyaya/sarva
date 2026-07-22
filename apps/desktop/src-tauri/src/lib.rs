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
//!
//! `on_window_event`'s `CloseRequested` handler only fires for a graceful
//! window close — it does not run when the app process is killed directly
//! (SIGTERM/SIGINT), which orphaned the sidecar in earlier testing. The
//! `#[cfg(unix)]` signal handler below closes that gap on macOS/Linux by
//! killing the sidecar itself before the process exits. Windows has no
//! equivalent yet (untested platform — see BUILD-JOURNAL.md).
//!
//! Killing the sidecar isn't just `child.kill()`, either: PyInstaller's
//! `--onefile` bootloader is the process we spawn, but it forks a second
//! process to run the actual frozen app and waits on it — confirmed with
//! `ps -o pid,ppid,pgid` while the sidecar was running. `child.kill()`
//! only reaps the bootloader; the real server (the grandchild) is
//! untouched and keeps holding the port. `kill_sidecar` below also kills
//! any direct children of the sidecar PID via `pgrep -P` before killing
//! the sidecar itself, on both shutdown paths.

use std::sync::Mutex;
use tauri::Manager;
use tauri_plugin_shell::{process::CommandChild, process::CommandEvent, ShellExt};

struct SidecarHandle(Mutex<Option<CommandChild>>);

fn kill_sidecar(child: CommandChild) {
    #[cfg(unix)]
    {
        let pid = child.pid();
        if let Ok(output) = std::process::Command::new("pgrep")
            .arg("-P")
            .arg(pid.to_string())
            .output()
        {
            for line in String::from_utf8_lossy(&output.stdout).lines() {
                if let Ok(grandchild_pid) = line.trim().parse::<u32>() {
                    let _ = std::process::Command::new("kill")
                        .args(["-9", &grandchild_pid.to_string()])
                        .status();
                }
            }
        }
    }
    let _ = child.kill();
}

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

            #[cfg(unix)]
            {
                let app_handle = app.handle().clone();
                std::thread::spawn(move || {
                    use signal_hook::consts::{SIGINT, SIGTERM};
                    use signal_hook::iterator::Signals;

                    let mut signals = Signals::new([SIGINT, SIGTERM])
                        .expect("failed to register SIGINT/SIGTERM handler");
                    if signals.forever().next().is_some() {
                        log::warn!("received termination signal, killing sarva-server sidecar");
                        if let Some(state) = app_handle.try_state::<SidecarHandle>() {
                            if let Some(child) = state.0.lock().unwrap().take() {
                                kill_sidecar(child);
                            }
                        }
                        std::process::exit(0);
                    }
                });
            }

            Ok(())
        })
        .on_window_event(|window, event| {
            if let tauri::WindowEvent::CloseRequested { .. } = event {
                let handle = window.app_handle().state::<SidecarHandle>();
                let child = handle.0.lock().unwrap().take();
                if let Some(child) = child {
                    kill_sidecar(child);
                }
            }
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
