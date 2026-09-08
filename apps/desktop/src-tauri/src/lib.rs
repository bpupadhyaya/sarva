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
//! holds the port. That claim is only actually true in every build now,
//! not just debug ones: `tauri_plugin_log`'s registration used to be
//! gated behind `cfg!(debug_assertions)`, so a real release build had no
//! logger at all and every `log::info!`/`log::warn!` call below —
//! including the one place a sidecar crash (`CommandEvent::Terminated`)
//! is ever noticed — was a silent no-op, confirmed directly against the
//! `log` crate's own documented behavior with no logger registered
//! (`log::max_level()` is `Off`). Registered unconditionally now — see
//! `run()`'s own comment for the rest. See BUILD-JOURNAL.md for what's
//! verified vs. not yet (cross-platform release bundles, code signing).
//!
//! `on_window_event`'s `CloseRequested` handler only fires for a graceful
//! window close — it does not run when the app process is killed directly
//! (SIGTERM/SIGINT), which orphaned the sidecar in earlier testing. The
//! `#[cfg(unix)]` signal handler below closes that gap on macOS/Linux by
//! killing the sidecar itself before the process exits.
//!
//! **Windows still has no equivalent for that specific gap, for a real,
//! checked reason, not just "untested":** `main.rs` sets
//! `windows_subsystem = "windows"` for release builds (required to avoid
//! popping a console window), and Win32's console-control-handler API
//! (`SetConsoleCtrlHandler`, the nearest equivalent to `signal-hook`'s
//! SIGINT/SIGTERM interception) only delivers events to a process that
//! actually has a console attached — a windows-subsystem GUI app doesn't.
//! Catching a genuine forceful-but-not-instant Windows shutdown/logoff
//! would need deeper Win32 message-loop hooking (`WM_QUERYENDSESSION`)
//! this project hasn't built and can't verify here (this dev environment
//! has no Windows machine — confirmed real, not assumed, since even the
//! *rest* of this file's Windows compile-correctness is verified only via
//! CI's `windows-latest` `cargo check` job, never a Windows runtime).
//!
//! **What Windows genuinely was missing until now, and a real bug, not
//! just an untested corner:** even the ordinary graceful-close path
//! (`on_window_event`'s `CloseRequested`, which already fires identically
//! on every platform) called `kill_sidecar`, whose grandchild-reaping
//! logic was unconditionally `#[cfg(unix)]`-gated — meaning a plain
//! window close on Windows only ever killed the PyInstaller bootloader,
//! silently orphaning the real frozen server process holding the port,
//! on the ONE shutdown path Windows already exercises. Fixed below with
//! a `#[cfg(windows)]` branch using `taskkill /F /T /PID` — Windows' own
//! native process-tree kill, simpler than Unix's `pgrep -P` + `kill`
//! loop since `/T` already recurses through every descendant.
//!
//! Killing the sidecar isn't just `child.kill()`, either: PyInstaller's
//! `--onefile` bootloader is the process we spawn, but it forks a second
//! process to run the actual frozen app and waits on it — confirmed with
//! `ps -o pid,ppid,pgid` while the sidecar was running. `child.kill()`
//! only reaps the bootloader; the real server (the grandchild) is
//! untouched and keeps holding the port. `kill_sidecar` below also kills
//! any descendant of the sidecar PID (via `pgrep -P` on Unix, via
//! `taskkill /T` on Windows) before killing the sidecar itself, on both
//! shutdown paths this app has.

use std::sync::Mutex;
use tauri::Manager;
use tauri_plugin_dialog::{DialogExt, MessageDialogKind};
use tauri_plugin_shell::{process::CommandChild, process::CommandEvent, ShellExt};

struct SidecarHandle(Mutex<Option<CommandChild>>);

// A real gap found by actually occupying port 8000 with an unrelated local
// server (not another `sarva serve` -- this module's own doc comment above
// already, deliberately, accepts that friendlier case: the UI just ends up
// talking to the pre-existing compatible instance) before launching `tauri
// dev`: the sidecar fails to bind and exits in well under a second, exactly
// as documented, but the WebView still successfully connects to port 8000 --
// there IS something listening, just not anything that serves the real API
// or a working bundle. Confirmed live with a plain `python3 -m http.server
// 8000` standing in for "some other unrelated dev tool defaulted to the
// same port" (extremely plausible: 8000 is Python's own `http.server`
// default):
// the window loaded a raw, un-transpiled `index.html` referencing
// `/src/main.tsx`, which a browser can't execute, leaving a permanently
// blank white window with zero indication anything went wrong -- the
// `CommandEvent::Terminated` branch below only ever reached `log::warn!`,
// invisible to anyone who didn't already know to open the app's log file.
// A real HTTP health check (not just "did the sidecar's own exit code look
// bad") is what actually distinguishes the two cases without regressing the
// documented friendly-coexistence one: if something on port 8000 answers
// `/health` correctly, the UI is fine and this stays silent; only an actual
// dead end shows the user a real, actionable native dialog instead of an
// unexplained blank window.
fn sarva_health_check_ok() -> bool {
    sarva_health_check_at("127.0.0.1:8000")
}

// Takes the address as a parameter (rather than hardcoding it inline) purely
// so the test below can point it at a throwaway loopback listener instead of
// the real port 8000 -- a live end-to-end check against 8000 itself risks
// colliding with whatever else might already be bound there on a real
// machine, the exact failure mode this function exists to detect in the
// first place.
fn sarva_health_check_at(addr: &str) -> bool {
    use std::io::{Read, Write};
    use std::net::TcpStream;
    use std::time::Duration;

    let Ok(mut stream) = TcpStream::connect(addr) else {
        return false;
    };
    let _ = stream.set_read_timeout(Some(Duration::from_secs(2)));
    let _ = stream.set_write_timeout(Some(Duration::from_secs(2)));
    let request = "GET /health HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n";
    if stream.write_all(request.as_bytes()).is_err() {
        return false;
    }
    let mut response = String::new();
    if stream.read_to_string(&mut response).is_err() {
        return false;
    }
    response.starts_with("HTTP/1.1 200") && response.contains("\"status\":\"ok\"")
}

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
    #[cfg(windows)]
    {
        // `/T` kills the whole process tree (the bootloader AND the
        // frozen-server grandchild it forks), `/F` forces it -- one
        // native command does what Unix needs a pgrep-then-kill loop
        // for. Best-effort: if the bootloader already exited (e.g. it
        // crashed on its own), `taskkill` fails harmlessly and
        // `child.kill()` below is still attempted.
        let _ = std::process::Command::new("taskkill")
            .args(["/F", "/T", "/PID", &child.pid().to_string()])
            .status();
    }
    let _ = child.kill();
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_dialog::init())
        .setup(|app| {
            // A real bug found by actually checking what happens with no
            // logger registered (a standalone `log`-crate repro, not just
            // reading the plugin's source): `log::max_level()` is `Off`
            // and every `log::info!`/`log::warn!` call below is a silent
            // no-op -- confirmed directly, not assumed. Gating this
            // registration on `cfg!(debug_assertions)` (as originally
            // written) meant every sidecar stdout/stderr line AND the one
            // place a sidecar crash is ever noticed
            // (`CommandEvent::Terminated`) vanished with no record at all
            // in a real release build -- the module doc comment's own
            // claim that "the failure surfaces as a log line from the
            // sidecar process" was only true in debug builds. Registered
            // unconditionally now; `Builder::default()`'s own default
            // targets already write to both stdout AND a real log file in
            // the platform's app-log directory, so this needs no extra
            // target configuration to fix the release-build blind spot.
            app.handle().plugin(
                tauri_plugin_log::Builder::default()
                    .level(log::LevelFilter::Info)
                    .build(),
            )?;

            // A real bug found by giving the CLI's own equivalent gap
            // (core/sarva/cli.py's `serve` command, see BUILD-JOURNAL.md)
            // a fresh-eyes sweep here too: with no `--workdir` passed,
            // the sidecar's file/shell tools defaulted to whatever
            // directory the OS happened to launch this app process from
            // -- for a one-click desktop app (this is the one surface
            // whose entire T4 definition of done is "no terminal," so
            // there's no `cd` for a user to have gotten right even in
            // principle), that's not a deliberate, chosen boundary at
            // all, just ambient OS behavior a non-developer has no way
            // to reason about or control. Fixed by resolving a
            // dedicated, per-app workspace directory (Tauri's own
            // `app_data_dir()`, the platform-appropriate location every
            // other well-behaved desktop app already uses for its own
            // files) and passing it explicitly via `--workdir` -- the
            // same flag `sarva serve`'s own fix just added, reused here
            // rather than duplicated, so both surfaces share one real
            // fix instead of the desktop app needing its own separate
            // one later.
            let workspace_dir = app
                .path()
                .app_data_dir()
                .expect("could not resolve the app data directory")
                .join("workspace");
            std::fs::create_dir_all(&workspace_dir)
                .expect("could not create the sarva-server workspace directory");
            let workspace_dir_str = workspace_dir
                .to_str()
                .expect("workspace directory path is not valid UTF-8")
                .to_string();

            let (mut rx, child) = app
                .shell()
                .sidecar("sarva-server")
                .expect("sarva-server sidecar not found — run scripts/freeze-server.sh first")
                .args(["serve", "--workdir", &workspace_dir_str])
                .spawn()
                .expect("failed to spawn sarva-server sidecar");

            // Managed here, before the sidecar-event task below, rather than
            // after it: the Terminated branch reads this same state to tell
            // an unexpected crash apart from our own intentional shutdown
            // (`kill_sidecar`'s callers `.take()` it right before killing),
            // so it must already be `Some` by the time any real event can
            // possibly arrive.
            app.manage(SidecarHandle(Mutex::new(Some(child))));

            let dialog_app_handle = app.handle().clone();
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
                            // An intentional shutdown (window close, SIGINT/SIGTERM)
                            // already `.take()`s the handle right before calling
                            // `kill_sidecar` -- if it's gone, this Terminated event
                            // is that expected kill, not a real crash, and a
                            // health check would obviously fail (we just killed
                            // it) and show a false-alarm dialog on ordinary quit.
                            let intentional_shutdown = dialog_app_handle
                                .try_state::<SidecarHandle>()
                                .map(|s| s.0.lock().unwrap().is_none())
                                .unwrap_or(false);
                            if !intentional_shutdown && !sarva_health_check_ok() {
                                // `.show()`, not `.blocking_show()`: the
                                // plugin's own simplest documented form,
                                // callable directly with no extra thread of
                                // our own -- there's no result here worth
                                // blocking on, just a fire-and-forget
                                // notification.
                                dialog_app_handle
                                    .dialog()
                                    .message(
                                        "Sarva's local backend did not start correctly \
                                         (it may have crashed, or port 8000 is already \
                                         used by something else). The window may not \
                                         work until this is resolved -- free up port \
                                         8000 or check the app log, then restart Sarva.",
                                    )
                                    .title("Sarva backend unavailable")
                                    .kind(MessageDialogKind::Error)
                                    .show(|_| {});
                            }
                        }
                        _ => {}
                    }
                }
            });

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

#[cfg(test)]
mod tests {
    use super::sarva_health_check_at;
    use std::io::{Read, Write};
    use std::net::TcpListener;

    // Each test binds its own ephemeral port (`:0`, OS-assigned, never 8000)
    // specifically to avoid the real collision this function was written to
    // detect -- confirmed live once already that testing against the real
    // port 8000 is not safe to repeat on a machine that might have something
    // else genuinely running there.
    fn serve_once(response: &'static str) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap().to_string();
        std::thread::spawn(move || {
            if let Ok((mut stream, _)) = listener.accept() {
                let mut buf = [0u8; 1024];
                let _ = stream.read(&mut buf);
                let _ = stream.write_all(response.as_bytes());
            }
        });
        addr
    }

    #[test]
    fn a_real_sarva_health_endpoint_is_recognized_as_healthy() {
        let addr = serve_once(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nConnection: close\r\n\r\n\
             {\"status\":\"ok\"}",
        );
        assert!(sarva_health_check_at(&addr));
    }

    #[test]
    fn an_unrelated_service_on_the_same_port_is_recognized_as_unhealthy() {
        // The exact shape of the real bug: some other, unrelated local
        // service (a plain static file server, in the live repro) answers
        // on the port instead of a real sarva-server, and its response
        // doesn't match the health check.
        let addr = serve_once(
            "HTTP/1.0 404 File not found\r\nConnection: close\r\n\r\n<html>not found</html>",
        );
        assert!(!sarva_health_check_at(&addr));
    }

    #[test]
    fn nothing_listening_at_all_is_recognized_as_unhealthy() {
        // Port 0 never accepts a real connection -- the plain "nothing
        // answered" case, e.g. the sidecar crashed and no other process
        // happens to be squatting on the port either.
        assert!(!sarva_health_check_at("127.0.0.1:0"));
    }
}
