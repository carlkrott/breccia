//! bash, process, execute_code. Each command runs in its own process group with output
//! going to a log file (no pipe deadlocks). `process kill` signals only groups started
//! here and not yet reaped; remaining background groups are killed when the session ends.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::os::unix::process::CommandExt;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use serde_json::json;

use crate::{opt_b, opt_s, opt_u, s, Args, Ctx, Gate, Tool};

pub fn tools() -> Vec<Tool> {
    vec![
        Tool {
            name: "bash",
            description: "Run a shell command (bash -c) in the workdir. Foreground commands time out (default 120 s, max 600). background=true returns a process id for the process tool (servers, watchers, long builds).",
            schema: || json!({"type": "object", "properties": {
                "command": {"type": "string"},
                "timeout": {"type": "integer", "description": "Seconds (foreground only)"},
                "workdir": {"type": "string", "description": "Directory inside the workdir"},
                "background": {"type": "boolean"}}, "required": ["command"]}),
            gate: Gate::Exec,
            keys: &[("cmd", "command"), ("script", "command"), ("timeout_secs", "timeout"), ("timeout_seconds", "timeout"), ("cwd", "workdir"),
                    ("directory", "workdir"), ("run_in_background", "background"), ("is_background", "background")],
            run: bash,
        },
        Tool {
            name: "process",
            description: "Manage background commands: action list | poll (status + new output) | log (offset/limit lines) | wait (until exit or timeout) | kill | write (send text to stdin).",
            schema: || json!({"type": "object", "properties": {
                "action": {"type": "string", "enum": ["list", "poll", "log", "wait", "kill", "write"]},
                "id": {"type": "integer"}, "input": {"type": "string"},
                "timeout": {"type": "integer"}, "offset": {"type": "integer"}, "limit": {"type": "integer"}}, "required": ["action"]}),
            gate: Gate::Exec,
            keys: &[("process_id", "id"), ("session_id", "id"), ("pid", "id"), ("shell_id", "id"), ("bash_id", "id"), ("data", "input"),
                    ("text", "input"), ("op", "action"), ("command", "action")],
            run: process,
        },
        Tool {
            name: "execute_code",
            description: "Run a Python 3 script (python3 -I) in the workdir; returns stdout+stderr. Timeout default 120 s.",
            schema: || json!({"type": "object", "properties": {
                "code": {"type": "string"}, "timeout": {"type": "integer"}}, "required": ["code"]}),
            gate: Gate::Exec,
            keys: &[("source", "code"), ("script", "code"), ("python", "code")],
            run: execute_code,
        },
    ]
}

struct Proc {
    child: Child,
    cmd: String,
    log: PathBuf,
    started: Instant,
    exit: Option<i32>,
    /// bytes of the log already returned by `poll`
    seen: u64,
}

impl Proc {
    fn status(&mut self) -> Option<i32> {
        if self.exit.is_none() {
            if let Ok(Some(st)) = self.child.try_wait() {
                self.exit = Some(st.code().unwrap_or(-1));
            }
        }
        self.exit
    }
    /// SIGTERM, then SIGKILL after 2 s, to the whole group. Never after reaping (pid reuse).
    fn kill(&mut self) {
        if self.status().is_some() {
            return;
        }
        let pg = self.child.id() as i32;
        unsafe { libc::killpg(pg, libc::SIGTERM) };
        let t = Instant::now();
        while t.elapsed() < Duration::from_secs(2) && self.status().is_none() {
            std::thread::sleep(Duration::from_millis(50));
        }
        if self.status().is_none() {
            unsafe { libc::killpg(pg, libc::SIGKILL) };
            let _ = self.child.wait();
            self.exit = Some(-9);
        }
    }
}

#[derive(Default)]
pub struct Procs {
    next: u64,
    map: BTreeMap<u64, Proc>,
}

impl Drop for Procs {
    fn drop(&mut self) {
        for p in self.map.values_mut() {
            p.kill();
            let _ = std::fs::remove_file(&p.log);
        }
    }
}

/// Spawn `cmd` in its own process group with stdout+stderr into a fresh log file.
fn spawn(mut cmd: Command, stdin: Stdio) -> Result<(Child, PathBuf), String> {
    let log = crate::tmp_path("tool-log");
    let f = File::create(&log).map_err(|e| e.to_string())?;
    let f2 = f.try_clone().map_err(|e| e.to_string())?;
    cmd.stdin(stdin).stdout(f).stderr(f2);
    unsafe {
        cmd.pre_exec(|| {
            libc::setpgid(0, 0);
            Ok(())
        });
    }
    let child = cmd.spawn().map_err(|e| e.to_string())?;
    Ok((child, log))
}

fn tail(log: &PathBuf, max: usize) -> String {
    let mut b = Vec::new();
    if let Ok(mut f) = File::open(log) {
        let n = f.metadata().map(|m| m.len()).unwrap_or(0);
        let skip = n.saturating_sub(max as u64);
        let _ = f.seek(SeekFrom::Start(skip));
        let _ = f.read_to_end(&mut b);
        if skip > 0 {
            let mut s = format!("… [{skip} earlier bytes omitted]\n");
            s.push_str(&String::from_utf8_lossy(&b));
            return s;
        }
    }
    String::from_utf8_lossy(&b).into_owned()
}

/// Run to completion or timeout; returns "exit N" plus the tail of the output.
fn run_fg(ctx: &Ctx, cmd: Command, timeout: u64) -> Result<String, String> {
    let (mut child, log) = spawn(cmd, Stdio::null())?;
    let t = Instant::now();
    let limit = Duration::from_secs(timeout.clamp(1, 600));
    let code = loop {
        if let Some(st) = child.try_wait().map_err(|e| e.to_string())? {
            break Some(st.code().unwrap_or(-1));
        }
        if t.elapsed() > limit {
            unsafe { libc::killpg(child.id() as i32, libc::SIGKILL) };
            let _ = child.wait();
            break None;
        }
        std::thread::sleep(Duration::from_millis(20));
    };
    let out = tail(&log, ctx.cfg.max_output.saturating_sub(200));
    let _ = std::fs::remove_file(&log);
    let head = match code {
        Some(0) => String::new(),
        Some(c) => format!("[exit {c}]\n"),
        None => format!("[timed out after {}s and was killed; use background=true for long commands]\n", limit.as_secs()),
    };
    Ok(if out.is_empty() && head.is_empty() { "(no output)".into() } else { head + &out })
}

fn workdir(ctx: &Ctx, a: &Args) -> Result<PathBuf, String> {
    let d = crate::fs::resolve(ctx, opt_s(a, "workdir").unwrap_or("."))?;
    if !d.is_dir() {
        return Err(format!("{} is not a directory", d.display()));
    }
    Ok(d)
}

fn bash(ctx: &Ctx, a: &Args) -> Result<String, String> {
    let command = s(a, "command")?;
    let mut cmd = Command::new("bash");
    cmd.arg("-c").arg(command).current_dir(workdir(ctx, a)?);
    if !opt_b(a, "background") {
        return run_fg(ctx, cmd, opt_u(a, "timeout").unwrap_or(120));
    }
    let (child, log) = spawn(cmd, Stdio::piped())?;
    let mut ps = ctx.procs.lock().unwrap();
    ps.next += 1;
    let id = ps.next;
    let pid = child.id();
    ps.map.insert(id, Proc { child, cmd: command.to_string(), log, started: Instant::now(), exit: None, seen: 0 });
    Ok(format!("started background process id={id} (pid {pid}); check it with process action=poll id={id}"))
}

fn process(ctx: &Ctx, a: &Args) -> Result<String, String> {
    let action = s(a, "action")?;
    let mut ps = ctx.procs.lock().unwrap();
    if action == "list" {
        let rows: Vec<String> = ps
            .map
            .iter_mut()
            .map(|(id, p)| {
                let st = p.status().map_or("running".to_string(), |c| format!("exited {c}"));
                format!("{id}\t{st}\t{}s\t{}", p.started.elapsed().as_secs(), p.cmd.chars().take(120).collect::<String>())
            })
            .collect();
        return Ok(if rows.is_empty() { "no background processes".into() } else { rows.join("\n") });
    }
    let id = opt_u(a, "id").ok_or("missing id (see action=list)")?;
    let max = ctx.cfg.max_output.saturating_sub(200);
    let p = ps.map.get_mut(&id).ok_or_else(|| format!("no process {id} (see action=list)"))?;
    let st = |p: &mut Proc| p.status().map_or("running".to_string(), |c| format!("exited {c}"));
    match action {
        "poll" => {
            let status = st(p);
            let mut b = Vec::new();
            if let Ok(mut f) = File::open(&p.log) {
                let _ = f.seek(SeekFrom::Start(p.seen));
                let _ = f.read_to_end(&mut b);
            }
            p.seen += b.len() as u64;
            let new = String::from_utf8_lossy(&b);
            let new = if new.len() > max { format!("… {}", &new[new.floor_char_boundary(new.len() - max)..]) } else { new.into_owned() };
            Ok(format!("[{status}]\n{new}"))
        }
        "log" => {
            let text = tail(&p.log, 4 << 20);
            let lines: Vec<&str> = text.lines().collect();
            let limit = opt_u(a, "limit").unwrap_or(200) as usize;
            let start = opt_u(a, "offset").map_or(lines.len().saturating_sub(limit), |o| (o as usize).saturating_sub(1));
            Ok(format!("[{}] lines {}..{} of {}\n{}", st(p), start + 1, (start + limit).min(lines.len()), lines.len(),
                lines.iter().skip(start).take(limit).copied().collect::<Vec<_>>().join("\n")))
        }
        "wait" => {
            let t = Instant::now();
            let limit = Duration::from_secs(opt_u(a, "timeout").unwrap_or(60).clamp(1, 600));
            while p.status().is_none() && t.elapsed() < limit {
                std::thread::sleep(Duration::from_millis(50));
            }
            Ok(format!("[{}]\n{}", st(p), tail(&p.log, max)))
        }
        "kill" => {
            p.kill();
            Ok(format!("process {id}: {}", st(p)))
        }
        "write" => {
            let input = s(a, "input")?;
            let w = p.child.stdin.as_mut().ok_or("stdin is closed")?;
            w.write_all(input.as_bytes()).and_then(|_| w.flush()).map_err(|e| e.to_string())?;
            Ok(format!("wrote {} bytes to process {id}", input.len()))
        }
        _ => Err(format!("unknown action {action:?}")),
    }
}

fn execute_code(ctx: &Ctx, a: &Args) -> Result<String, String> {
    let code = s(a, "code")?;
    let f = crate::tmp_path("code").with_extension("py");
    std::fs::write(&f, code).map_err(|e| e.to_string())?;
    let mut cmd = Command::new("python3");
    cmd.arg("-I").arg(&f).current_dir(ctx.cfg.root.canonicalize().map_err(|e| e.to_string())?);
    let r = run_fg(ctx, cmd, opt_u(a, "timeout").unwrap_or(120));
    let _ = std::fs::remove_file(&f);
    r
}

#[cfg(test)]
mod tests {
    use crate::{call, enabled, Config, Ctx};
    use serde_json::json;

    #[test]
    fn bash_background_and_kill() {
        let d = std::env::temp_dir().join(format!("breccia-tools-exec-{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        let mut c = Config::new(d.clone());
        c.allow_exec = true;
        let c = Ctx::new(c);
        let t = enabled(&c, None);
        assert_eq!(call(&c, &t, "run_terminal_cmd", &json!({"cmd": "echo hi; echo err >&2"})), "hi\nerr\n");
        assert!(call(&c, &t, "bash", &json!({"command": "exit 3"})).starts_with("[exit 3]"));
        assert!(call(&c, &t, "bash", &json!({"command": "sleep 5", "timeout": 1})).contains("timed out"));
        let r = call(&c, &t, "bash", &json!({"command": "echo up; read x; echo got $x; sleep 30", "background": true}));
        assert!(r.contains("id=1"), "{r}");
        assert!(call(&c, &t, "process", &json!({"action": "write", "id": 1, "input": "abc\n"})).starts_with("wrote"));
        std::thread::sleep(std::time::Duration::from_millis(300));
        let r = call(&c, &t, "process", &json!({"action": "poll", "id": 1}));
        assert_eq!(r, "[running]\nup\ngot abc\n");
        let r = call(&c, &t, "process", &json!({"action": "kill", "id": "1"}));
        assert!(r.contains("exited"), "{r}");
        assert!(call(&c, &t, "process", &json!({"action": "kill", "id": 9})).starts_with("error: process: no process 9"));
        assert_eq!(call(&c, &t, "python", &json!({"code": "print(6*7)"})), "42\n");
        std::fs::remove_dir_all(d).ok();
    }
}
