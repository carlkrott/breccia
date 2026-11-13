//! The sequence-slot loop (`Backend::slots` > 1) over real HTTP, with a mock that
//! counts upward from 'a' per slot and records how many rows each step carried.

use breccia_server::{serve, Backend};
use std::io::{Read, Write};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

struct Slots {
    widest: Arc<AtomicUsize>,
    alone: Arc<AtomicUsize>,
    live: Vec<bool>,
}

impl Backend for Slots {
    fn model_name(&self) -> String {
        "slots".into()
    }
    fn tokenize(&self, text: &str) -> Result<Vec<u32>, String> {
        Ok(text.bytes().map(|b| b as u32 + 1).collect())
    }
    fn decode_text(&self, ids: &[u32]) -> String {
        ids.iter().map(|&i| char::from_u32(i - 1).unwrap_or('?')).collect()
    }
    fn complete(&mut self, _prompt: &[u32], max_tokens: usize) -> Result<Vec<u32>, String> {
        assert!(self.live.iter().all(|l| !l), "a request ran alone while slots were live");
        self.alone.fetch_add(1, Ordering::SeqCst);
        Ok((0..max_tokens as u32).map(|i| b'A' as u32 + 1 + i).collect())
    }
    fn eos(&self) -> Option<u32> {
        Some(b'z' as u32 + 1)
    }
    fn special(&self, _text: &str) -> Option<u32> {
        None
    }
    fn spec_hint(&mut self, explicit: Option<bool>, _code: bool) -> bool {
        explicit.unwrap_or(false)
    }
    fn slots(&self) -> usize {
        self.live.len()
    }
    fn admit(&mut self, slot: usize, _prompt: &[u32], max_tokens: usize) -> Result<Option<(u32, usize)>, String> {
        assert!(!self.live[slot]);
        self.live[slot] = true;
        Ok(Some((b'a' as u32 + 1, max_tokens)))
    }
    fn step(&mut self, rows: &[(usize, u32)]) -> Result<Vec<u32>, String> {
        assert!(rows.iter().all(|&(s, _)| self.live[s]));
        self.widest.fetch_max(rows.len(), Ordering::SeqCst);
        std::thread::sleep(std::time::Duration::from_millis(5));
        Ok(rows.iter().map(|&(_, t)| t + 1).collect())
    }
    fn release(&mut self, slot: usize) {
        self.live[slot] = false;
    }
}

fn post(port: u16, body: &str) -> serde_json::Value {
    let mut s = std::net::TcpStream::connect(("127.0.0.1", port)).unwrap();
    write!(
        s,
        "POST /v1/completions HTTP/1.1\r\nHost: x\r\nContent-Length: {}\r\n\r\n{body}",
        body.len()
    )
    .unwrap();
    let mut out = String::new();
    s.read_to_string(&mut out).unwrap();
    let (_, json) = out.split_once("\r\n\r\n").unwrap();
    serde_json::from_str(json).unwrap()
}

#[test]
fn slots_batch_concurrent_requests_and_spec_runs_alone() {
    let port = std::net::TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port();
    let (widest, alone) = (Arc::new(AtomicUsize::new(0)), Arc::new(AtomicUsize::new(0)));
    let b = Slots { widest: widest.clone(), alone: alone.clone(), live: vec![false; 3] };
    std::thread::spawn(move || serve(&format!("127.0.0.1:{port}"), b));
    std::thread::sleep(std::time::Duration::from_millis(100));

    let handles: Vec<_> = [5usize, 20, 24, 12]
        .into_iter()
        .map(|n| std::thread::spawn(move || (n, post(port, &format!(r#"{{"prompt":"hi","max_tokens":{n}}}"#)))))
        .collect();
    for h in handles {
        let (n, v) = h.join().unwrap();
        let want: String = ('a'..='z').take(n).collect();
        assert_eq!(v["choices"][0]["text"], want.as_str(), "max_tokens {n}");
    }
    assert!(widest.load(Ordering::SeqCst) >= 2, "no step batched two slots");
    assert_eq!(alone.load(Ordering::SeqCst), 0);

    // EOS ('z') ends a slot early; an explicit spec request runs alone through complete
    let v = post(port, r#"{"prompt":"hi","max_tokens":40}"#);
    assert_eq!(v["choices"][0]["finish_reason"], "stop");
    assert_eq!(v["choices"][0]["text"].as_str().unwrap().len(), 25);
    let v = post(port, r#"{"prompt":"hi","max_tokens":3,"mtp":true}"#);
    assert_eq!(v["choices"][0]["text"], "ABC");
    assert_eq!(alone.load(Ordering::SeqCst), 1);
}

/// Chunked admission: the prompt goes in one `feed` per loop turn, one char per chunk.
struct Feeder {
    pending: Vec<usize>,
    live: Vec<bool>,
    /// decode steps that ran while some slot was still feeding its prompt
    interleaved: Arc<AtomicUsize>,
}

impl Backend for Feeder {
    fn model_name(&self) -> String {
        "feeder".into()
    }
    fn tokenize(&self, text: &str) -> Result<Vec<u32>, String> {
        Ok(text.bytes().map(|b| b as u32 + 1).collect())
    }
    fn decode_text(&self, ids: &[u32]) -> String {
        ids.iter().map(|&i| char::from_u32(i - 1).unwrap_or('?')).collect()
    }
    fn complete(&mut self, _prompt: &[u32], _max_tokens: usize) -> Result<Vec<u32>, String> {
        Err("slots only".into())
    }
    fn eos(&self) -> Option<u32> {
        Some(b'z' as u32 + 1)
    }
    fn slots(&self) -> usize {
        self.live.len()
    }
    fn admit_start(&mut self, slot: usize, prompt: &[u32], max_tokens: usize) -> Result<Option<(Option<u32>, usize)>, String> {
        assert!(!self.live[slot]);
        self.live[slot] = true;
        self.pending[slot] = prompt.len();
        Ok(Some((None, max_tokens)))
    }
    fn feed(&mut self, slot: usize, _decoding: usize) -> Result<Option<u32>, String> {
        assert!(self.pending[slot] > 0, "fed a slot with its prompt in");
        self.pending[slot] -= 1;
        Ok((self.pending[slot] == 0).then_some(b'a' as u32 + 1))
    }
    fn step(&mut self, rows: &[(usize, u32)]) -> Result<Vec<u32>, String> {
        assert!(rows.iter().all(|&(s, _)| self.live[s] && self.pending[s] == 0), "a feeding slot stepped");
        if self.pending.iter().any(|&p| p > 0) {
            self.interleaved.fetch_add(1, Ordering::SeqCst);
        }
        std::thread::sleep(std::time::Duration::from_millis(5));
        Ok(rows.iter().map(|&(_, t)| t + 1).collect())
    }
    fn release(&mut self, slot: usize) {
        self.live[slot] = false;
    }
}

#[test]
fn a_long_prompt_feeds_between_the_live_slots_steps() {
    let port = std::net::TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port();
    let interleaved = Arc::new(AtomicUsize::new(0));
    let b = Feeder { pending: vec![0; 2], live: vec![false; 2], interleaved: interleaved.clone() };
    std::thread::spawn(move || serve(&format!("127.0.0.1:{port}"), b));
    std::thread::sleep(std::time::Duration::from_millis(100));

    let a = std::thread::spawn(move || post(port, r#"{"prompt":"hi","max_tokens":40}"#));
    std::thread::sleep(std::time::Duration::from_millis(30));
    let v = post(port, r#"{"prompt":"a much longer prompt","max_tokens":3}"#);
    assert_eq!(v["choices"][0]["text"], "abc");
    let v = a.join().unwrap();
    assert_eq!(v["choices"][0]["text"].as_str().unwrap().len(), 25);
    assert!(interleaved.load(Ordering::SeqCst) >= 10, "the live slot waited for the prompt");
}

/// A spec run alone (`complete_alone`) hands over to batched decode when a request arrives.
struct Spec {
    live: Vec<bool>,
    yielded: Arc<AtomicUsize>,
    widest: Arc<AtomicUsize>,
}

impl Backend for Spec {
    fn model_name(&self) -> String {
        "spec".into()
    }
    fn tokenize(&self, text: &str) -> Result<Vec<u32>, String> {
        Ok(text.bytes().map(|b| b as u32 + 1).collect())
    }
    fn decode_text(&self, ids: &[u32]) -> String {
        ids.iter().map(|&i| char::from_u32(i - 1).unwrap_or('?')).collect()
    }
    fn complete(&mut self, _prompt: &[u32], _max_tokens: usize) -> Result<Vec<u32>, String> {
        Err("complete_alone only".into())
    }
    fn eos(&self) -> Option<u32> {
        Some(b'z' as u32 + 1)
    }
    fn spec_hint(&mut self, explicit: Option<bool>, _code: bool) -> bool {
        explicit.unwrap_or(false)
    }
    fn slots(&self) -> usize {
        self.live.len()
    }
    fn complete_alone(&mut self, slot: usize, _prompt: &[u32], max_tokens: usize, waiting: &Arc<std::sync::atomic::AtomicBool>) -> Result<(Vec<u32>, usize, bool), String> {
        assert!(self.live.iter().all(|l| !l), "a spec run started beside live slots");
        let mut ids = vec![b'A' as u32 + 1];
        while ids.len() < max_tokens {
            if waiting.load(Ordering::SeqCst) {
                self.live[slot] = true;
                self.yielded.fetch_add(1, Ordering::SeqCst);
                return Ok((ids, max_tokens, true));
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
            ids.push(ids.last().unwrap() + 1);
        }
        Ok((ids, max_tokens, false))
    }
    fn admit(&mut self, slot: usize, _prompt: &[u32], max_tokens: usize) -> Result<Option<(u32, usize)>, String> {
        assert!(!self.live[slot]);
        self.live[slot] = true;
        Ok(Some((b'a' as u32 + 1, max_tokens)))
    }
    fn step(&mut self, rows: &[(usize, u32)]) -> Result<Vec<u32>, String> {
        assert!(rows.iter().all(|&(s, _)| self.live[s]));
        self.widest.fetch_max(rows.len(), Ordering::SeqCst);
        std::thread::sleep(std::time::Duration::from_millis(5));
        Ok(rows.iter().map(|&(_, t)| t + 1).collect())
    }
    fn release(&mut self, slot: usize) {
        self.live[slot] = false;
    }
}

#[test]
fn a_spec_run_hands_over_when_a_request_arrives() {
    let port = std::net::TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port();
    let (yielded, widest) = (Arc::new(AtomicUsize::new(0)), Arc::new(AtomicUsize::new(0)));
    let b = Spec { live: vec![false; 2], yielded: yielded.clone(), widest: widest.clone() };
    std::thread::spawn(move || serve(&format!("127.0.0.1:{port}"), b));
    std::thread::sleep(std::time::Duration::from_millis(100));

    // alone to the end: the slot is released, the next spec run may start
    let v = post(port, r#"{"prompt":"hi","max_tokens":4,"mtp":true}"#);
    assert_eq!(v["choices"][0]["text"], "ABCD");
    let a = std::thread::spawn(move || post(port, r#"{"prompt":"hi","max_tokens":40,"mtp":true}"#));
    std::thread::sleep(std::time::Duration::from_millis(50));
    let v = post(port, r#"{"prompt":"hi","max_tokens":3}"#);
    assert_eq!(v["choices"][0]["text"], "abc");
    let v = a.join().unwrap();
    let want: String = (b'A'..b'A' + 40).map(char::from).collect();
    assert_eq!(v["choices"][0]["text"], want.as_str(), "the handed-over stream continued in order");
    assert_eq!(yielded.load(Ordering::SeqCst), 1);
    assert!(widest.load(Ordering::SeqCst) >= 2, "the two streams never stepped together");
}

fn get(port: u16, path: &str) -> serde_json::Value {
    let mut s = std::net::TcpStream::connect(("127.0.0.1", port)).unwrap();
    write!(s, "GET {path} HTTP/1.1\r\nHost: x\r\n\r\n").unwrap();
    let mut out = String::new();
    s.read_to_string(&mut out).unwrap();
    serde_json::from_str(out.split_once("\r\n\r\n").unwrap().1).unwrap()
}

/// A client that hangs up mid-generation frees its slot; `/slots` shows the slot busy, then idle.
#[test]
fn disconnect_cancels_and_slots_report_busy() {
    let port = std::net::TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port();
    let b = Forever { live: vec![false; 2] };
    std::thread::spawn(move || serve(&format!("127.0.0.1:{port}"), b));
    std::thread::sleep(std::time::Duration::from_millis(100));
    let idle = serde_json::json!([{"id": 0, "is_processing": false}, {"id": 1, "is_processing": false}]);
    assert_eq!(get(port, "/slots"), idle);

    let mut s = std::net::TcpStream::connect(("127.0.0.1", port)).unwrap();
    let body = r#"{"prompt":"hi","max_tokens":100000}"#;
    write!(s, "POST /v1/completions HTTP/1.1\r\nHost: x\r\nContent-Length: {}\r\n\r\n{body}", body.len()).unwrap();
    std::thread::sleep(std::time::Duration::from_millis(30));
    let busy = get(port, "/slots");
    assert_eq!(busy[0]["is_processing"], true, "{busy}");
    drop(s);
    // the connection thread checks every 500 ms; the engine frees the slot on its next turn
    let t = std::time::Instant::now();
    while get(port, "/slots") != idle {
        assert!(t.elapsed().as_secs() < 5, "slot still busy after the client left");
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
    // the serve still answers
    let v = post(port, r#"{"prompt":"hi","max_tokens":3}"#);
    assert_eq!(v["choices"][0]["text"], "aaa");
}

/// Repeats 'a' (never EOS), so only max_tokens or a hang-up ends a request.
struct Forever {
    live: Vec<bool>,
}

impl Backend for Forever {
    fn model_name(&self) -> String {
        "forever".into()
    }
    fn tokenize(&self, text: &str) -> Result<Vec<u32>, String> {
        Ok(text.bytes().map(|b| b as u32 + 1).collect())
    }
    fn decode_text(&self, ids: &[u32]) -> String {
        ids.iter().map(|&i| char::from_u32(i - 1).unwrap_or('?')).collect()
    }
    fn complete(&mut self, _prompt: &[u32], _max_tokens: usize) -> Result<Vec<u32>, String> {
        Err("slots only".into())
    }
    fn eos(&self) -> Option<u32> {
        Some(b'z' as u32 + 1)
    }
    fn slots(&self) -> usize {
        self.live.len()
    }
    fn admit(&mut self, slot: usize, _prompt: &[u32], max_tokens: usize) -> Result<Option<(u32, usize)>, String> {
        assert!(!self.live[slot]);
        self.live[slot] = true;
        Ok(Some((b'a' as u32 + 1, max_tokens)))
    }
    fn step(&mut self, rows: &[(usize, u32)]) -> Result<Vec<u32>, String> {
        assert!(rows.iter().all(|&(s, _)| self.live[s]), "stepped a released slot");
        std::thread::sleep(std::time::Duration::from_millis(5));
        Ok(rows.iter().map(|&(_, t)| t).collect())
    }
    fn release(&mut self, slot: usize) {
        self.live[slot] = false;
    }
}

/// The child half of `sigterm_answers_503_and_exits_0`: a serve on the port in `BRECCIA_TEST_PORT`.
#[test]
#[ignore]
fn sigterm_child() {
    let Ok(port) = std::env::var("BRECCIA_TEST_PORT") else { return };
    let _ = serve(&format!("127.0.0.1:{port}"), Forever { live: vec![false; 2] });
}

/// SIGTERM mid-request: the open request gets 503 and the process exits 0 at once.
#[test]
fn sigterm_answers_503_and_exits_0() {
    let port = std::net::TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port();
    let mut child = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--ignored", "--exact", "sigterm_child", "--nocapture"])
        .env("BRECCIA_TEST_PORT", port.to_string())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .unwrap();
    let t = std::time::Instant::now();
    let mut s = loop {
        match std::net::TcpStream::connect(("127.0.0.1", port)) {
            Ok(s) => break s,
            Err(_) if t.elapsed().as_secs() < 10 => std::thread::sleep(std::time::Duration::from_millis(20)),
            Err(e) => panic!("child serve never listened: {e}"),
        }
    };
    let body = r#"{"prompt":"hi","max_tokens":100000}"#;
    write!(s, "POST /v1/completions HTTP/1.1\r\nHost: x\r\nContent-Length: {}\r\n\r\n{body}", body.len()).unwrap();
    std::thread::sleep(std::time::Duration::from_millis(100));
    let t = std::time::Instant::now();
    assert_eq!(unsafe { libc_kill(child.id() as i32, 15) }, 0);
    let mut out = String::new();
    s.read_to_string(&mut out).unwrap();
    assert!(out.starts_with("HTTP/1.1 503"), "{out}");
    let st = child.wait().unwrap();
    assert!(st.success(), "exit {st:?}");
    assert!(t.elapsed().as_secs() < 5, "stop took {:?}", t.elapsed());
}

extern "C" {
    #[link_name = "kill"]
    fn libc_kill(pid: i32, sig: i32) -> i32;
}
