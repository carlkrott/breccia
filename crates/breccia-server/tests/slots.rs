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
