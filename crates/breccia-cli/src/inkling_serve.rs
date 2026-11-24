//! `breccia serve` for an `inkling2_loop` GGUF: greedy decode on one card over
//! [`InklingForward`], `BRECCIA_SERVE_SEQS` slots decoded together in one
//! batched step (bit-equal to running them one by one).
//!
//! Placement knobs (the same as the parity tests): `BRECCIA_INKLING_RESIDENT=dense|all`
//! keeps weights on the device, `BRECCIA_INKLING_HOST_LAYERS="0..6,18..24"` runs those
//! layers' experts on the CPU lane.

use std::sync::Arc;

use anyhow::{Context, Result};
use breccia_engine::inkling::{self, InklingConfig};
use breccia_engine::inkling_forward::{InklingForward, InklingSeq, Residency};
use breccia_format::tokenizer::Tokenizer;
use breccia_format::ShardSet;
use breccia_kernels::{cpu, Device};

pub fn is_inkling(gguf: &str) -> bool {
    breccia_format::shard::ShardSet::open_glob(gguf)
        .ok()
        .is_some_and(|s| s.primary().architecture().is_some_and(|a| a.starts_with("inkling")))
}

struct InklingBackend<'g> {
    fwd: InklingForward<'g>,
    seqs: Vec<Option<InklingSeq<'g>>>,
    tok: Tokenizer,
    template: Option<breccia_chat::ChatTemplate>,
    ctx: usize,
    chunk: usize,
    max_tokens: usize,
}

fn argmax(l: &[f32]) -> u32 {
    let mut best = 0;
    for (i, &v) in l.iter().enumerate() {
        if v > l[best] {
            best = i;
        }
    }
    best as u32
}

fn layers(s: &str) -> Result<Vec<usize>> {
    let mut v = Vec::new();
    for p in s.split(',').map(str::trim).filter(|p| !p.is_empty()) {
        match p.split_once("..") {
            Some((a, b)) => v.extend(a.trim().parse::<usize>()?..b.trim().parse::<usize>()?),
            None => v.push(p.parse()?),
        }
    }
    Ok(v)
}

impl breccia_server::Backend for InklingBackend<'_> {
    fn model_name(&self) -> String {
        "breccia-inkling".into()
    }
    fn tokenize(&self, text: &str) -> Result<Vec<u32>, String> {
        Ok(self.tok.encode(text))
    }
    fn tokenize_special(&self, text: &str, literals: &[String]) -> Result<Vec<u32>, String> {
        Ok(self.tok.encode_templated(text, literals))
    }
    fn control_texts(&self) -> Vec<String> {
        self.tok.control_texts()
    }
    fn chat_template(&self) -> Option<&breccia_chat::ChatTemplate> {
        self.template.as_ref()
    }
    fn decode_text(&self, ids: &[u32]) -> String {
        self.tok.decode(ids)
    }
    fn eos(&self) -> Option<u32> {
        self.tok.eos
    }
    fn complete(&mut self, prompt: &[u32], max_tokens: usize) -> Result<Vec<u32>, String> {
        let Some((first, n)) = self.admit(0, prompt, max_tokens)? else {
            return Err("prompt does not fit".into());
        };
        let mut out = vec![first];
        while out.len() < n && Some(*out.last().unwrap()) != self.tok.eos {
            let t = self.step(&[(0, *out.last().unwrap())]);
            match t {
                Ok(t) => out.push(t[0]),
                Err(e) => {
                    self.release(0);
                    return Err(e);
                }
            }
        }
        self.release(0);
        Ok(out)
    }
    fn slots(&self) -> usize {
        self.seqs.len()
    }
    fn admit(&mut self, slot: usize, prompt: &[u32], max_tokens: usize) -> Result<Option<(u32, usize)>, String> {
        if prompt.is_empty() || prompt.len() >= self.ctx {
            return Err(format!("prompt {} tokens, context {}", prompt.len(), self.ctx));
        }
        // shortcut: each request allocates its slot's full-ctx KV (a few ms); pool pages
        // across slots when many slots x long ctx stop fitting on the card.
        self.seqs[slot] = None;
        let mut seq = self.fwd.new_seq(self.ctx, self.chunk).map_err(|e| format!("{e:#}"))?;
        let logits = self.fwd.forward(&mut seq, prompt, None).map_err(|e| format!("{e:#}"))?;
        self.seqs[slot] = Some(seq);
        let n = max_tokens.min(self.max_tokens).min(self.ctx - prompt.len()).max(1);
        Ok(Some((argmax(&logits), n)))
    }
    fn step(&mut self, rows: &[(usize, u32)]) -> Result<Vec<u32>, String> {
        let mut by: Vec<Option<&mut InklingSeq>> = self.seqs.iter_mut().map(Option::as_mut).collect();
        let mut refs = Vec::with_capacity(rows.len());
        for &(s, _) in rows {
            refs.push(by.get_mut(s).and_then(Option::take).ok_or_else(|| format!("slot {s} is not live"))?);
        }
        let ids: Vec<u32> = rows.iter().map(|r| r.1).collect();
        let logits = self.fwd.forward_batch(&mut refs, &ids, None).map_err(|e| format!("{e:#}"))?;
        Ok(logits.iter().map(|l| argmax(l)).collect())
    }
    fn release(&mut self, slot: usize) {
        self.seqs[slot] = None;
    }
    fn context(&self) -> Option<usize> {
        Some(self.ctx)
    }
}

/// Serve `gguf` on `device` (a Vulkan selector, e.g. a PCI address).
pub fn serve(port: u16, gguf: &str, device: Option<&str>, ctx: usize, chunk: usize, max_tokens: usize) -> Result<()> {
    let dev = Device::open(device).context("opening the Vulkan device")?;
    let shards = Arc::new(ShardSet::open_family(gguf).with_context(|| format!("opening {gguf}"))?);
    let cfg = InklingConfig::from_metadata(&shards.metadata()?)?;
    inkling::bind(&shards, &cfg)?;
    let n_layers = cfg.n_layers;
    let mut fwd = InklingForward::new(dev.gpu(), &shards, cfg)?;
    let r = std::env::var("BRECCIA_INKLING_RESIDENT").unwrap_or_default();
    fwd.set_residency(Residency { dense: r == "dense" || r == "all", experts: r == "all" });
    let host = match std::env::var("BRECCIA_INKLING_HOST_LAYERS") {
        Ok(s) => layers(&s)?,
        Err(_) => Vec::new(),
    };
    anyhow::ensure!(host.iter().all(|&l| l < n_layers), "BRECCIA_INKLING_HOST_LAYERS: a layer >= {n_layers}");
    if !host.is_empty() {
        let threads = std::thread::available_parallelism().map_or(8, |n| n.get().min(32));
        fwd.set_host_experts(cpu::ExpertPool::new(shards.clone(), threads, cpu::path()), &host)?;
    }
    let seqs = std::env::var("BRECCIA_SERVE_SEQS").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(1).max(1);
    let tok = Tokenizer::from_shards(&shards)?;
    let template = if std::env::var("BRECCIA_JINJA").as_deref() == Ok("1") {
        let source = shards.primary().metadata_str("tokenizer.chat_template").context("BRECCIA_JINJA=1 but the model has no tokenizer.chat_template")?.to_string();
        let text = |id: Option<u32>| id.map(|i| tok.decode(&[i])).unwrap_or_default();
        Some(breccia_chat::ChatTemplate::new(&source, &text(tok.bos), &text(tok.eos)).map_err(|e| anyhow::anyhow!("chat template: {e}"))?)
    } else {
        None
    };
    println!(
        "inkling  device {device:?}, resident {r:?}, host layers {host:?}, {seqs} slots x ctx {ctx}, chunk {chunk}, eos {:?}, jinja {}",
        tok.eos,
        template.is_some()
    );
    let backend = InklingBackend { fwd, seqs: (0..seqs).map(|_| None).collect(), tok, template, ctx, chunk, max_tokens };
    breccia_server::serve(&format!("127.0.0.1:{port}"), backend).map_err(|e| anyhow::anyhow!("server: {e}"))
}

#[cfg(test)]
mod tests {
    #[test]
    fn layer_ranges() {
        assert_eq!(super::layers("0..3, 18,20..22").unwrap(), vec![0, 1, 2, 18, 20, 21]);
        assert!(super::layers("x").is_err());
        assert_eq!(super::argmax(&[0.0, 2.0, 1.0, 2.0]), 1);
    }

    /// BRECCIA_INKLING_GGUF=<gguf> cargo test -p breccia-cli -- --ignored inkling_template:
    /// the model's own chat template renders and tokenizes (no GPU).
    #[test]
    #[ignore]
    fn inkling_template() {
        let Ok(gguf) = std::env::var("BRECCIA_INKLING_GGUF") else { return };
        let shards = breccia_format::ShardSet::open_family(&gguf).unwrap();
        let tok = breccia_format::tokenizer::Tokenizer::from_shards(&shards).unwrap();
        let source = shards.primary().metadata_str("tokenizer.chat_template").unwrap().to_string();
        let text = |id: Option<u32>| id.map(|i| tok.decode(&[i])).unwrap_or_default();
        let t = breccia_chat::ChatTemplate::new(&source, &text(tok.bos), &text(tok.eos)).unwrap();
        let r = t.render(&serde_json::json!({"messages": [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "What is the capital of France?"}]})).unwrap();
        let ids = tok.encode_templated(&r.text, &[]);
        println!("eos {:?} bos {:?} think_open {}\nTEXT<<{}>>\nIDS {:?}", tok.eos, tok.bos, r.think_open, r.text, ids);
        assert_eq!(tok.decode(&ids), r.text);
    }
}
