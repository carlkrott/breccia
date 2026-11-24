//! The model's tokenizer: GGUF `tokenizer.ggml.*` → text ⇄ ids.
//!
//! The engine's prompts have always had to arrive as `--prompt-ids` from another
//! program; this is the piece that removes that dependency. It implements what the
//! model declares and llama.cpp runs:
//!
//!   * `tokenizer.ggml.model = "gpt2"`: GPT-2 byte-level BPE — a byte→unicode map so
//!     every byte is a printable symbol, then merges by rank.
//!   * `tokenizer.ggml.pre = "qwen35"`: a hand-written pre-tokenizer (the regex has a
//!     lookahead, which Rust's `regex` cannot express, so the scan is ported from
//!     llama.cpp's `unicode_regex_split_custom_qwen35` branch for branch).
//!   * `tokenizer.ggml.tokens` / `merges` / `token_type` / `bos_token_id` /
//!     `eos_token_id` / `add_bos_token`, read from the GGUF metadata.
//!
//! `encode` is byte-exact against `llama-tokenize` on the corpus in
//! `tests/tokenizer.rs`; `decode` is the inverse map. Control tokens are **not**
//! split out of text here: a caller that wants to interleave `<|...|>` specials must
//! do so explicitly (the ids are exposed through [`Tokenizer::special`]).

use std::collections::HashMap;
use std::path::Path;

use anyhow::{anyhow, Context, Result};

use crate::gguf::Gguf;
use crate::metadata::MetadataValue;

/// A GPT-2 byte-level BPE tokenizer built from one GGUF.
pub struct Tokenizer {
    /// id → token text (the byte-encoded symbol string).
    tokens: Vec<String>,
    /// token text → id.
    ids: HashMap<String, u32>,
    /// `(left, right) -> rank` for every merge, in file order.
    merges: HashMap<(String, String), u32>,
    /// GPT-2's `bytes_to_unicode`: every byte as one printable code point.
    byte_to_char: [char; 256],
    char_to_byte: HashMap<char, u8>,
    /// Per-token type from `tokenizer.ggml.token_type` (1 normal, 2 unknown,
    /// 3 control, 4 user-defined, 5 unused, 6 byte).
    types: Vec<u8>,
    /// Control / user-defined token texts (types 3 and 4) by first byte, longest first:
    /// what [`Tokenizer::encode_special`] matches.
    specials: Vec<Vec<(String, u32)>>,
    /// `tokenizer.ggml.pre`.
    pub pre: String,
    pub bos: Option<u32>,
    pub eos: Option<u32>,
    pub add_bos: bool,
}

/// First placeholder char of [`Tokenizer::encode_templated`] (plane-15 private use;
/// breccia-server uses the same value).
pub const LITERAL_BASE: u32 = 0xF0000;

fn is_letter_or_mark(c: char) -> bool {
    // llama.cpp tests L* and M* separately but every branch treats them alike, so
    // the union is what matters. `is_alphabetic` includes Other_Alphabetic marks,
    // and `is_combining_mark` covers the rest of M*.
    c.is_alphabetic() || unicode_normalization::char::is_combining_mark(c)
}

impl Tokenizer {
    /// Read the tokenizer out of a GGUF.
    pub fn from_gguf(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let gguf = Gguf::open(path).with_context(|| format!("opening {}", path.display()))?;
        let md = gguf.metadata()?;
        Self::from_metadata(&md)
    }

    /// Read the tokenizer out of a multi-shard model (metadata lives on shard 1).
    pub fn from_shards(shards: &crate::shard::ShardSet) -> Result<Self> {
        let md = shards.metadata()?;
        Self::from_metadata(&md)
    }

    fn from_metadata<'a>(md: &crate::metadata::MetadataMap<'a>) -> Result<Self> {
        let tokens: Vec<String> = match md.get("tokenizer.ggml.tokens") {
            Some(MetadataValue::Array(a)) => {
                a.as_strings()?.into_iter().map(|s| s.to_string()).collect()
            }
            _ => return Err(anyhow!("the model has no tokenizer.ggml.tokens")),
        };
        let merge_strings: Vec<&str> = match md.get("tokenizer.ggml.merges") {
            Some(MetadataValue::Array(a)) => a.as_strings()?,
            _ => return Err(anyhow!("the model has no tokenizer.ggml.merges")),
        };
        let types: Vec<u8> = match md.get("tokenizer.ggml.token_type") {
            Some(MetadataValue::Array(a)) => a
                .to_u64_vec()
                .map(|v| v.into_iter().map(|x| x as u8).collect())
                .or_else(|_| {
                    a.to_i64_vec()
                        .map(|v| v.into_iter().map(|x| x as u8).collect())
                })
                .unwrap_or_default(),
            _ => Vec::new(),
        };

        let model = md.str("tokenizer.ggml.model").unwrap_or("gpt2");
        if model != "gpt2" {
            return Err(anyhow!(
                "tokenizer model {model:?} is not gpt2 (only byte-level BPE is implemented)"
            ));
        }
        let pre = md.str("tokenizer.ggml.pre").unwrap_or("").to_string();

        // The merge list is `"left right"`; the symbols never contain a space
        // (spaces are byte-encoded to `Ġ`), so the first space separates them.
        let mut merges = HashMap::with_capacity(merge_strings.len());
        for (rank, m) in merge_strings.iter().enumerate() {
            let Some(sp) = m.find(' ') else { continue };
            let left = m[..sp].to_string();
            let right = m[sp + 1..].to_string();
            merges.entry((left, right)).or_insert(rank as u32);
        }

        let mut ids = HashMap::with_capacity(tokens.len());
        for (i, t) in tokens.iter().enumerate() {
            ids.entry(t.clone()).or_insert(i as u32);
        }

        let (byte_to_char, char_to_byte) = gpt2_byte_maps();
        let mut specials = vec![Vec::new(); 256];
        for (i, t) in tokens.iter().enumerate() {
            if matches!(types.get(i), Some(3) | Some(4)) && !t.is_empty() {
                specials[t.as_bytes()[0] as usize].push((t.clone(), i as u32));
            }
        }
        for v in &mut specials {
            v.sort_by(|a, b| b.0.len().cmp(&a.0.len()));
        }
        Ok(Self {
            tokens,
            ids,
            merges,
            byte_to_char,
            char_to_byte,
            types,
            specials,
            pre,
            bos: md
                .get("tokenizer.ggml.bos_token_id")
                .and_then(|v| v.as_u64())
                .map(|v| v as u32),
            eos: md
                .get("tokenizer.ggml.eos_token_id")
                .and_then(|v| v.as_u64())
                .map(|v| v as u32),
            add_bos: md
                .get("tokenizer.ggml.add_bos_token")
                .and_then(|v| v.as_bool())
                .unwrap_or(false),
        })
    }

    pub fn vocab_size(&self) -> usize {
        self.tokens.len()
    }

    /// The token text for `id`.
    pub fn token(&self, id: u32) -> Option<&str> {
        self.tokens.get(id as usize).map(|s| s.as_str())
    }

    /// The id of a token's text, if the vocabulary has it.
    pub fn id(&self, text: &str) -> Option<u32> {
        self.ids.get(text).copied()
    }

    /// Control / user-defined specials (by `token_type`), for a caller that wants
    /// to interleave them explicitly.
    pub fn special(&self, id: u32) -> bool {
        matches!(self.types.get(id as usize), Some(3) | Some(4))
    }

    /// Text → token ids (byte-exact with llama.cpp's BPE for this vocabulary).
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut out = self.bos_prefix();
        self.encode_plain(text, &mut out);
        out
    }

    /// Control token texts (type 3: `<|im_start|>`, `<|im_end|>` …), longest first. A
    /// chat server swaps them out of client text before rendering, see
    /// [`Tokenizer::encode_templated`].
    pub fn control_texts(&self) -> Vec<String> {
        let mut v: Vec<String> = (0..self.tokens.len())
            .filter(|&i| self.types.get(i) == Some(&3) && !self.tokens[i].is_empty())
            // the raw text, the same string `specials` matches
            .map(|i| self.tokens[i].clone())
            .collect();
        v.sort_by(|a, b| b.len().cmp(&a.len()).then(a.cmp(b)));
        v
    }

    /// Like [`Tokenizer::encode`], but control / user-defined token texts in `text`
    /// (`<|im_start|>`, `<tool_call>` …) become their single token, the way llama.cpp
    /// tokenizes a templated prompt (`parse_special`). Leftmost match, longest first.
    pub fn encode_special(&self, text: &str) -> Vec<u32> {
        self.encode_templated(text, &[])
    }

    /// [`Tokenizer::encode_special`] for a rendered chat template whose client strings
    /// had their control-token texts replaced by placeholders: the char
    /// `LITERAL_BASE + i` stands for `literals[i]`, encoded as ordinary text, so a
    /// client cannot inject `<|im_end|>` / `<|im_start|>` as real tokens.
    pub fn encode_templated(&self, text: &str, literals: &[String]) -> Vec<u32> {
        let mut out = self.bos_prefix();
        let (mut start, mut i) = (0, 0);
        let b = text.as_bytes();
        while i < b.len() {
            if b[i] == 0xF3 {
                let c = text[i..].chars().next().map_or(0, |c| c as u32);
                if let Some(lit) = c
                    .checked_sub(LITERAL_BASE)
                    .and_then(|k| literals.get(k as usize))
                {
                    self.encode_plain(&text[start..i], &mut out);
                    self.encode_plain(lit, &mut out);
                    i += 4;
                    start = i;
                    continue;
                }
            }
            match self.specials[b[i] as usize]
                .iter()
                .find(|(t, _)| text[i..].starts_with(t.as_str()))
            {
                Some((t, id)) => {
                    self.encode_plain(&text[start..i], &mut out);
                    out.push(*id);
                    i += t.len();
                    start = i;
                }
                None => i += 1,
            }
        }
        self.encode_plain(&text[start..], &mut out);
        out
    }

    fn bos_prefix(&self) -> Vec<u32> {
        self.bos.filter(|_| self.add_bos).into_iter().collect()
    }

    fn encode_plain(&self, text: &str, out: &mut Vec<u32>) {
        let gpt4o = matches!(self.pre.as_str(), "inkling2" | "gpt-4o" | "llama4");
        // llama.cpp `ignore_merges`: a pre-token that is already a vocab entry is one token.
        let ignore_merges = self.pre == "inkling2";
        let words = if gpt4o {
            split_gpt4o(text)
        } else {
            self.split(text)
        };
        for word in words {
            let encoded = self.byte_encode(&word);
            if ignore_merges {
                if let Some(&id) = self.ids.get(&encoded) {
                    out.push(id);
                    continue;
                }
            }
            for sym in self.bpe(&encoded) {
                match self.ids.get(&sym) {
                    Some(&id) => out.push(id),
                    None => {
                        // A symbol the vocabulary lacks: fall back to its bytes, the
                        // same rule llama.cpp uses.
                        for c in sym.chars() {
                            if let Some(&b) = self.char_to_byte.get(&c) {
                                let s = self.byte_to_char[b as usize].to_string();
                                if let Some(&id) = self.ids.get(&s) {
                                    out.push(id);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /// Token ids → text.
    pub fn decode(&self, ids: &[u32]) -> String {
        let mut bytes = Vec::new();
        for &id in ids {
            let Some(t) = self.tokens.get(id as usize) else {
                continue;
            };
            for c in t.chars() {
                match self.char_to_byte.get(&c) {
                    Some(&b) => bytes.push(b),
                    // A special token: append its text as-is.
                    None => bytes.extend_from_slice(c.to_string().as_bytes()),
                }
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }

    fn byte_encode(&self, word: &str) -> String {
        word.bytes()
            .map(|b| self.byte_to_char[b as usize])
            .collect()
    }

    /// Rank-ordered BPE merges over one byte-encoded word.
    fn bpe(&self, word: &str) -> Vec<String> {
        let mut syms: Vec<String> = word.chars().map(|c| c.to_string()).collect();
        while syms.len() > 1 {
            let mut best: Option<(u32, usize)> = None;
            for i in 0..syms.len() - 1 {
                let key = (syms[i].clone(), syms[i + 1].clone());
                if let Some(&r) = self.merges.get(&key) {
                    if best.is_none_or(|(br, _)| r < br) {
                        best = Some((r, i));
                    }
                }
            }
            let Some((_, i)) = best else { break };
            let merged = format!("{}{}", syms[i], syms[i + 1]);
            syms.splice(i..i + 2, [merged]);
        }
        syms
    }

    /// The `qwen35` pre-tokenizer, ported branch for branch from llama.cpp's
    /// `unicode_regex_split_custom_qwen35` (the equivalent regex uses a lookahead
    /// that Rust's `regex` cannot express).
    fn split(&self, text: &str) -> Vec<String> {
        let cps: Vec<char> = text.chars().collect();
        let n = cps.len();
        let mut out = Vec::new();
        let mut prev = 0usize; // end of the last emitted piece
        let mut pos = 0usize;
        let at = |i: usize| -> Option<char> { cps.get(i).copied() };
        let lower = |c: char| c.to_lowercase().next().unwrap_or(c);

        while pos < n {
            let cpt = cps[pos];
            let is_num = cpt.is_numeric();
            let is_alpha = is_letter_or_mark(cpt);
            let next_alpha = at(pos + 1).is_some_and(is_letter_or_mark);

            // (?i:'s|'t|'re|'ve|'m|'ll|'d)
            if cpt == '\'' && pos + 1 < n {
                let c1 = lower(cps[pos + 1]);
                if matches!(c1, 's' | 't' | 'm' | 'd') {
                    out.push(cps[prev..pos + 2].iter().collect());
                    prev = pos + 2;
                    pos += 2;
                    continue;
                }
                if pos + 2 < n {
                    let c2 = lower(cps[pos + 2]);
                    if (c2 == 'e' && matches!(c1, 'r' | 'v')) || (c1 == 'l' && c2 == 'l') {
                        out.push(cps[prev..pos + 3].iter().collect());
                        prev = pos + 3;
                        pos += 3;
                        continue;
                    }
                }
            }

            // [^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+
            if !matches!(cpt, '\r' | '\n') && !is_num && (is_alpha || next_alpha) {
                pos += 1;
                while pos < n && is_letter_or_mark(cps[pos]) {
                    pos += 1;
                }
                out.push(cps[prev..pos].iter().collect());
                prev = pos;
                continue;
            }

            // \p{N}
            if is_num {
                pos += 1;
                out.push(cps[prev..pos].iter().collect());
                prev = pos;
                continue;
            }

            // <space>?[^\s\p{L}\p{M}\p{N}]+[\r\n]*
            let is_punct = |c: char| !(c.is_whitespace() || is_letter_or_mark(c) || c.is_numeric());
            let scan = pos + usize::from(cpt == ' ');
            if at(scan).is_some_and(is_punct) {
                let mut p = scan;
                while p < n && is_punct(cps[p]) {
                    p += 1;
                }
                while p < n && (cps[p] == '\r' || cps[p] == '\n') {
                    p += 1;
                }
                pos = p;
                out.push(cps[prev..pos].iter().collect());
                prev = pos;
                continue;
            }

            // whitespace: \s*[\r\n]+ | \s+(?!\S) | \s+
            let mut nws = 0usize;
            let mut last_rn = 0usize;
            while pos + nws < n && cps[pos + nws].is_whitespace() {
                if cps[pos + nws] == '\r' || cps[pos + nws] == '\n' {
                    last_rn = pos + nws + 1;
                }
                nws += 1;
            }
            if last_rn > 0 {
                pos = last_rn;
            } else if nws > 1 && pos + nws < n {
                pos += nws - 1;
            } else if nws > 0 {
                pos += nws;
            } else {
                pos += 1; // no match
            }
            out.push(cps[prev..pos].iter().collect());
            prev = pos;
        }
        out
    }
}

/// llama.cpp's code-point class for the GPT-4o regex. llama.cpp runs that regex on a
/// "collapsed" text (every non-ASCII code point -> one byte per Unicode category), so
/// non-ASCII letters are neither upper nor lower case: they pass both letter classes.
#[derive(Clone, Copy, PartialEq)]
enum Cls {
    Ws,
    Letter,
    Mark,
    Num,
    Other,
}

fn cls(c: char) -> Cls {
    if c.is_ascii() {
        return match c {
            ' ' | '\t' | '\n' | '\x0b' | '\x0c' | '\r' => Cls::Ws,
            'A'..='Z' | 'a'..='z' => Cls::Letter,
            '0'..='9' => Cls::Num,
            _ => Cls::Other,
        };
    }
    // shortcut: Rust's Unicode tables stand in for llama.cpp's unicode_cpt_flags; exotic
    // scripts may differ at the edges, revisit if a non-Latin parity test fails.
    if c.is_whitespace() {
        Cls::Ws
    } else if c.is_numeric() {
        Cls::Num
    } else if unicode_normalization::char::is_combining_mark(c) {
        Cls::Mark
    } else if c.is_alphabetic() {
        Cls::Letter
    } else {
        Cls::Other
    }
}

/// The GPT-4o pre-tokenizer (llama.cpp `LLAMA_VOCAB_PRE_TYPE_GPT4O`, used by inkling2),
/// a backtracking port of the collapsed regex, alternatives in order:
///   1 `[^\r\n\p{L}\p{N}]? U* L+ ('s|'t|'re|'ve|'m|'ll|'d)?`
///   2 `[^\r\n\p{L}\p{N}]? U+ L* (contraction)?`
///   3 `\p{N}{1,3}`   4 ` ?[^\s\p{L}\p{N}]+[\r\n/]*`
///   5 `\s*[\r\n]+`   6 `\s+(?!\S)`   7 `\s+`
/// with U = letter or mark but not a-z, L = letter or mark but not A-Z.
fn split_gpt4o(text: &str) -> Vec<String> {
    let cps: Vec<char> = text.chars().collect();
    let n = cps.len();
    let c = |i: usize| cps.get(i).copied();
    let k = |i: usize| c(i).map(cls);
    let upper = |i: usize| {
        c(i).is_some_and(|ch| {
            matches!(cls(ch), Cls::Letter | Cls::Mark) && !ch.is_ascii_lowercase()
        })
    };
    let lower = |i: usize| {
        c(i).is_some_and(|ch| {
            matches!(cls(ch), Cls::Letter | Cls::Mark) && !ch.is_ascii_uppercase()
        })
    };
    let run = |mut i: usize, f: &dyn Fn(usize) -> bool| {
        let s = i;
        while i < n && f(i) {
            i += 1;
        }
        i - s
    };
    let contraction = |i: usize| -> usize {
        if c(i) != Some('\'') {
            return 0;
        }
        let a = c(i + 1).map(|x| x.to_ascii_lowercase());
        let b = c(i + 2).map(|x| x.to_ascii_lowercase());
        match (a, b) {
            (Some('s' | 't' | 'm' | 'd'), _) => 2,
            (Some('r' | 'v'), Some('e')) | (Some('l'), Some('l')) => 3,
            _ => 0,
        }
    };
    let prefix_ok = |i: usize| {
        c(i).is_some_and(|ch| {
            ch != '\r' && ch != '\n' && !matches!(cls(ch), Cls::Letter | Cls::Num)
        })
    };
    let ws = |i: usize| k(i) == Some(Cls::Ws);

    let match_at = |p: usize| -> usize {
        let prefixes: &[usize] = if prefix_ok(p) { &[1, 0] } else { &[0] };
        // 1: U* L+ — U* backtracks from longest until L+ can take at least one.
        for &pre in prefixes {
            let s = p + pre;
            for u in (0..=run(s, &upper)).rev() {
                let l = run(s + u, &lower);
                if l > 0 {
                    let e = s + u + l;
                    return e + contraction(e) - p;
                }
            }
        }
        // 2: U+ L*
        for &pre in prefixes {
            let s = p + pre;
            let u = run(s, &upper);
            if u > 0 {
                let e = s + u + run(s + u, &lower);
                return e + contraction(e) - p;
            }
        }
        // 3: \p{N}{1,3}
        let nn = run(p, &|i| k(i) == Some(Cls::Num)).min(3);
        if nn > 0 {
            return nn;
        }
        // 4: ` ?[^\s\p{L}\p{N}]+[\r\n/]*`
        let punct = |i: usize| matches!(k(i), Some(Cls::Mark | Cls::Other));
        let s = p + usize::from(c(p) == Some(' ') && punct(p + 1));
        let pp = run(s, &punct);
        if pp > 0 {
            let e = s + pp;
            return e + run(e, &|i| matches!(c(i), Some('\r' | '\n' | '/'))) - p;
        }
        // 5-7: whitespace
        let w = run(p, &ws);
        if w == 0 {
            return 1; // unreachable: every class matches one of the alternatives
        }
        if let Some(last) = (p..p + w).rev().find(|&i| matches!(cps[i], '\r' | '\n')) {
            return last + 1 - p;
        }
        if p + w < n && w > 1 {
            w - 1
        } else {
            w
        }
    };

    let mut out = Vec::new();
    let mut p = 0;
    while p < n {
        let len = match_at(p);
        out.push(cps[p..p + len].iter().collect());
        p += len;
    }
    out
}

/// GPT-2's `bytes_to_unicode` and its inverse.
fn gpt2_byte_maps() -> ([char; 256], HashMap<char, u8>) {
    let mut bs: Vec<u16> = Vec::with_capacity(256);
    bs.extend(0x21..=0x7e);
    bs.extend(0xa1..=0xac);
    bs.extend(0xae..=0xff);
    let mut cs: Vec<u16> = bs.clone();
    let mut n = 0u16;
    for b in 0..=255u16 {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(256 + n);
            n += 1;
        }
    }
    let mut to_char = ['\0'; 256];
    let mut to_byte = HashMap::with_capacity(256);
    for (b, c) in bs.iter().zip(cs.iter()) {
        let ch = char::from_u32(*c as u32).expect("valid code point");
        to_char[*b as usize] = ch;
        to_byte.insert(ch, *b as u8);
    }
    (to_char, to_byte)
}
