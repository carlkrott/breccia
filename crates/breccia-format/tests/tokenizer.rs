//! The tokenizer is byte-exact against `llama-tokenize` for this vocabulary.
//!
//! The reference ids were produced by `llama-tokenize --ids --no-bos --no-escape`
//! on a tensor-free copy of the model's metadata (`general.architecture` renamed so
//! an upstream build can load it, `split.count = 1`):
//!
//! ```text
//! llama-tokenize -m vocab-only.gguf --ids --no-bos --no-escape -p "<text>"
//! ```
//!
//! Covers the branches of the `qwen35` pre-tokenizer: contractions, a leading
//! space, trailing spaces, newlines, digits one-per-token, mixed classes, tabs,
//! accented Latin, CJK, and an emoji. Skip if the model is not on this host.
//!
//! `BRECCIA_TOKENIZER_GGUF=<path>` overrides the model path.

use breccia_format::tokenizer::Tokenizer;

const CASES: &[(&str, &[u32])] = &[
    ("Hello, world!", &[9419, 11, 1814, 0]),
    ("  leading", &[220, 6187]),
    ("trailing  ", &[371, 13868, 256]),
    ("a\nb\n\nc", &[64, 198, 65, 271, 66]),
    ("don't", &[14572, 914]),
    ("123 4567", &[16, 17, 18, 220, 19, 20, 21, 22]),
    ("mixed 123abc DEF", &[54789, 220, 16, 17, 18, 13290, 23118]),
    ("tabs\tand spaces", &[29975, 50711, 12258]),
    ("héllo wörld", &[71, 17951, 379, 288, 9137, 497]),
    ("日本語のテキスト", &[247359, 15303, 210342]),
    ("emoji 😀 test", &[36280, 87209, 1228]),
    ("  \n  indented", &[2228, 220, 1214, 15397]),
    (
        "The quick brown fox jumps over the lazy dog.",
        &[760, 3841, 13477, 37550, 33075, 888, 279, 15217, 5388, 13],
    ),
    (
        "def f(x):\n    return x + 1\n",
        &[727, 281, 2007, 1590, 198, 262, 460, 830, 478, 220, 16, 198],
    ),
    ("a  b   c", &[64, 220, 292, 256, 272]),
    ("end.", &[400, 13]),
];

fn model_path() -> Option<std::path::PathBuf> {
    if let Ok(p) = std::env::var("BRECCIA_TOKENIZER_GGUF") {
        return Some(p.into());
    }
    let p = std::path::PathBuf::from(
        "/srv/strata-data/models/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf",
    );
    p.exists().then_some(p)
}

#[test]
fn encode_matches_llama_tokenize() {
    let Some(path) = model_path() else {
        eprintln!("tokenizer: SKIPPED — model not present (set BRECCIA_TOKENIZER_GGUF)");
        return;
    };
    let t = Tokenizer::from_gguf(&path).expect("tokenizer");
    assert_eq!(t.vocab_size(), 248320, "vocab size");
    assert_eq!(t.pre, "qwen35", "pre-tokenizer");
    assert_eq!(t.eos, Some(248046));
    let mut bad = Vec::new();
    for (text, want) in CASES {
        let got = t.encode(text);
        if got != *want {
            bad.push(format!("{text:?}: got {got:?} want {want:?}"));
        }
        // Decode → text → encode must return the same ids for text we produced.
        let back = t.decode(&got);
        let again = t.encode(&back);
        if again != got {
            bad.push(format!("{text:?}: round-trip {again:?} != {got:?}"));
        }
    }
    assert!(bad.is_empty(), "tokenizer mismatches:\n{}", bad.join("\n"));
    // A real UTF-8 file: encode -> decode must be the identity on the bytes, and
    // the ids must be stable under a second round trip. (The benchmark prompt ids
    // are a *sampled* token sequence — not valid UTF-8 text — so they are not a
    // decode/encode fixture.)
    let file = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/tokenizer.rs");
    let text = std::fs::read_to_string(&file).expect("source file");
    let ids = t.encode(&text);
    assert_eq!(t.decode(&ids), text, "encode -> decode changed the text");
    assert_eq!(t.encode(&t.decode(&ids)), ids, "ids are not stable");
    println!(
        "tokenizer: {} bytes of real text round-trip ({} ids)",
        text.len(),
        ids.len()
    );
    println!(
        "tokenizer: {} cases byte-exact against llama-tokenize (vocab {})",
        CASES.len(),
        t.vocab_size()
    );
}
