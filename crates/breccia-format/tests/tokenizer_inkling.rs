//! The `inkling2` tokenizer (GPT-4o pre-tokenizer + `ignore_merges`) is byte-exact
//! against llama.cpp's `llama-tokenize --ids --no-escape` (CPU build of
//! ~/llama.cpp-inkling) on the Inkling-2 GGUF. The cases cover the GPT-4o regex
//! branches: case runs (HELLO, HeLLo, CamelCase), non-ASCII letters that pass both
//! case classes (ÉCOLE), 3-digit groups, `[\r\n/]*` after punctuation, contractions.
//!
//! `BRECCIA_INKLING_GGUF=<path>` overrides the model path; skips if it is absent.
//! `BRECCIA_INKLING_GOLDEN=<dir>` adds the 1415-token prompt-long.txt / tokens-long.txt.

use breccia_format::tokenizer::Tokenizer;

const CASES: &[(&str, &[u32])] = &[
    ("Hello, world!", &[32269, 11, 150770, 0]),
    ("  leading", &[220, 154384]),
    ("trailing  ", &[31229, 279, 149859]),
    ("a\nb\n\nc", &[64, 198, 65, 68014, 66]),
    ("don't", &[1630, 6, 83]),
    ("123 4567", &[70904, 220, 81554, 22]),
    ("mixed 123abc DEF", &[77736, 220, 70904, 33782, 199206]),
    ("tabs\tand spaces", &[4338, 82, 197, 276, 156465]),
    ("héllo wörld", &[71378, 31835, 149997, 68175, 483]),
    (
        "日本語のテキスト",
        &[90626, 22912, 6084, 2804, 2064, 3527, 3529],
    ),
    ("emoji 😀 test", &[40688, 220, 110689, 150170]),
    ("  \n  indented", &[149865, 220, 181070, 302]),
    (
        "The quick brown fox jumps over the lazy dog.",
        &[
            371, 157021, 161319, 220, 32329, 160581, 82, 150053, 149876, 161703, 153350, 13,
        ],
    ),
    (
        "def f(x):\n    return x + 1\n",
        &[
            1719, 149983, 7, 87, 8, 25, 198, 149862, 150583, 149924, 196986, 220, 16, 198,
        ],
    ),
    ("a  b   c", &[64, 220, 149963, 149859, 149972]),
    ("end.", &[406, 13]),
    ("HELLO world", &[146066, 148284, 150770]),
    ("HeLLo", &[1198, 43, 1211]),
    ("ÉCOLE Normale", &[5195, 145889, 145876, 159695, 1115]),
    ("1234567", &[70904, 81554, 22]),
    (
        "path/to/file\r\n\nnext",
        &[1448, 14, 278, 14, 1082, 117762, 2157],
    ),
    (
        "He'LL say they'RE",
        &[1198, 6, 145896, 150657, 150051, 6, 145868],
    ),
    ("CamelCaseWords", &[47554, 4474, 103634]),
    (
        "x = [1, 2];\n\n\n// done",
        &[
            87, 196977, 196989, 16, 11, 220, 17, 60, 26, 68019, 145055, 151368,
        ],
    ),
    ("naïve café", &[1364, 5169, 290, 162389]),
    (
        "Ünïcödé ÀBC",
        &[5214, 77, 138472, 5176, 68096, 154549, 145944],
    ),
    ("\t\t tab", &[68038, 152376]),
    (" !!! ?? .\n", &[186093, 197038, 196981, 198]),
];

fn tokenizer() -> Option<Tokenizer> {
    let p = std::env::var("BRECCIA_INKLING_GGUF").unwrap_or_else(|_| {
        "/srv/strata-data/models/inkling2-unbound-v2-breccia/Inkling-2-Mini-Unbound-v2-breccia-Q4_K_M.gguf".into()
    });
    if !std::path::Path::new(&p).exists() {
        eprintln!("skip: {p} not on this host");
        return None;
    }
    Some(Tokenizer::from_gguf(&p).expect("tokenizer loads"))
}

#[test]
fn inkling_encode_matches_llama_tokenize() {
    let Some(tok) = tokenizer() else { return };
    assert_eq!(tok.pre, "inkling2");
    let mut bad = 0;
    for (text, want) in CASES {
        let got = tok.encode(text);
        if got != *want {
            eprintln!("{text:?}\n  got  {got:?}\n  want {want:?}");
            bad += 1;
        }
        assert_eq!(tok.decode(&got), *text, "round trip");
    }
    assert_eq!(bad, 0, "{bad} cases differ");
}

#[test]
fn inkling_encode_long_prompt() {
    let Ok(dir) = std::env::var("BRECCIA_INKLING_GOLDEN") else {
        return;
    };
    let Some(tok) = tokenizer() else { return };
    let dir = std::path::Path::new(&dir);
    let text = std::fs::read_to_string(dir.join("prompt-long.txt")).unwrap();
    let want: Vec<u32> = std::fs::read_to_string(dir.join("tokens-long.txt"))
        .unwrap()
        .trim()
        .trim_matches(|c| c == '[' || c == ']')
        .split(',')
        .map(|s| s.trim().parse().unwrap())
        .collect();
    let got = tok.encode(&text);
    let first = got.iter().zip(&want).position(|(a, b)| a != b);
    assert!(
        got == want,
        "len {} vs {}, first diff at {first:?}",
        got.len(),
        want.len()
    );
}
