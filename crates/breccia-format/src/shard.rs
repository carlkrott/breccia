//! Multi-shard support: one logical tensor namespace over N GGUF files.
//!
//! The real model splits as shard 1 = 67 KVs / 1223 tensors and shard 2 =
//! 3 KVs / 1 tensor (`per_layer_token_embd.weight`), with `split.tensors.count`
//! = 1224 on both. Metadata lives on shard 1 only, so [`ShardSet::metadata`]
//! reads the primary shard.

use crate::error::{FormatError, Result};
use crate::gguf::{Gguf, SplitRef, TensorView};
use crate::layout::ModelLayout;
use crate::metadata::MetadataMap;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Validated split description of a [`ShardSet`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SplitInfo {
    /// Number of shards.
    pub count: u32,
    /// `split.tensors.count` as declared by the shards (`1224` for this model).
    pub declared_tensors: Option<u64>,
    /// Tensor count of each shard, in split order.
    pub shard_tensors: Vec<usize>,
}

impl SplitInfo {
    pub fn total_tensors(&self) -> usize {
        self.shard_tensors.iter().sum()
    }
}

/// N shards of one split model, addressed as a single tensor namespace.
pub struct ShardSet {
    shards: Vec<Gguf>,
    index: HashMap<Box<str>, (u16, u32)>,
    split: SplitInfo,
}

impl ShardSet {
    /// Open a single, non-split GGUF.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        Self::open_paths(&[path.as_ref().to_path_buf()])
    }

    /// Open an explicit, ordered list of shards and validate `split.*`.
    pub fn open_paths(paths: &[PathBuf]) -> Result<Self> {
        if paths.is_empty() {
            return Err(FormatError::SplitMismatch("no shard paths given".into()));
        }
        let mut shards = Vec::with_capacity(paths.len());
        for p in paths {
            shards.push(Gguf::open(p)?);
        }
        Self::assemble(shards)
    }

    /// Open every file matching `pattern` (`*` / `?` wildcards, e.g.
    /// `…-0000*-of-00002.gguf`), then validate the split.
    pub fn open_glob(pattern: &str) -> Result<Self> {
        let paths = expand_glob(pattern)?;
        if paths.is_empty() {
            return Err(FormatError::split(format!(
                "glob '{pattern}' matched no files"
            )));
        }
        Self::open_paths(&paths)
    }

    /// Open `path` plus its siblings, discovered from the
    /// `-NNNNN-of-MMMMM` file-name convention and the file's own `split.count`.
    ///
    /// The already-mmap'd header of `path` is reused (no double parse).
    pub fn open_family(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let first = Gguf::open(path)?;
        let split = match first.split() {
            Some(s) if s.count > 1 => s,
            _ => return Self::assemble(vec![first]),
        };
        let paths = sibling_paths(path, split.count)?;
        let mut first = Some(first);
        let mut shards = Vec::with_capacity(paths.len());
        for (i, p) in paths.iter().enumerate() {
            if i as u32 == split.no {
                shards.push(first.take());
            } else {
                shards.push(Some(Gguf::open(p)?));
            }
        }
        let shards = shards
            .into_iter()
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| FormatError::split("internal: missing shard after family expansion"))?;
        Self::assemble(shards)
    }

    fn assemble(mut shards: Vec<Gguf>) -> Result<Self> {
        let n = shards.len();
        let refs: Vec<Option<SplitRef>> = shards.iter().map(Gguf::split).collect();

        // Split-order the shards and validate consistency.
        let has_any_split = refs.iter().any(Option::is_some);
        if has_any_split {
            let mut order: Vec<usize> = Vec::with_capacity(n);
            for (i, r) in refs.iter().enumerate() {
                let r = r.ok_or_else(|| {
                    FormatError::split(format!(
                        "{}: no split.* metadata while other shards declare a split",
                        shards[i].path().display()
                    ))
                })?;
                if r.count as usize != n {
                    return Err(FormatError::split(format!(
                        "{}: split.count {} but {n} shard(s) opened",
                        shards[i].path().display(),
                        r.count
                    )));
                }
                if r.no as usize >= n {
                    return Err(FormatError::split(format!(
                        "{}: split.no {} out of range for {n} shards",
                        shards[i].path().display(),
                        r.no
                    )));
                }
                order.push(i);
            }
            order.sort_by_key(|&i| refs[i].map_or(0, |r| r.no));
            for (pos, &i) in order.iter().enumerate() {
                if refs[i].map_or(0, |r| r.no) as usize != pos {
                    return Err(FormatError::split(format!(
                        "duplicate or non-contiguous split.no (shard {} claims {})",
                        i,
                        refs[i].map_or(0, |r| r.no)
                    )));
                }
            }
            let mut slots: Vec<Option<Gguf>> = shards.into_iter().map(Some).collect();
            let mut sorted = Vec::with_capacity(n);
            for &i in &order {
                if let Some(shard) = slots[i].take() {
                    sorted.push(shard);
                }
            }
            shards = sorted;
        } else if n > 1 {
            return Err(FormatError::split(format!(
                "{n} shards opened but none declares split.* metadata"
            )));
        }

        let declared: Vec<Option<u64>> = refs.iter().flatten().map(|r| r.tensors_count).collect();
        let declared_tensors = match declared.first() {
            Some(first) => {
                if declared.iter().any(|d| d != first) {
                    return Err(FormatError::split("shards disagree on split.tensors.count"));
                }
                *first
            }
            None => None,
        };

        // One index over every shard; duplicate names are a hard error because
        // the namespace must stay unambiguous.
        let mut index: HashMap<Box<str>, (u16, u32)> = HashMap::new();
        let mut shard_tensors = Vec::with_capacity(shards.len());
        for (si, shard) in shards.iter().enumerate() {
            shard_tensors.push(shard.tensor_count());
            for (ti, name) in shard.tensor_names().enumerate() {
                let key: Box<str> = name.into();
                if let Some(prev) = index.insert(key, (si as u16, ti as u32)) {
                    return Err(FormatError::DuplicateTensor(format!(
                        "{name} (shard {} tensor {}, also shard {} tensor {})",
                        prev.0, prev.1, si, ti
                    )));
                }
            }
        }

        let split = SplitInfo {
            count: shards.len() as u32,
            declared_tensors,
            shard_tensors,
        };
        if let Some(declared) = split.declared_tensors {
            let actual = split.total_tensors() as u64;
            if declared != actual {
                return Err(FormatError::split(format!(
                    "split.tensors.count = {declared} but {actual} tensors found across {} shards",
                    split.count
                )));
            }
        }
        Ok(Self {
            shards,
            index,
            split,
        })
    }

    /// Shards in split order (`parts()[0]` is the primary shard).
    pub fn parts(&self) -> &[Gguf] {
        &self.shards
    }

    /// Shard 0: carries the model metadata (67 KVs for this model).
    pub fn primary(&self) -> &Gguf {
        &self.shards[0]
    }

    pub fn split(&self) -> &SplitInfo {
        &self.split
    }

    /// Metadata of the primary shard.
    pub fn metadata(&self) -> Result<MetadataMap<'_>> {
        self.primary().metadata()
    }

    pub fn architecture(&self) -> Option<&str> {
        self.primary().architecture()
    }

    /// Tensor lookup across all shards.
    pub fn tensor(&self, name: &str) -> Option<TensorView<'_>> {
        let &(si, ti) = self.index.get(name)?;
        let shard = self.shards.get(si as usize)?;
        shard.tensor_by_index(ti as usize)
    }

    /// Every tensor, shard by shard (primary first).
    pub fn tensors(&self) -> impl Iterator<Item = TensorView<'_>> {
        self.shards.iter().flat_map(|s| s.tensors())
    }

    pub fn tensor_count(&self) -> usize {
        self.index.len()
    }

    /// Every tensor name across shards (primary first).
    pub fn tensor_names(&self) -> impl Iterator<Item = &str> {
        self.shards.iter().flat_map(|s| s.tensor_names())
    }

    /// Sum of tensor payload bytes across shards.
    pub fn total_tensor_bytes(&self) -> u64 {
        self.shards.iter().map(Gguf::total_tensor_bytes).sum()
    }

    /// Group-classify every tensor into dense / experts / PLE / embedding /
    /// output (see [`ModelLayout`]).
    pub fn layout(&self) -> Result<ModelLayout<'_>> {
        ModelLayout::from_shards(self)
    }
}

impl std::fmt::Debug for ShardSet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ShardSet")
            .field("split", &self.split)
            .field("tensors", &self.index.len())
            .field("bytes", &self.total_tensor_bytes())
            .field(
                "paths",
                &self
                    .shards
                    .iter()
                    .map(|s| s.path().display().to_string())
                    .collect::<Vec<_>>(),
            )
            .finish()
    }
}

/// Expand a glob pattern (`*`, `?`) into sorted paths.
///
/// A minimal implementation (no dependency): components without wildcards are
/// used verbatim, wildcard components are matched against directory listings.
pub fn expand_glob(pattern: &str) -> Result<Vec<PathBuf>> {
    let absolute = pattern.starts_with('/');
    let mut current: Vec<PathBuf> = vec![if absolute {
        PathBuf::from("/")
    } else {
        PathBuf::new()
    }];

    for comp in pattern.split('/').filter(|c| !c.is_empty() && *c != ".") {
        if comp.contains(['*', '?']) {
            let mut next = Vec::new();
            for base in &current {
                let dir = if base.as_os_str().is_empty() {
                    Path::new(".")
                } else {
                    base.as_path()
                };
                let entries = std::fs::read_dir(dir).map_err(|e| FormatError::io(dir, e))?;
                for entry in entries {
                    let entry = entry.map_err(|e| FormatError::io(dir, e))?;
                    if let Some(name) = entry.file_name().to_str() {
                        if wildcard_match(comp, name) {
                            next.push(base.join(name));
                        }
                    }
                }
            }
            current = next;
        } else {
            for p in &mut current {
                p.push(comp);
            }
        }
    }
    current.sort();
    Ok(current)
}

/// `*` matches any run of characters, `?` exactly one.
pub fn wildcard_match(pattern: &str, text: &str) -> bool {
    let p: Vec<char> = pattern.chars().collect();
    let t: Vec<char> = text.chars().collect();
    let (mut pi, mut ti) = (0usize, 0usize);
    let mut star: Option<usize> = None;
    let mut mark = 0usize;
    while ti < t.len() {
        if pi < p.len() && (p[pi] == '?' || p[pi] == t[ti]) {
            pi += 1;
            ti += 1;
        } else if pi < p.len() && p[pi] == '*' {
            star = Some(pi);
            pi += 1;
            mark = ti;
        } else if let Some(s) = star {
            pi = s + 1;
            mark += 1;
            ti = mark;
        } else {
            return false;
        }
    }
    while pi < p.len() && p[pi] == '*' {
        pi += 1;
    }
    pi == p.len()
}

/// Sibling shard paths for `-NNNNN-of-MMMMM.gguf` naming.
pub fn sibling_paths(path: &Path, count: u32) -> Result<Vec<PathBuf>> {
    let name = path
        .file_name()
        .and_then(|n| n.to_str())
        .ok_or_else(|| FormatError::split(format!("{}: not a file path", path.display())))?;
    let of_pos = name.rfind("-of-").ok_or_else(|| {
        FormatError::split(format!(
            "{}: cannot derive sibling shard names (no '-of-' in the file name)",
            path.display()
        ))
    })?;
    let digits_start = name[..of_pos]
        .rfind(|c: char| !c.is_ascii_digit())
        .map_or(0, |i| i + 1);
    if digits_start == of_pos {
        return Err(FormatError::split(format!(
            "{}: no shard index digits before '-of-'",
            path.display()
        )));
    }
    let width = of_pos - digits_start;
    let head = &name[..digits_start];
    let tail = &name[of_pos..];
    let mut out = Vec::with_capacity(count as usize);
    for i in 1..=count {
        let candidate = format!("{head}{i:0width$}{tail}");
        let p = path.with_file_name(candidate);
        if !p.is_file() {
            return Err(FormatError::split(format!(
                "missing shard {} (split.count = {count})",
                p.display()
            )));
        }
        out.push(p);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant::GgmlType;
    use crate::testutil::{Builder, TempDir, Value};

    fn shard_builder(no: u16, count: u16, tensors_count: i32, tensors: &[&str]) -> Builder {
        let mut b = Builder::new()
            .kv("general.architecture", Value::Str("qwen4exp".into()))
            .kv("split.no", Value::U16(no))
            .kv("split.count", Value::U16(count))
            .kv("split.tensors.count", Value::I32(tensors_count));
        for name in tensors {
            b = b.tensor(name, &[8], GgmlType::F32.id(), vec![no as u8; 32]);
        }
        b
    }

    fn two_shards(dir: &TempDir, total: i32) -> (PathBuf, PathBuf) {
        let a = dir.join("m-00001-of-00002.gguf");
        let b = dir.join("m-00002-of-00002.gguf");
        shard_builder(0, 2, total, &["a.weight", "b.weight"]).write(&a);
        shard_builder(1, 2, total, &["per_layer_token_embd.weight"]).write(&b);
        (a, b)
    }

    #[test]
    fn opens_two_shards_as_one_namespace() {
        let dir = TempDir::new("shard-two");
        let (a, b) = two_shards(&dir, 3);
        // Deliberately reversed order: the set must sort by split.no.
        let set = ShardSet::open_paths(&[b.clone(), a.clone()]).unwrap();
        assert_eq!(set.split().count, 2);
        assert_eq!(set.split().total_tensors(), 3);
        assert_eq!(set.split().shard_tensors, vec![2, 1]);
        assert_eq!(set.primary().path(), a);
        assert_eq!(set.tensor_count(), 3);
        assert_eq!(set.architecture(), Some("qwen4exp"));
        assert_eq!(set.metadata().unwrap().len(), 4);
        for name in ["a.weight", "b.weight", "per_layer_token_embd.weight"] {
            assert!(set.tensor(name).is_some(), "{name} missing");
        }
        // Shard 1 bytes are 0x00, shard 2's are 0x01.
        assert_eq!(set.tensor("b.weight").unwrap().as_slice(), &[0u8; 32][..]);
        assert_eq!(
            set.tensor("per_layer_token_embd.weight")
                .unwrap()
                .as_slice(),
            &[1u8; 32][..]
        );
        assert_eq!(set.tensors().count(), 3);
        assert_eq!(set.total_tensor_bytes(), 96);
    }

    #[test]
    fn open_family_finds_siblings_and_glob_works() {
        let dir = TempDir::new("shard-family");
        let (a, _b) = two_shards(&dir, 3);
        let set = ShardSet::open_family(&a).unwrap();
        assert_eq!(set.split().count, 2);
        assert_eq!(set.tensor_count(), 3);
        // Re-opening a non-primary shard yields the same set.
        let other = ShardSet::open_family(dir.join("m-00002-of-00002.gguf")).unwrap();
        assert_eq!(other.tensor_count(), 3);

        let pattern = format!("{}/*-of-00002.gguf", dir.path().display());
        let globbed = ShardSet::open_glob(&pattern).unwrap();
        assert_eq!(globbed.tensor_count(), 3);

        assert!(matches!(
            ShardSet::open_glob(&format!("{}/nothing-*.gguf", dir.path().display())),
            Err(FormatError::SplitMismatch(_))
        ));
    }

    #[test]
    fn duplicate_split_index_is_rejected() {
        let dir = TempDir::new("shard-dup-no");
        let a = dir.join("a.gguf");
        let b = dir.join("b.gguf");
        shard_builder(0, 2, 2, &["a.weight"]).write(&a);
        shard_builder(0, 2, 2, &["b.weight"]).write(&b);
        assert!(matches!(
            ShardSet::open_paths(&[a, b]),
            Err(FormatError::SplitMismatch(_))
        ));
    }

    #[test]
    fn split_count_and_tensor_total_are_validated() {
        let dir = TempDir::new("shard-bad-count");
        let a = dir.join("a.gguf");
        let b = dir.join("b.gguf");
        shard_builder(0, 3, 2, &["a.weight"]).write(&a);
        shard_builder(1, 2, 2, &["b.weight"]).write(&b);
        assert!(matches!(
            ShardSet::open_paths(&[a, b]),
            Err(FormatError::SplitMismatch(_))
        ));

        let dir = TempDir::new("shard-bad-total");
        let a = dir.join("a.gguf");
        let b = dir.join("b.gguf");
        shard_builder(0, 2, 99, &["a.weight"]).write(&a);
        shard_builder(1, 2, 99, &["b.weight"]).write(&b);
        let err = ShardSet::open_paths(&[a, b]).unwrap_err();
        assert!(
            err.to_string().contains("split.tensors.count"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn multiple_files_without_split_metadata_are_rejected() {
        let dir = TempDir::new("shard-plain");
        let a = dir.join("a.gguf");
        let b = dir.join("b.gguf");
        Builder::new()
            .tensor("a", &[8], GgmlType::F32.id(), vec![0; 32])
            .write(&a);
        Builder::new()
            .tensor("b", &[8], GgmlType::F32.id(), vec![0; 32])
            .write(&b);
        assert!(matches!(
            ShardSet::open_paths(&[a, b]),
            Err(FormatError::SplitMismatch(_))
        ));
        // A single file still opens fine.
        let one = ShardSet::open(dir.join("a.gguf")).unwrap();
        assert_eq!(one.split().count, 1);
    }

    #[test]
    fn duplicate_names_across_shards_are_rejected() {
        let dir = TempDir::new("shard-dup-name");
        let a = dir.join("a.gguf");
        let b = dir.join("b.gguf");
        shard_builder(0, 2, 2, &["same.weight"]).write(&a);
        shard_builder(1, 2, 2, &["same.weight"]).write(&b);
        assert!(matches!(
            ShardSet::open_paths(&[a, b]),
            Err(FormatError::DuplicateTensor(name)) if name.starts_with("same.weight")
        ));
    }

    #[test]
    fn wildcard_matcher() {
        assert!(wildcard_match("*-of-00002.gguf", "m-00001-of-00002.gguf"));
        assert!(wildcard_match("a?c", "abc"));
        assert!(!wildcard_match("a?c", "ac"));
        assert!(wildcard_match("*", "anything"));
        assert!(wildcard_match("abc", "abc"));
        assert!(!wildcard_match("abc", "abcd"));
        assert!(wildcard_match("a*b*c", "aXXbYYc"));
    }

    #[test]
    fn sibling_path_detection() {
        let dir = TempDir::new("shard-sibling");
        let (a, _) = two_shards(&dir, 3);
        let paths = sibling_paths(&a, 2).unwrap();
        assert_eq!(paths[0], a);
        assert!(paths[1].ends_with("m-00002-of-00002.gguf"));
        assert!(sibling_paths(&a, 3).is_err());
        assert!(sibling_paths(&dir.join("plain.gguf"), 2).is_err());
    }
}
