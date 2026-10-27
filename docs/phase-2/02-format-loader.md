# Phase 2 — `breccia-format`: zero-copy loader (GGUF v3 + strata pack + MTP)

Report for the ingestion layer of the engine (Phase 3 M2/M3 dependency). Owner: `crates/breccia-format/**`.
All numbers below were produced on this machine on 2026-10-26 against the real 75 GB IQ3_XXS shards;
nothing is copied from a README without a run.

**Status: complete.** `cargo build --workspace`, `cargo test -p breccia-format`,
`cargo clippy -p breccia-format --all-targets -- -D warnings`, `cargo fmt --check` are green, and
`examples/inspect` validates both shards plus the pack and MTP runtime against
`docs/specs/model-artifacts-inventory.md`.

---

## 1. What the crate is

`breccia-format` is the only place in the engine that knows what a model file looks like. It never
copies tensor bytes: every payload is a `&[u8]` slice of a read-only `memmap2` mapping, and opening
the 75 GB model reads only the header pages (≈11 MB of metadata for shard 1).

| API | Purpose |
|---|---|
| `Gguf::open(path)` | map + parse one GGUF: magic/version, typed metadata (`u8…u64`, `i8…i64`, `f32/f64`, `bool`, `string`, arrays incl. arrays of strings), tensor table (name, `n_dims`, dims, ggml type, offset), `general.alignment` (default 32), `data_offset` = aligned header end, validation that every tensor range ends inside the file, `data_end`/`trailing_bytes` |
| `Gguf::metadata() -> MetadataMap` | all KVs, in file order, borrowed from the mapping; arrays (`MetadataArray`) decode lazily and offer `as_strings`, `to_u64_vec`, `to_i64_vec`, `to_f64_vec`, `to_bool_vec`, `iter`, `get` |
| `Gguf::tensor(name)` / `tensors()` / `tensor_by_index(i)` | `TensorView` views |
| `TensorView` | `name()`, `dims()/ne0..ne(n)`, `n_elements()`, `n_bytes()`, `ty()`, `data_offset()`, `file_range()`, `shape()`, `as_slice() -> &'a [u8]` |
| `ShardSet` | N shards as one tensor namespace; `open`, `open_paths`, `open_glob` (built-in `*`/`?` matcher, no extra dependency), `open_family` (finds `-0000N-of-0000M.gguf` siblings from the file's own `split.count` and reuses the already-mapped first shard) |
| `ShardSet::metadata()` | metadata of shard 0 (the only shard carrying model KVs) |
| `ShardSet::layout() -> ModelLayout` | split-residency classification with per-group byte totals and quant histograms |
| `ExpertBlock` | per-block routed-expert geometry: `n_experts`, `bytes_per_expert(role)`, `blob_bytes()`, `blob_offsets()`, `expert_ranges(e)`, `expert_slices(e)` |
| `Pack::open(dir)` | strata native pack (index v3, `native_experts.txt`, `dense.bin`, `experts.bin`, sidecars) |
| `MtpRt::open(dir)` | MTP draft runtime (`dense.txt`, `dense.bin`, `experts.bin`, `draft_vocab.bin`) |
| `GgmlType` | ggml type id → name + block geometry, including the fork-only `Q2_0 = 64/18` |

Design notes worth keeping:

* **Metadata is lazy and borrowed.** `metadata()` re-decodes 67 KV values from recorded byte spans on
  each call (cheap) and hands out `&'a str` / `MetadataArray<'a>` that point into the mapping. The
  247,587-entry tokenizer merge list costs nothing unless asked for. It returns `Result` because value
  decoding is fallible in principle (all values are validated during `open`).
* **Names are owned, payloads are not.** Tensor names and metadata keys are small `Box<str>`; keeping
  them owned avoids self-referential lifetimes and makes `TensorView` a plain borrow of the mapping.
* **Split validation is strict.** shard set opens only if every shard declares `split.count == N`, each
  `split.no` is unique and contiguous, `split.tensors.count` agrees across shards and equals the sum of
  the per-shard tensor counts, and no tensor name appears twice.
* **No panics on malformed input.** Zero `unwrap`/`expect`/`panic!` in non-test code (checked by grep);
  every failure is a typed `FormatError` (`BadMagic`, `UnsupportedVersion`, `Truncated`, `BadUtf8`,
  `UnknownValueType`, `NestedArray`, `BadAlignment`, `TooManyDims`, `UnsupportedTensorType`,
  `ShapeNotAligned`, `TensorOutOfBounds`, `DuplicateTensor`, `SplitMismatch`, `MissingMetadata`,
  `MetadataType`, `ArrayType`, `UnknownTensor`, `ExpertOutOfRange`, `ExpertGeometry`, `Malformed`, `Io`).
  Implausible `tensor_count`/`kv_count` are rejected before allocating.
* **Edge cases handled explicitly:** zero-tensor files (no data section, no padding — `data_offset` is
  the file end), zero-size tensors (`dims` containing 0 → empty slice, following tensor unaffected),
  alignment ≠ 32 (must be a non-zero power of two), unknown ggml type ids (reported with the tensor
  name), nested metadata arrays (refused).

---

## 2. Group classification (split residency)

Classification is by name only (`classify()`), matching the layout doc §2.4:

| Group | Names | Residency intent (M2/M3) | Measured here |
|---|---|---|---|
| `Embedding` | `token_embd.weight` | always resident | 1 tensor, 273,152,000 B, IQ3_S |
| `Output` | `output.weight` | always resident (never tied) | 1 tensor, 437,043,200 B, Q5_K |
| `Dense` | `attn_qkv`, `attn_q/k/v/output`, `attn_gate`, `attn_{q,k}_norm`, `ssm_*`, `hc_*`, `output_hc_*`, `indexer.*`, all norms, router `ffn_gate_inp*`, shared expert `ffn_*_shexp` | resident | 1077 tensors, 3,405,667,840 B |
| `Experts` | `blk.N.ffn_{gate,up,down}_exps.weight` | streamed / cached (Phase 4 LRU) | 144 tensors, **42,912,972,800 B** |
| `Ple` | `per_layer_token_embd.weight` | its own table (28.8 GB, shard 2) | 1 tensor, 28,800,138,240 B, IQ4_NL |

Per-group quant histograms are exposed (`GroupSummary::types`, `types_summary()`), e.g. dense =
`BF16×484 F32×292 Q6_K×71 IQ4_XS×67 IQ3_S×55 Q4_K×45 IQ4_NL×40 Q5_K×14 Q2_0×8 F16×1`.

### 2.1 Routed experts: the layout question, answered empirically

The GGUF stores experts **role-major**: three tensors per block, `[ne0, ne1, 512]`, so expert *e* is a
contiguous slice *inside each role tensor*, but the three roles are not adjacent in the file. The
pack's `experts.bin` stores them **expert-major**: `[gate rows | up rows | down rows]` per expert.
Therefore:

* `ExpertBlock::expert_slices(e) -> ExpertSlices { gate, up, down }` — three zero-copy slices in the
  documented `[gate|up|down]` order (this is the general GGUF path; no copy, no concatenation).
* `Pack::expert_blob(layer, e) -> &[u8]` — the contiguous 1,868,800-B blob from `experts.bin`.
* `Pack::expert_slices(layer, e, n_embd, n_ff)` — that blob split into the same three roles.

Verified two ways on the real files:

1. `blob_bytes` from GGUF geometry equals the pack's declared `blob_bytes` for all 48 layers
   (checked inside `Pack::validate_against`), e.g. layer 0: 473,600 (gate) + 473,600 (up) + 921,600
   (down) = 1,868,800 ✓; Σ over 48 layers × 512 experts = 42,912,972,800 = `experts.bin` size ✓.
2. `experts.bin` bytes are **byte-identical** to the GGUF slices in that order at layers 0, 1, 47 for
   experts 0, 1 and 511 (`inspect` prints `[ok] experts.bin is byte-identical to the GGUF …`).

Per-block type distribution (48 blocks, from `--all-blocks`; gate/up and down are uniform per layer):

```
8 IQ2_XXS/Q2_0   7 IQ3_S/Q2_0   6 IQ3_S/IQ4_NL   6 IQ2_XS/Q2_0   6 IQ2_S/Q2_0
4 IQ2_XS/IQ4_NL  4 IQ2_S/IQ4_NL 3 IQ3_XXS/Q2_0   3 IQ3_XXS/IQ4_NL 1 IQ2_XXS/IQ4_NL
```

Down projection: Q2_0 on 30 layers, IQ4_NL on 18 ✓ (§2.5). Gate/up: IQ2_XXS 9, IQ2_XS 10, IQ3_XXS 6,
IQ3_S 13, IQ2_S 10 ✓ (§4.2 `native_experts.txt`).

---

## 3. Quant metadata

`GgmlType` is a newtype over the raw `u32`, so unknown ids survive to the error message instead of being
remapped. Block geometry comes from the verified table (§2.5) and is used to size tensors, not read from
the tensor table's offsets:

| id | type | block (elems, bytes) | note |
|---:|---|---|---|
| 0 / 1 / 30 | F32 / F16 / BF16 | 1/4, 1/2, 1/2 | |
| 12 / 13 / 14 | Q4_K / Q5_K / Q6_K | 256/144, 256/176, 256/210 | |
| 16 / 17 / 21 / 22 | IQ2_XXS / IQ2_XS / IQ3_S / IQ2_S | 256/66, 256/74, 256/110, 256/82 | |
| 18 / 20 / 23 | IQ3_XXS / IQ4_NL / IQ4_XS | 256/98, 32/18, 256/136 | |
| 2 / 6 / 8 / 10 / 11 / 15 | Q4_0 / Q5_0 / Q8_0 / Q2_K / Q3_K / Q8_K | for the Q2_0 release (§3) | |
| 19 / 29 / 34 / 35 | IQ1_S / IQ1_M / TQ1_0 / TQ2_0 | other releases | |
| **42** | **Q2_0 (fork)** | **64/18 = 2.25 bpw** | 16 B 2-bit codes + 2 B fp16 scale; **not** upstream ggml's Q2_0 |
| 41 | Q1_0 (fork) | **no geometry** | id known, block size unverified → `UnsupportedTensorType` if ever met |

`GgmlType::checked_bytes(elements)` returns `None` when the element count is not a whole number of
blocks; `Gguf::open` turns that into `ShapeNotAligned` (with the tensor name) rather than silently
truncating.

---

## 4. Validation against the real shards

Command (release build, `<REPO_ROOT>`):

```bash
cargo build -p breccia-format --release --example inspect
./target/release/examples/inspect \
    '/srv/strata-data/models/IQ3_XXS/*-0000?-of-00002.gguf' \
    --pack /srv/strata-data/packs/iq3_xxs \
    --mtp  /srv/strata-data/mtp/rt
```

Exit code 0; elapsed **0.006 s** warm (0.118 s on the first, cold run that faults in `experts.bin`
pages while byte-comparing blobs) — i.e. header-only work, not a 75 GB read.

### 4.1 Inventory cross-check

| quantity | inventory (§) | measured | how |
|---|---|---|---|
| shard 1 KV / tensors | 67 / 1223 (§2.1) | 67 / 1223 | `Gguf::kv_count`, `tensor_count` |
| shard 2 KV / tensors | 3 / 1 (§2.1) | 3 / 1 | same |
| `data_offset` shard 1/2 | 11,024,256 / 192 (§2.1) | 0xa83780 / 0xc0 | aligned header end |
| data section end | exactly EOF, 0 trailing (§2.1) | `data_end == file_len`, 0 trailing, both shards | `trailing_bytes()` asserted |
| alignment | 32 (default, no `general.alignment`) (§2.1) | 32 | `alignment()` |
| `split.*` | no=0/1, count=2, tensors.count=1224 (§2.2) | count 2, declared 1224, per-shard `[1223,1]` | `SplitInfo` |
| expert bytes (144 tensors) | 42,912,972,800 (§2.1/§2.5) | 42,912,972,800 | Σ `TensorView::n_bytes` over `Experts` |
| Σ blob×512 | 42,912,972,800 (§4.2) | 42,912,972,800 | `ExpertBlock` geometry |
| blocks × experts | 48 × 512 (§2.3/§4.2) | 48 × 512 | `ModelLayout` |
| blob/expert blk.0 / blk.1 | 1,868,800 / 1,305,600 (§4.3) | 1,868,800 / 1,305,600 | block geometry |
| blk.0 types gate/up/down | IQ2_XS / IQ4_NL (§4.2 row 0) | IQ2_XS / IQ4_NL | tensor types |
| PLE table | `[160, 320001536]`, IQ4_NL, shard 2, 28,800,138,240 B (§2.6) | identical | `ModelLayout::ple` + shard-2 lookup |
| `blk.1` PLE module | `ple_key/value/conv1d/norm_*` only in blk.1 (§2.4) | all 6 present, none in blk.0 | name scan |
| router | `blk.0.ffn_gate_inp.weight` `[2560,512]` BF16 (§2.4) | identical | |
| embedding / output | `[2560,248320]` IQ3_S / Q5_K (§2.5) | identical, 273,152,000 / 437,043,200 B | |
| total tensor bytes | — | 75,828,974,080 (70.62 GiB) | Σ over both shards |

Non-reference artifacts are opened without false assertions: the vision projector
(`mmproj-Qwen3.8-Flash-Next-BF16.gguf`) reports 31 KVs / 334 tensors / alignment 32 / data ending at
EOF, all `Dense`, `BF16×110 F32×224`, 907,523,008 B — matching §6 of the inventory.

### 4.2 Output (excerpt)

```
== container ==
shard 1: …IQ3_XXS-00001-of-00002.gguf
  version 3  kv 67  tensors 1223  alignment 32  data_offset 0xa83780  data_end 0xaf3cb4d80 (file 0xaf3cb4d80, trailing 0 B)
shard 2: …IQ3_XXS-00002-of-00002.gguf
  version 3  kv 3  tensors 1  alignment 32  data_offset 0xc0  data_end 0x6b49f3cc0 (file 0x6b49f3cc0, trailing 0 B)
split: count 2  declared tensors Some(1224)  per-shard tensors [1223, 1]
tensors: 1224 total, 75828974080 bytes (70.62 GiB)
[ok] container matches inventory §2.1/§2.2 (67+3 KVs, 1223+1 tensors, align 32)

== residency groups ==
group      tensors            bytes       GiB  types
embedding        1        273152000     0.254  IQ3_S×1
output           1        437043200     0.407  Q5_K×1
dense         1077       3405667840     3.172  BF16×484 F32×292 Q6_K×71 IQ4_XS×67 IQ3_S×55 Q4_K×45 IQ4_NL×40 Q5_K×14 Q2_0×8 F16×1
experts        144      42912972800    39.966  Q2_0×30 IQ3_S×26 IQ2_XS×20 IQ2_S×20 IQ2_XXS×18 IQ4_NL×18 IQ3_XXS×12
ple              1      28800138240    26.822  IQ4_NL×1
TOTAL         1224      75828974080    70.621

== routed experts (per block) ==
blk    gate/up        down    gate/expert    up/expert  down/expert  blob/expert
0      IQ2_XS         IQ4_NL       473600       473600       921600      1868800
1      IQ2_XXS        Q2_0         422400       422400       460800      1305600
  … 45 blocks omitted (--all-blocks to show) …
47     IQ3_XXS        IQ4_NL       627200       627200       921600      2176000
total: 48 blocks, 512 experts each, 42912972800 B of expert weights (39.97 GiB), Σ blob×experts = 42912972800

== key tensors ==
  token_embd.weight                  [2560, 248320]         IQ3_S         273152000 B
  output.weight                      [2560, 248320]         Q5_K          437043200 B
  per_layer_token_embd.weight        [160, 320001536]       IQ4_NL      28800138240 B
  blk.0.ffn_gate_inp.weight          [2560, 512]            BF16            2621440 B
  blk.1.ple_key.weight               [2560, 10240]          BF16           52428800 B
  blk.1.ple_value.weight             [2560, 2560]           BF16           13107200 B
  blk.1.ple_conv1d.weight            [4, 10240]             F16               81920 B
  blk.47.indexer.q_proj.weight       [2560, 512]            BF16            2621440 B
[ok] layout matches inventory §2.4-§2.6/§4.2 (experts 42912972800 B, 48 blocks × 512, PLE [160, 320001536] IQ4_NL, router [2560, 512], blk.1 PLE module)
```

---

## 5. Strata pack reader (opt-in)

`Pack::open("/srv/strata-data/packs/iq3_xxs")` parses and validates:

* `index.txt` v3: header (`# strata pack index v3`, `# align A pool P tensors N` + source shard name)
  and 19-field rows, **refusing anything that is not exactly 19 fields** (mirrors
  `strata/src/core/weights.cpp:177`). Row struct: `name, file, kind, src_off, src_bytes, dst_off,
  dst_bytes, ne0, ne1, code_bits, code_bias, group_elems, codebook, has_offset, codes_bytes,
  scales_bytes, offset_bytes, scales_fp16, act_kind`. `PackKind` = 0 native, 1 BF16-in-F32 (reserved),
  2 F32, 3 F16-in-F32 (reserved), 4 BF16, 5 F16.
* Invariants: row count = header `tensors`; non-native `dst_bytes` sum = `pool` = `dense.bin` size;
  64-byte `dst_off` alignment and in-pool ranges; native rows carry no bytes; unique names.
* `native_experts.txt`: header `(n_expert 512, total 42912972800; …)` + 48 rows
  `layer gu_type d_type offset blob_bytes gate_off up_off down_off [shard]`; rows must be sequential and
  the offset chain contiguous (`offset[L+1] = offset[L] + blob_bytes[L] × n_expert`), and
  `Σ blob_bytes × n_expert = total = experts.bin size`. v4 rows with the optional 9th field parse.
* `conversions.json` / `experts.bin.src.json` → typed serde structs.

Resolution API:

* `Pack::tensor_view(name, Some(&shards)) -> PackTensor::{Dense(&[u8]) | Native(TensorView)}` — kind 0
  rows resolve through the GGUF shard set (the pack's `--native` tensors), kinds 1–5 resolve into
  `dense.bin`. Native rows without a `ShardSet` are a typed error, so the pack never silently guesses.
* `Pack::expert_blob(layer, expert) -> &[u8]` (contiguous, expert-major) and
  `Pack::expert_slices(layer, expert, n_embd, n_ff)`.
* `Pack::validate_against(&ShardSet)` cross-checks every row against the GGUF (name, `ne0`/`ne1`, raw
  float type vs kind) and every native expert layer (types, absolute `gate_off`/`up_off`/`down_off` equal
  to the GGUF tensor file offsets, blob size).

Measured on the real pack: index v3, align 64, pool 1,538,035,200, **1079 rows = 302 native + 777
dense**, 48 native expert layers, `experts.bin` 42,912,972,800 B; conversions `schema 1`, `tensors: []`;
provenance sha256 `ae28cf7a…d462c0ee`. All cross-checks pass.

## 6. MTP draft runtime reader (opt-in)

`MtpRt::open("/srv/strata-data/mtp/rt")` parses `dense.txt` (`name kind rows cols offset bytes`,
kinds `q8_0|bf16|f32`), verifies each row's byte size against the block geometry
(`q8_0` 32/34, `bf16` 1/2, `f32` 1/4), that rows tile `dense.bin` contiguously and exactly, and that
`experts.bin = n_expert × BLOB` with `BLOB = 3 × (n_embd · n_ff · 18/64) = 1,382,400` B for
`(2560, 640, 512)` (geometry is a `MtpGeometry`, default = the draft layer's). `draft_vocab.bin` is
optional; when present its size must be a multiple of 4 and tokens are read as `i32` (`draft_token(i)`,
`draft_tokens()`, `draft_vocab_len()`).

Measured: 29 dense rows (`bf16` 10, `f32` 9, `q8_0` 10) = 116,099,072 B, 512 × 1,382,400 B experts,
106,299 draft token ids, max id 248,319 < 248,320 ✓.

---

## 7. Tests

`cargo test -p breccia-format` → **44 unit tests + 1 integration test + 1 doctest, all green**
(integration test skips cleanly when `/srv/strata-data` is absent; on this machine it asserts the
inventory numbers above and that opening the shards takes < 2 s).

Synthetic fixtures are built in-test (`src/testutil.rs` writes bytes to a temp dir, removed on drop;
no binary fixtures in the repo). Coverage of the requested fuzz-ish cases:

| case | test |
|---|---|
| unsorted / duplicate metadata keys | `gguf::tests::unsorted_metadata_keys_keep_file_order_and_resolve` |
| every scalar KV type + `f64`/`f32` precision, u64/i64 extremes | `every_scalar_metadata_type_round_trips` |
| u64 arrays, string arrays, i32 arrays, empty arrays, wrong-type access | `u64_and_string_arrays_decode` |
| alignment ≠ 32 (64) | `alignment_other_than_32_is_honoured` |
| non-power-of-two alignment | `bad_alignment_is_rejected` |
| truncated data section | `truncated_data_is_a_typed_error` |
| truncated header / bad magic / GGUF v2 | `truncated_header_and_bad_magic` |
| unknown ggml type, unaligned shape, too many dims | `unknown_type_and_unaligned_shape_are_errors` |
| zero-tensor file (no data section) and zero-size tensor | `zero_tensor_file_has_no_data_section`, `zero_size_tensor_is_allowed` |
| duplicate tensor names | `duplicate_tensor_names_are_rejected` |
| split mismatch: bad count, duplicate `split.no`, wrong `split.tensors.count`, no split metadata, duplicate names, glob/family/sibling discovery | `shard::tests::*` |
| group classification + expert geometry, missing role, transposed down, histograms | `layout::tests::*` |
| pack: 18/20-field rows, unknown kind, pool mismatch, native-with-bytes, duplicate rows, offset chain, v4 shard field, role sizes, blob identity | `pack::tests::*` |
| MTP: row sizes, contiguity, expert/vocab sizes, unknown kind, missing vocab | `mtp::tests::*` |
| fork `Q2_0 = 64/18`, documented block table | `quant::tests::*` |

Verification commands and results (2026-10-26, `cargo` 1.95.0):

```
$ cargo build --workspace                                  # Finished
$ cargo test -p breccia-format
  44 passed (lib) · 1 passed (tests/real_shards.rs) · 1 passed (doc-tests)
$ cargo clippy -p breccia-format --all-targets -- -D warnings   # Finished, no warnings
$ cargo fmt --check                                        # clean
```

---

## 8. Open questions / follow-ups

1. **Metadata merging across shards.** Only shard 0's KVs are exposed. That is correct for this model
   (shard 2 carries only `split.*`), but a model that duplicates e.g. tokenizer KVs per shard would
   need a merge policy (first-wins vs. conflict check). Currently `ShardSet::parts()` gives access to
   each shard's own map.
2. **Pack kinds 1/3** (`BF16_IN_F32`, `F16_IN_F32`) are parsed and typed but have no artifact here, so
   their `src_bytes`/`dst_bytes` relationship is unverified (arena validation uses `dst_bytes` as the
   stored size, which is right for kinds 2/4/5). Treat as reserved until a pack emits them.
3. **Pack v4 per-role `shard` field.** Parsed, but `expert_blob` assumes `experts.bin` contains every
   layer. If a v4 pack splits a layer's experts across shards, the reader needs a per-shard base offset.
4. **`Q1_0` (41)** is recognised by name only; its block size was not verified against any artifact, so
   it deliberately has no geometry (a tensor using it fails with `UnsupportedTensorType`).
5. **PLE head slicing** (`ple.head_offsets`/`head_vocab_sizes`, Σ = 320,001,446 vs the table's
   `ne1 = 320,001,536`) is *not* interpreted here. The tensor shape is authoritative; mapping n-gram
   hashes to rows belongs to the M2 PLE kernel work.
6. **mmap vs. concurrent writers.** Mappings are read-only; a process with write access could still
   change the file underneath the map. Artifacts are treated as immutable inputs (read-only
   `/srv/strata-data`), which is documented at the two `Mmap::map` call sites.
7. **`Pack::validate_against` assumes the qwen4exp layout** (it derives `n_embd`/`n_ff` from the
   embedding and routed-expert tensors). It is a model-level cross-check, not a generic pack parser;
   generic use is `Pack::open` + `tensor_view`.
8. Dequantisation is intentionally absent: this crate sizes and locates quantized blocks, it does not
   decode them (that is `breccia-kernels`, M2).
