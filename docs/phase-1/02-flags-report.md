# Phase 1 · `breccia-flags` — llama.cpp/HIP-compatible launch-flag parser

**Status:** implemented and green (2026-10-26). No gate decision depends on this
crate alone; it is the M1a input surface that lets the Phase-2/3 engine be driven
by the existing production launch line.

**Scope decided by the Phase-1 note** (`docs/plan/BRECCIA-PLAN.md` line 20):
Vulkan is the primary backend **while keeping the HIP-flavoured flag surface**, so
the exact `qwen-dual-serve.sh` invocation — including `ROCm0`/`ROCm1` — must parse
and run on Vulkan by default, with HIP as a selectable secondary backend.

## 1. What the crate does

`parse_llama_server_args(&[String]) -> Result<LaunchPlan, FlagError>` turns a
llama.cpp `llama-server` argv into a typed, `serde`-serialisable `LaunchPlan`.
Two derived operations matter downstream:

- `LaunchPlan::resolve_devices() -> Vec<ResolvedDevice>` (plus
  `resolve_devices_detailed() -> DeviceResolution` for the warnings) maps device
  *aliases* onto the *selected backend*.
- `LaunchPlan::render_args() -> Vec<String>` re-emits a canonical,
  equivalent llama.cpp argv, so plans can be logged, diffed and handed to the
  reference binary.

Design rules baked in:

| Rule | Consequence |
|---|---|
| No silent drops | an unknown flag is a hard error naming the flag and its 0-based argv position |
| Typed values | `-ctk q4_0` becomes `KvCacheType::Q4_0`; `--port 70000` fails at parse time, not at `bind()` |
| Alias ≠ backend | `ROCm1` on Vulkan resolves to Vulkan device 1 and emits a **warning**, never an error |
| Nothing invented | `Option::None` means "flag absent → llama.cpp default"; no default is guessed for a value we did not see |
| No panics | every malformed input path returns `FlagError`; `--` handling and empty values included |

### Module map

| File | Responsibility |
|---|---|
| `src/lib.rs` | crate docs, re-exports, `#![forbid(unsafe_code)]`, `#![deny(missing_docs)]` |
| `src/parse.rs` | argv walker, alias table, value coercion, `parse_str`/`tokenize` |
| `src/plan.rs` | `LaunchPlan`, backend/alias resolution, `render_args` |
| `src/device.rs` | `Backend`, `BackendHint`, `DeviceRef`, `ResolvedDevice`, `LaunchWarning` |
| `src/overrides.rs` | `-ot` top-level splitter, entry splitter, regex compilation/validation |
| `src/value.rs` | `Switch`, `ReasoningMode`, `KvCacheType` |
| `src/error.rs` | `FlagError` (position-carrying variants) |
| `tests/launch_surface.rs` | verbatim production line, round-trip, strictness, JSON |

## 2. Supported flag surface

Every flag in the production line is modelled. Field names are the JSON keys of
`LaunchPlan` (`serde` snake_case).

| Flag as launched | Accepted spellings | Plan field | Type | Production value |
|---|---|---|---|---|
| `-D <path>` | `-D`, `--binary` | `binary` | `Option<PathBuf>` | HIP `llama-server` |
| `-m <path>` | `-m`, `--model` | `model` | `Option<PathBuf>` | IQ3_XXS shard 1/2 |
| `--device` | `--device`, `--devices`, `-dev` | `devices` | `Vec<DeviceRef>` | `ROCm0,ROCm1` |
| `-ngl` | `-ngl`, `--n-gpu-layers`, `--gpu-layers` | `gpu_layers` | `Option<i32>` | `99` |
| `-ts` | `-ts`, `--tensor-split` | `tensor_split` | `Option<Vec<f32>>` | `9,15` |
| `-ot` | `-ot`, `--override-tensor` | `override_tensors` | `Vec<(String, DeviceRef)>` | 2 expert-placement rules |
| `-cmoe` | `-cmoe`, `--cpu-moe` | `cpu_moe` | `bool` | set |
| `--moe-cache-mib` | `--moe-cache-mib` | `moe_cache_mib` | `Option<u64>` | `0` |
| `-lzm` | `-lzm` | `lzm` | `Option<Switch>` | `on` |
| `-lm` | `-lm` | `lm` | `Option<String>` | `none` |
| `--no-host` | `--no-host` | `no_host` | `bool` | set |
| `-c` | `-c`, `--ctx-size` | `ctx_size` | `Option<u32>` | `200000` |
| `-kvu` | `-kvu`, `--kv-unified` | `kv_unified` | `bool` | set |
| `-np` | `-np`, `--parallel` | `parallel` | `Option<u32>` | `1` |
| `--kv-unified-per-slot` | `--kv-unified-per-slot` | `kv_unified_per_slot` | `Option<u32>` | `200000` |
| `-ctk` | `-ctk`, `--cache-type-k` | `cache_type_k` | `Option<KvCacheType>` | `q4_0` |
| `-ctv` | `-ctv`, `--cache-type-v` | `cache_type_v` | `Option<KvCacheType>` | `q4_0` |
| `-fa` | `-fa`, `--flash-attn` | `flash_attn` | `Option<Switch>` | `on` |
| `-cram` | `-cram` | `cram` | `Option<u64>` | `0` |
| `-ctxcp` | `-ctxcp` | `ctxcp` | `Option<u32>` | `1` |
| `--slot-save-path` | `--slot-save-path` | `slot_save_path` | `Option<PathBuf>` | kv-slots dir |
| `--mmproj` | `--mmproj` | `mmproj` | `Option<PathBuf>` | BF16 projector |
| `-mmdev` | `-mmdev`, `--mmdev`, `--mmproj-device` | `mmproj_device` | `Option<DeviceRef>` | `ROCm0` |
| `--image-min-tokens` | `--image-min-tokens` | `image_min_tokens` | `Option<u32>` | `1024` |
| `--image-max-tokens` | `--image-max-tokens` | `image_max_tokens` | `Option<u32>` | `2048` |
| `-lv` | `-lv`, `--log-verbosity` | `log_verbosity` | `Option<u32>` | `4` |
| `--reasoning` | `--reasoning` | `reasoning` | `Option<ReasoningMode>` | `off` |
| `-t` | `-t`, `--threads` | `threads` | `Option<u32>` | `16` |
| `-tb` | `-tb`, `--threads-batch` | `threads_batch` | `Option<u32>` | `16` |
| `--host` | `--host` | `host` | `Option<String>` | `127.0.0.1` |
| `--port` | `--port` | `port` | `Option<u16>` | `8160` |
| `--jinja` | `--jinja` | `jinja` | `bool` | set |
| `--metrics` | `--metrics` | `metrics` | `bool` | set |
| `--fit` | `--fit` | `fit` | `Option<Switch>` | `off` |
| `--no-op-offload` | `--no-op-offload` | `no_op_offload` | `bool` | set |
| `-ub` | `-ub`, `--ubatch-size` | `ubatch_size` | `Option<u32>` | `256` |
| *(none)* | `--backend`, `-be` | `backend` | `Backend` | default `vulkan` |

`--backend` is a **Breccia extension**, not llama.cpp: the production line omits
it and therefore gets `vulkan`. `-be` is a short form added for convenience.

Note on `-D`: `qwen-dual-serve.sh` itself contains no `-D` flag — the binary
path arrives as `argv[0]` (`exec $D/llama-server …`) and `parse_llama_server_args`
records a leading token that looks like a program path into `binary`
automatically. `-D`/`--binary` is the explicit on-line spelling, used to
override the path and by tests to model `argv[0]` inside a flag string.

Value forms accepted for every entry: `-c 200000` (space-separated; the value
token is taken verbatim, so `-ngl -1` works), `-c=200000` / `--port=8160`
(inline), and for booleans a bare flag plus an optional `--jinja=false`.

### Value grammars

- `DeviceRef` — `ROCmN`, `HIPN`, `CUDAN`, `VulkanN`, `GPU.N`, `GPUN`, `CPU`
  (case-insensitive). The ordinal must be a non-negative decimal that fits `u32`;
  `ROCm-1`, `ROCm1.5`, `ROCm` and a bare `1` are rejected with reasons.
- `-ts` — comma list of finite `f32`; empty elements and `inf`/`nan` rejected.
- `-ot` — see §4.
- `KvCacheType` — `f32`, `f16`, `bf16`, `q8_0`, `q5_1`, `q5_0`, `q4_1`, `q4_0`,
  `iq4_nl` (the set llama.cpp's KV path accepts; anything else fails at parse).
- `Switch` — `on|off|true|false|yes|no|1|0` (`--fit`, `-lzm`, `-fa`).
- `ReasoningMode` — `on|off|auto`.

## 3. Device aliases and backend resolution

`DeviceRef` keeps only `(backend_hint, index)` — the alias *family* and the
ordinal. Spelling is normalised at parse time (`CUDA1` → `ROCm1`, `GPU.3` →
`Vulkan3`), because resolution only depends on family + ordinal.

`Backend::Vulkan` (default) | `Hip` | `Auto`. Resolution is **ordinal-preserving
and order-preserving**: index `N` stays `N`, device order is kept, duplicates are
kept. The flag layer never re-orders, partitions or de-duplicates devices — that
belongs to `breccia-profile`'s partition plan (Gate B).

| `backend` | alias family | resolved device | warning |
|---|---|---|---|
| `vulkan` | `ROCmN` | Vulkan `N` | `hip_alias_on_vulkan` |
| `vulkan` | `CUDAN` | Vulkan `N` | `hip_alias_on_vulkan` |
| `vulkan` | `VulkanN` | Vulkan `N` | — |
| `vulkan` | `GPU.N` | Vulkan `N` | — |
| `vulkan` | `CPU` | CPU | — |
| `hip` | `ROCmN` / `CUDAN` | HIP `N` | — |
| `hip` | `VulkanN` / `GPU.N` | HIP `N` | `vulkan_alias_on_hip` |
| `hip` | `CPU` | CPU | — |
| `auto` | `ROCmN` / `CUDAN` | HIP `N` | — |
| `auto` | `VulkanN` / `GPU.N` | Vulkan `N` | — |
| `auto` | `CPU` | CPU | — |

Warnings are **data, not log strings**: `LaunchWarning { code, requested,
resolved_backend, resolved_index, message }` serialises to JSON with a stable
`code` (`hip_alias_on_vulkan`, `vulkan_alias_on_hip`), so a warning can be
asserted in a test or counted by a supervisor without parsing prose. The
production line therefore yields exactly two warnings (one per `ROCmN`), and the
plan still launches.

Semantics, stated plainly: **the alias names the ordinal, the backend policy names
the driver.** `ROCm0,ROCm1` + `--backend vulkan` means "Vulkan devices 0 and 1".
If the operator actually wants ROCm, `--backend hip` makes the same line silent —
no flag rewrite required.

## 4. `-ot` parsing

One argv element holds a comma-separated list of `regex=device` entries:

```text
blk\.(0|1|2|4|8|18|20|22|23)\.ffn_(up|down|gate)_exps\.=ROCm0,blk\.(24|35)\.ffn_(up|down|gate)_exps\.=ROCm1
```

Splitting on raw commas is wrong in general — a regex can legitimately contain a
comma in a group (`(1,2)`), a character class (`[a,b]`) or a counted repetition
(`\d{1,3}`). `split_top_level(input, ',')` splits only when:

- the comma is at nesting depth 0, where depth counts `(`/`)`, `[`/`]`, `{`/`}`;
- the comma is not backslash-escaped (`,`) — escapes are preserved verbatim, not
  unescaped;
- an unmatched closing bracket cannot drive depth negative (lenient, no panic).

`split_override_entry` then splits each entry at the **last top-level `=`**, so a
pattern containing `=` still parses. Both empty sides, a missing `=`, a stray
comma (empty entry) and a bad device alias produce an error indexed into the
*accumulated* override list across repeated `-ot` flags.

`validate_overrides` / `LaunchPlan::validated_overrides()` compile all patterns
with the `regex` crate and return **all** failures as `Vec<FlagError>` with
`entry #N` indices; on success it yields `CompiledOverride { pattern, regex,
device }` with an unanchored `matches(tensor_name)` (llama.cpp semantics). This is
the loader's startup check, and it is the only place regex compilation happens —
parsing never pays for it.

## 5. Rendering (round-trip)

`render_args()` emits a canonical argv: fixed flag order (binary, model, backend,
devices, placement, KV, multimodal, logging, threads, server), `--long` spellings
where llama.cpp has one, verified short spellings for fork-only flags (`-cmoe`,
`-kvu`, `-lzm`, `-lm`, `-cram`, `-ctxcp`, `-mmdev`), values always space-separated
(so a leading `-` stays unambiguous), absent flags omitted, and `--backend`
omitted when it is the default `vulkan`.

Guarantee asserted by test: `parse(render(plan)) == plan` **and**
`render(parse(render(plan))) == render(plan)` — rendering is idempotent, so a plan
logged to JSON and rendered later produces a byte-identical argv.

## 6. Strictness and error surface

| Condition | Error | Example message |
|---|---|---|
| unknown flag | `UnknownFlag` | ``unknown flag `--frobnicate` at argument position 2 (0-based)`` |
| bare token | `UnexpectedPositional` | ``unexpected positional argument `stray` at argument position 2 (0-based)`` |
| missing value | `MissingValue` | ``flag `-c` at argument position 0 expects a value`` |
| bad value | `InvalidValue` | ``invalid value `70000` for flag `--port` ...`` + reason |
| bad device | `InvalidDevice` | ``invalid device reference `ROCm-1` ...: negative index ...`` |
| duplicate flag | `DuplicateFlag` | ``flag `-c` at argument position 2 duplicates the one at position 0`` |
| bad `-ot` entry | `OverrideEntry` | ``-ot override entry #1 `broken` is invalid: expected `regex=device` `` |
| bad `-ot` regex | `OverrideRegex` | ``-ot override entry #1 ... has an invalid regex: ...`` |
| bad launch line | `Tokenize` | ``could not tokenize launch line at token 0: unterminated ' quote`` |

`position`/`index` is the 0-based index into the argv slice, so it is directly
`argv[position]` in a log line. `--device`, `-ot` may repeat (values append, like
llama.cpp); every other flag may appear once.

`parse_str` is the convenience entry point (tests, config files, CLI dry runs):
whitespace-separated, `'...'`/`"..."` grouping with quotes stripped, backslash
literal **except** before a newline (line continuation) so regex escapes survive
unquoted, no other escapes. A real launch should pass an argv slice.

## 7. Test evidence

Run from the repo root on 2026-10-26 (rustc 1.95.0, cargo 1.95.0, no feature
flags, no environment overrides — this crate touches no GPU/backend state):

```bash
cargo build -p breccia-flags      # Finished dev profile, 0 warnings
cargo test  -p breccia-flags      # 41 unit + 10 integration + 1 doctest = 52 passed, 0 failed
cargo clippy -p breccia-flags --all-targets   # clean (0 warnings)
cargo fmt -p breccia-flags --check            # clean
cargo build --workspace           # Finished dev profile (unchanged by this crate)
```

```
running 41 tests ... test result: ok. 41 passed; 0 failed
running 10 tests ... test result: ok. 10 passed; 0 failed   (tests/launch_surface.rs)
running  1 test  ... test result: ok.  1 passed; 0 failed   (doctest, lib.rs)
```

| Area | Tests |
|---|---|
| Verbatim production line | `parses_the_verbatim_production_line` asserts all 36 flags field-by-field; `argv_and_string_forms_agree` (66-token argv), `every_override_pattern_compiles` (both expert-placement regexes compile and match `blk.22`/`blk.35` but not each other's blocks) |
| Device alias matrix | `backend_alias_resolution_matrix` — 18 cells (6 aliases × 3 backends), each asserting resolved backend, index, `is_remapped()` and warning presence/code; plus `parses_all_alias_families`, `rejects_malformed_aliases` (9 bad spellings), `display_is_canonical_and_round_trips` |
| Backend policy | `rocm_aliases_resolve_to_vulkan_with_warnings` (production line ⇒ 2 Vulkan devices + 2 warnings, `resolve_devices` agrees with the detailed API), `switching_to_hip_keeps_the_same_devices`, `resolution_preserves_order_and_duplicates`, `empty_device_list_resolves_to_nothing`, `warnings_have_machine_readable_codes` |
| `-ot` splitter | `does_not_split_inside_parens`, `does_not_split_inside_pipes_groups_or_repetition` (nested groups, `[a,b]`, `\d{1,3}`, escaped comma, top-level pipes), `unmatched_closing_bracket_cannot_underflow`, `splits_entry_at_last_top_level_equals`, `rejects_bad_entries`, `indices_are_accumulated_and_devices_typed`, `validation_reports_all_bad_regexes_with_indices` |
| Round-trip | `round_trip_parse_render_parse_is_a_fixed_point`, `render_is_canonical_for_every_field` (full expected argv), `render_omits_absent_flags_and_defaults`, `render_emits_non_default_backend_only`, `inline_equals_form_is_equivalent` (whole line rewritten as `--flag=value`) |
| Strictness | `unknown_flag_error_names_flag_and_position` (exact text, plus inserted into the production line and position-checked), `missing_value_error_names_flag`, `duplicate_flag_is_rejected_with_both_positions`, `positional_argument_is_rejected`, `bad_values_name_flag_and_reason`, `bad_device_names_flag_position`, `override_errors_are_indexed`, `malformed_lines_never_panic` (every prefix of the production line + 15 hostile strings) |
| Serde | `plan_serialises_and_deserialises`, `plan_json_round_trips_with_overrides_and_warnings` (partial JSON falls back to defaults) |

## 8. Gaps and open questions

1. **Only the modelled surface parses.** Any llama.cpp flag not in §2 is an
   error, by design. If the service line gains `--chat-template-file`, `-b`,
   `--rope-*`, grammar/`--lora` flags or similar, this crate must grow a field
   per flag before Breccia can accept the line. That is deliberate (a dropped
   flag silently changes behaviour) but it makes flag-surface drift a maintenance
   item for whoever edits `qwen-dual-serve.sh`.
2. **Fork-only flag semantics are not encoded**, only their values: `-lzm`,
   `-lm`, `-cram`, `-ctxcp`, `--fit` have no upstream documentation in this repo.
   `-lm` is stored as a pass-through string precisely because its value set is
   unknown; `-cram`/`-ctxcp` are typed as integers because the production line
   passes integers, not because the meaning is known. **Open question for the
   engine owner:** what do these four actually control, and must Breccia emulate
   them or reject them?
3. **No cross-flag validation.** `-np 1` with `--kv-unified-per-slot 200000` and
   `-c 200000` are not checked for consistency, `len(-ts) != len(--device)` is not
   caught, and `-ot` devices are not checked against `--device` membership. These
   need policy decisions (and probably belong in `breccia-profile`'s plan
   validator), so they are not guessed here.
4. **Alias spelling is canonicalised**, so the plan cannot tell you whether the
   operator wrote `ROCm0` or `CUDA0`. This is what keeps `DeviceRef` a two-field
   value, but it means a warning message shows the canonical family
   (`hip_alias_on_vulkan` covers both HIP and CUDA spellings).
5. **Ordinal-preserving resolution is not a placement plan.** Resolving `ROCm0` →
   Vulkan 0 assumes llama.cpp's Vulkan enumeration order equals the HIP one on
   this box. Gate A's hazard list warns that Vulkan device visibility is
   environment-sensitive (`MESA_VK_DEVICE_SELECT`), so the profile layer must
   verify the mapping against real device properties before trusting it. Worth an
   explicit check at Gate B.
6. **`-ot` patterns with a literal top-level comma cannot round-trip**: the
   splitter would re-split them on re-parse. No known pattern in production does
   this (commas appear only inside groups), and no quoting/esaping scheme exists
   upstream to express it. Documented rather than invented.
7. **`parse_str` is not a shell.** No `$VAR`, backticks, `~` expansion, or
   backslash escapes other than line continuation. The service line's quotes are
   handled; anything more should pass a real argv.
8. ~~**Foreign files in `src/`.`**~~ **Resolved 2026-10-26.** `src/ot.rs` and
   `src/tests.rs` appeared at 09:27–09:29 during this implementation — remnants
   of a superseded parallel attempt at the same crate (`split_ot_list`,
   `GpuLayers`, `ValidateError` API), never declared in `lib.rs` and therefore
   never compiled. Their owner removed them after this report was written;
   `cargo build`/`cargo test`/`cargo clippy`/`cargo fmt` were re-run afterwards
   and remain clean (41 + 10 + 1 tests, no warnings), confirming they were not
   load-bearing.
9. No benchmark runs and no service restarts were performed; this crate is pure
   parsing and the verification above is the complete evidence.
