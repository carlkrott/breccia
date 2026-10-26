# Git workflow

Standard trunk-based development with short-lived branches, Conventional Commits, and a hard publication gate. Adapted from common Rust project practice (trunk + conventional commits + PR review).

## Repository identity

- Local root: `/home/korphaus/projects/Breccia` (its own repo — **not** part of the `$HOME` repo that contains it).
- GitHub remote: **`origin` = https://github.com/carlkrott/breccia** — **PUBLIC** since 2026-10-26
  (user approved publication and the public switch; `gh` auth as `carlkrott`, SSH key also works).
- **History was rewritten on 2026-10-26** before going public: dropped the accidentally-committed
  `harnesses/vk-probe/target/` build artifacts (65 MB rlib/rmeta) and replaced the machine hostname
  with `<host>` across every commit. Repo shrank 63 MB → 916 KB; 26 commits preserved; force-pushed
  (no collaborators existed). Any pre-rewrite clone is obsolete.
- Pre-publication audit (full history, all blobs): **no secrets** (no `tskey`/`ghp`/`gho`/`sk-`/
  `AKIA`/private-key/password/Bearer patterns), and no tailnet IPs, node names, router endpoints or
  live status were ever committed.
- The GitHub **MCP** server's configured token is stale (`Bad credentials`); use `gh`/git until it is
  refreshed.

## Branching

| Branch | Rule |
|---|---|
| `main` | Always buildable. Contains only gate-passed work plus documentation commits. |
| `phase-<N>-<slug>` | Work for one phase step, e.g. `phase-0-vulkan-probe`, `phase-2-dma-rings`. |
| `experiment/<slug>` | Throwaway measurements / spike branches; may be merged as docs-only. |

- One logical change per commit.
- Rebase feature branches onto `main` before merging; no merge bubbles.
- Never commit weights, `.gguf`, or raw benchmark binaries (see `.gitignore`).

## Commits — Conventional Commits

```
feat(profile): emit hw_profile.json with BLAKE3 fingerprint
fix(vk): tolerate devices without dedicated transfer queue
bench(phase-0): add HIP vs RADV bandwidth probe
docs(specs): qwen4exp forward-pass ops inventory
chore(references): pin strata @ 61b3fb5d in manifest
```

Scope = crate or doc area. Gate outcomes use `docs(phase-N): ...`.

## Gates are records, not edits

When a phase gate is decided, commit the report under `docs/phase-<N>/`. Do not rewrite previous measurement history to fit a later outcome; append corrections instead.

## Publication gate (GitHub)

Pushing is a gated action. Preconditions, in order:

1. Remote added and credentials present.
2. No weights, secrets, or machine-specific paths with credentials in history (`git log --stat` review).
3. User explicitly approves the push (impact-aware gate — publishing is irreversible in effect).

Until then the remote stays unset; "upload to GitHub later" is satisfied by keeping history clean and conventional so `git push` is a no-brainer later.

## Before any merge to main

- [ ] `cargo test` / harness builds pass (commands in `docs/PROJECT-SETUP.md`)
- [ ] New measurements recorded under `docs/phase-*/` with exact commands
- [ ] `Reference Projects/` untouched (`git status` clean there)
- [ ] Commit messages follow Conventional Commits
