"""F0a codec lab: what the static weight classes compress to, beyond doc 37's order-0.

Held-out cross-entropy (static table trained on even experts, scored on odd ones), so
the numbers are what a pre-shared rANS table would actually deliver, not plug-in entropy.
Usage: python3 -I codeclab.py <shard1.gguf> [experts_per_tensor] > out.json
"""
import sys, json, struct, hashlib, collections
import numpy as np, zstandard

def rd(f, fmt):
    return struct.unpack('<' + fmt, f.read(struct.calcsize('<' + fmt)))

def rs(f):
    (n,) = rd(f, 'Q'); return f.read(n).decode('utf-8', 'replace')

SC = {0: 'B', 1: 'b', 2: 'H', 3: 'h', 4: 'I', 5: 'i', 6: 'f', 7: '?', 10: 'Q', 11: 'q', 12: 'd'}

def rv(f, t):
    if t == 8: return rs(f)
    if t == 9:
        et, n = rd(f, 'IQ'); return [rv(f, et) for _ in range(n)]
    return rd(f, SC[t])[0]

def tensors(path):
    f = open(path, 'rb'); assert f.read(4) == b'GGUF'
    _, nt, nkv = rd(f, 'IQQ'); kv = {}
    for _ in range(nkv):
        k = rs(f); (t,) = rd(f, 'I'); kv[k] = rv(f, t)
    out = {}
    for _ in range(nt):
        name = rs(f); (nd,) = rd(f, 'I'); dims = rd(f, 'Q' * nd); (ty,) = rd(f, 'I'); (off,) = rd(f, 'Q')
        out[name] = (ty, dims, off)
    al = kv.get('general.alignment', 32)
    return out, (f.tell() + al - 1) // al * al

# block bytes, values per block, fields as (name, byte index array within the block)
def r(a, b): return np.arange(a, b)
LAYOUT = {
    18: (98, 256, {'d': r(0, 2), 'grid': r(2, 66), 'ss': r(66, 98)}),                          # IQ3_XXS
    21: (110, 256, {'d': r(0, 2), 'qs': r(2, 66), 'qh': r(66, 74), 'signs': r(74, 106), 'scales': r(106, 110)}),  # IQ3_S
    16: (66, 256, {'d': r(0, 2), 'grid': np.concatenate([r(2 + 8 * g, 6 + 8 * g) for g in range(8)]),
                   'ss': np.concatenate([r(6 + 8 * g, 10 + 8 * g) for g in range(8)])}),       # IQ2_XXS
    17: (74, 256, {'d': r(0, 2), 'qlo': r(2, 66)[0::2], 'qhi': r(2, 66)[1::2], 'scales': r(66, 74)}),  # IQ2_XS
    22: (82, 256, {'d': r(0, 2), 'qs': r(2, 34), 'signs': r(34, 66), 'qh': r(66, 74), 'scales': r(74, 82)}),  # IQ2_S
    20: (18, 32, {'d': r(0, 2), 'qs': r(2, 18)}),                                              # IQ4_NL
    42: (18, 64, {'d': r(0, 2), 'qs': r(2, 18)}),                                              # Q2_0 (fork)
}
NIBBLE = {20}  # code IQ4_NL qs as two nibble symbols, not bytes

def H(counts):
    p = counts[counts > 0] / counts.sum(); return float(-(p * np.log2(p)).sum())

def xent0(train, test, k):
    c = np.bincount(train, minlength=k).astype(np.float64) + 0.5
    p = c / c.sum(); return float(-np.log2(p[test]).mean())

def xent1(train, test, k):
    """order-1: context = previous symbol in the same field stream (bits/symbol, held out)."""
    c = np.zeros((k, k)) + 0.5 / k * 8
    np.add.at(c, (train[:-1], train[1:]), 1)
    p = c / c.sum(1, keepdims=True)
    return float(-np.log2(p[test[:-1], test[1:]]).mean())

def xentc(train, ctx_tr, test, ctx_te, k, kc):
    """arbitrary side context (e.g. the sub-block scale nibble)."""
    c = np.zeros((kc, k)) + 0.5
    np.add.at(c, (ctx_tr, train), 1)
    p = c / c.sum(1, keepdims=True)
    return float(-np.log2(p[ctx_te, test]).mean())

def symbols(blocks, idx, nib):
    s = blocks[:, idx].reshape(-1)
    if nib: s = np.stack([s & 15, s >> 4], 1).reshape(-1)
    return s.astype(np.int64)

def zr(buf, lvl=9):
    return len(zstandard.ZstdCompressor(level=lvl).compress(buf)) / max(1, len(buf))

def expert_tensor(mm, base, name, ty, dims, n_exp):
    bsz, bvals, fields = LAYOUT[ty]
    n_all = dims[-1]; per = int(np.prod(dims[:-1])) // bvals * bsz
    pick = np.linspace(0, n_all - 1, n_exp).astype(int)
    ex = [np.asarray(mm[base + e * per: base + (e + 1) * per]).reshape(-1, bsz) for e in pick]
    tr = np.concatenate(ex[0::2]); te = np.concatenate(ex[1::2])
    res = {'type': ty, 'experts_sampled': int(n_exp), 'expert_bytes': per, 'fields': {}}
    total_bits0 = total_best = 0.0
    for fname, idx in fields.items():
        nib = ty in NIBBLE and fname == 'qs'; k = 16 if nib else 256
        a, b = symbols(tr, idx, nib), symbols(te, idx, nib)
        raw_bits = 4 if nib else 8
        h0 = xent0(a, b, k); h1 = xent1(a, b, k)
        f = {'raw_bits': raw_bits, 'x0': round(h0, 4), 'x1_prev': round(h1, 4)}
        if fname == 'grid' and ty in (18, 16):   # grid | sub-block scale nibble (top 4 bits of each ss u32)
            ss = 'ss'; sidx = fields[ss]
            def scale_ctx(blk):
                w = blk[:, sidx].reshape(len(blk), -1, 4)[:, :, 3] >> 4        # one scale per 32 values
                per_sub = len(idx) // w.shape[1]
                return np.repeat(w, per_sub, axis=1).reshape(-1).astype(np.int64)
            f['x_scale'] = round(xentc(a, scale_ctx(tr), b, scale_ctx(te), k, 16), 4)
        if fname != 'd':  # cross-expert same-position equality vs chance (sum p^2)
            e0, e1 = symbols(ex[0], idx, nib), symbols(ex[1], idx, nib)
            p = np.bincount(np.concatenate([e0, e1]), minlength=k) / (2 * len(e0))
            f['xexp_eq'] = round(float((e0 == e1).mean()), 5); f['chance'] = round(float((p * p).sum()), 5)
        else:  # d: fp16 scale; delta vs previous block, zstd byte-split
            d = tr[:, idx].copy().view('<u2').reshape(-1)
            dd = np.diff(d.astype(np.int32)) & 0xFFFF
            f['zstd_raw'] = round(zr(d.tobytes()), 4)
            f['zstd_split'] = round(zr((d & 255).astype(np.uint8).tobytes() + (d >> 8).astype(np.uint8).tobytes()), 4)
            f['zstd_delta_split'] = round(zr((dd & 255).astype(np.uint8).tobytes() + (dd >> 8).astype(np.uint8).tobytes()), 4)
        n_sym = len(idx) * (2 if nib else 1)
        best = min(v for kk, v in f.items() if kk.startswith('x') and kk != 'xexp_eq') if fname != 'd' else \
            8 * min(f['zstd_raw'], f['zstd_split'], f['zstd_delta_split'])
        total_bits0 += n_sym * (raw_bits if fname == 'd' else min(h0, raw_bits))
        total_best += n_sym * min(best, raw_bits)
        res['fields'][fname] = f
    res['ratio_order0'] = round(total_bits0 / (8 * bsz), 5)
    res['ratio_best'] = round(total_best / (8 * bsz), 5)
    # dedup: identical whole blocks and identical grid-only fields across the sampled experts
    allb = np.concatenate(ex)
    hb = collections.Counter(hashlib.blake2b(x.tobytes(), digest_size=8).digest() for x in allb)
    res['dup_blocks'] = round(1 - len(hb) / len(allb), 6)
    res['zstd19_whole'] = round(zr(ex[0].tobytes(), 19), 4)
    return res

def bf16_tensor(mm, base, ty, dims, nbytes):
    n = min(nbytes, 64 << 20) // 2
    u = np.asarray(mm[base: base + 2 * n]).view('<u2')
    exp = ((u >> 7) & 0xFF).astype(np.int64); sm = (((u >> 15) << 7) | (u & 0x7F)).astype(np.int64)
    he = H(np.bincount(exp, minlength=256)); hs = H(np.bincount(sm, minlength=256))
    hi = (u >> 8).astype(np.uint8).tobytes(); lo = (u & 255).astype(np.uint8).tobytes()
    return {'type': ty, 'H_exp': round(he, 4), 'H_sign_mant': round(hs, 4),
            'dfloat_ratio': round((8 + he) / 16, 4),             # exponent entropy-coded, sign+mantissa raw
            'zstd_raw': round(zr(u.tobytes()), 4), 'zstd_split': round(zr(hi + lo), 4)}

def raw_tensor(mm, base, ty, nbytes):
    b = np.asarray(mm[base: base + min(nbytes, 32 << 20)]).tobytes()
    return {'type': ty, 'zstd9': round(zr(b), 4), 'H_byte': round(H(np.bincount(np.frombuffer(b, np.uint8), minlength=256)) / 8, 4)}

def main():
    path = sys.argv[1]; n_exp = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    ts, base = tensors(path); mm = np.memmap(path, np.uint8, 'r')
    names = sorted(ts, key=lambda n: ts[n][2]); offs = [ts[n][2] for n in names] + [len(mm) - base]
    size = {n: offs[i + 1] - offs[i] for i, n in enumerate(names)}
    out = {'experts': {}, 'bf16': {}, 'dense': {}}
    seen = set()
    for n in names:   # one sampled tensor per (class, type)
        ty, dims, off = ts[n]; cls = n.split('.', 2)[-1] if n.startswith('blk.') else n
        if (cls, ty) in seen: continue
        seen.add((cls, ty)); a = base + off
        if '_exps' in cls and ty in LAYOUT: out['experts'][n] = expert_tensor(mm, a, n, ty, dims, n_exp)
        elif ty == 30: out['bf16'][n] = bf16_tensor(mm, a, ty, dims, size[n])
        elif size[n] >= 1 << 20: out['dense'][n] = raw_tensor(mm, a, ty, size[n])
        print(n, file=sys.stderr, flush=True)
    json.dump(out, sys.stdout, indent=1)

if __name__ == '__main__':
    main()
