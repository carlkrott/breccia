"""F2/F3 bench frames: static interleaved rANS over real model bytes, plus a numpy reference decoder.

One file per class (bf16 / q2_0 / iq4_nl), N frames of ~16 MiB raw each, the shape `breccia bench codec`
loads into ring slots. Tables are trained on held-out data (other experts / block 0) as in F0a.
Usage: python3 -I codecframes.py <shard1.gguf> <out_dir> [lanes] [n_frames]

rANS: 12-bit probabilities, 32-bit state in [2^16, 2^32), 16-bit renorm, one stream per lane.
Lane i owns raw u32 words i, i+L, i+2L, ...; mode 0 codes the 4 bytes of each word with a
context = ctxmap[byte offset % blk]; mode 1 codes the two bf16 exponents of each word (one
table) and ships sign+mantissa as a raw byte plane.

File (little endian): u32 magic 'BRCF', version, mode, blk, n_ctx, lanes, frame_raw, n_frames;
ctxmap u8[256]; n_ctx x (lut u8[4096], fc u32[256] = freq | cum << 16);
per frame: u32 image_bytes, image (u32 offsets[lanes+1] | lane streams | plane), raw bytes.
"""
import sys, os, json, struct, math
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import codeclab as cl

PROB_BITS, RL = 12, 1 << 16
M = 1 << PROB_BITS

def build_table(counts):
    """counts[256] -> freq[256] summing to M, every symbol >= 1 (any byte must stay encodable)."""
    p = (counts + 0.5) / (counts + 0.5).sum()
    f = np.maximum(1, np.floor(p * M)).astype(np.int64)
    while f.sum() > M: f[np.argmax(f)] -= 1
    while f.sum() < M: f[np.argmax(p * M - f)] += 1
    assert f.min() >= 1 and f.max() < M and f.sum() == M
    cum = np.concatenate([[0], np.cumsum(f)[:-1]])
    lut = np.repeat(np.arange(256, dtype=np.uint8), f)
    return f, cum, lut

def lane_symbols(raw, mode, lanes, ctxmap, blk):
    """[L, nsym] symbols and contexts in decode order; plane bytes for mode 1."""
    words = len(raw) // 4; wpl = words // lanes; assert words == wpl * lanes
    if mode == 0:
        sym = raw.reshape(wpl, lanes, 4).transpose(1, 0, 2).reshape(lanes, -1)
        off = np.arange(len(raw), dtype=np.int64).reshape(wpl, lanes, 4).transpose(1, 0, 2).reshape(lanes, -1)
        return sym.astype(np.int64), ctxmap[off % blk].astype(np.int64), None
    u = raw.view('<u2')
    exp = ((u >> 7) & 0xFF).reshape(wpl, lanes, 2).transpose(1, 0, 2).reshape(lanes, -1)
    plane = (((u >> 15) << 7) | (u & 0x7F)).astype(np.uint8)
    return exp.astype(np.int64), np.zeros(exp.shape, np.int64), plane

def encode(sym, ctx, F, CUM):
    L, n = sym.shape
    x = np.full(L, RL, np.uint64); out = np.zeros((L, n), np.uint16); cnt = np.zeros(L, np.int64)
    rows = np.arange(L)
    for t in range(n - 1, -1, -1):
        f = F[ctx[:, t], sym[:, t]].astype(np.uint64); c = CUM[ctx[:, t], sym[:, t]].astype(np.uint64)
        m = x >= (f << np.uint64(32 - PROB_BITS))   # x_max = ((RL >> PROB_BITS) << 16) * f
        out[rows[m], cnt[m]] = (x[m] & np.uint64(0xFFFF)).astype(np.uint16); cnt[m] += 1; x[m] >>= np.uint64(16)
        x = ((x // f) << np.uint64(PROB_BITS)) + x % f + c
    streams = []
    for i in range(L):
        s = np.concatenate([[x[i] & 0xFFFF, x[i] >> 16], out[i, :cnt[i]][::-1]]).astype(np.uint16)
        if len(s) & 1: s = np.append(s, np.uint16(0))
        streams.append(s.view('<u4'))
    return streams

def decode(streams, ctx, F, CUM, LUT):
    L, n = ctx.shape
    width = max(len(s) for s in streams) * 2
    buf = np.zeros((L, width + 2), np.uint64)
    for i, s in enumerate(streams): buf[i, :len(s) * 2] = s.view('<u2')
    rows = np.arange(L)
    x = buf[:, 0] | (buf[:, 1] << np.uint64(16)); p = np.full(L, 2, np.int64)
    sym = np.zeros((L, n), np.int64)
    for t in range(n):
        c = ctx[:, t]; slot = (x & np.uint64(M - 1)).astype(np.int64)
        s = LUT[c, slot].astype(np.int64); sym[:, t] = s
        x = F[c, s].astype(np.uint64) * (x >> np.uint64(PROB_BITS)) + slot.astype(np.uint64) - CUM[c, s].astype(np.uint64)
        m = x < RL
        x[m] = (x[m] << np.uint64(16)) | buf[rows[m], p[m]]; p[m] += 1
    return sym

def frame_image(streams, plane):
    offs = np.zeros(len(streams) + 1, np.uint32); offs[1:] = np.cumsum([len(s) for s in streams])
    parts = [offs.tobytes(), np.concatenate(streams).tobytes()]
    if plane is not None: parts.append(plane.tobytes() + b'\0' * (-len(plane) % 4))
    return b''.join(parts)

def sources(path):
    ts, base = cl.tensors(path); mm = np.memmap(path, np.uint8, 'r')
    def tb(n):
        ty, dims, off = ts[n]
        if ty == 30: nb = int(np.prod(dims)) * 2
        else:
            bsz, bv, _ = cl.LAYOUT[ty]; nb = int(np.prod(dims)) // bv * bsz
        return ty, dims, np.asarray(mm[base + off: base + off + nb])
    out = {}
    hc = ['hc_attn_down', 'hc_attn_up', 'hc_ffn_down', 'hc_ffn_up']
    tr = np.concatenate([tb(f'blk.0.{h}.weight')[2] for h in hc])
    te = np.concatenate([tb(f'blk.{b}.{h}.weight')[2] for b in range(1, 6) for h in hc])
    out['bf16'] = dict(mode=1, blk=1, ctxmap=np.zeros(256, np.uint8), train=tr, test=te,
                       src='train blk.0 hc_*, encode blk.1-5 hc_*')
    for cls, name in (('iq4_nl', 'blk.0.ffn_down_exps.weight'), ('q2_0', 'blk.1.ffn_down_exps.weight')):
        ty, dims, b = tb(name); bsz = cl.LAYOUT[ty][0]
        per = len(b) // dims[-1]; ex = b.reshape(dims[-1], per)
        cm = np.full(256, 2, np.uint8); cm[0], cm[1] = 0, 1   # d lo byte, d hi byte, qs
        out[cls] = dict(mode=0, blk=bsz, ctxmap=cm, train=ex[0::2].reshape(-1), test=ex[1::2].reshape(-1),
                        src=f'{name}: train even experts, encode odd experts')
    return out

def main():
    path, od = sys.argv[1], sys.argv[2]
    lanes = int(sys.argv[3]) if len(sys.argv) > 3 else 8192
    n_frames = int(sys.argv[4]) if len(sys.argv) > 4 else 6
    os.makedirs(od, exist_ok=True); summary = {}
    for cls, s in sources(path).items():
        mode, blk, cm = s['mode'], s['blk'], s['ctxmap']
        unit = 4 * lanes * blk // math.gcd(4 * lanes, blk)
        frame_raw = (16 << 20) // unit * unit
        n_ctx = int(cm[:blk].max()) + 1
        # tables from the training bytes, same symbol/context split as the frames
        tr = np.ascontiguousarray(s['train'][: len(s['train']) // unit * unit])
        if mode == 0:
            b = tr.astype(np.int64); c = cm[np.arange(len(b)) % blk].astype(np.int64)
        else:
            u = tr.view('<u2'); b = ((u >> 7) & 0xFF).astype(np.int64); c = np.zeros(len(b), np.int64)
        F = np.zeros((n_ctx, 256), np.int64); CUM = np.zeros_like(F); LUT = np.zeros((n_ctx, M), np.uint8)
        for k in range(n_ctx):
            F[k], CUM[k], LUT[k] = build_table(np.bincount(b[c == k], minlength=256).astype(np.float64))
        assert len(s['test']) >= n_frames * frame_raw, (cls, len(s['test']), n_frames * frame_raw)
        hdr = struct.pack('<8I', 0x46435242, 1, mode, blk, n_ctx, lanes, frame_raw, n_frames)
        tabs = cm.tobytes() + b''.join(LUT[k].tobytes() + (F[k] | (CUM[k] << 16)).astype('<u4').tobytes() for k in range(n_ctx))
        st = dict(cls=cls, src=s['src'], mode=mode, blk=blk, n_ctx=n_ctx, lanes=lanes, frame_raw=frame_raw,
                  n_frames=n_frames, frames=[])
        with open(os.path.join(od, f'{cls}-L{lanes}.brcf'), 'wb') as fo:
            fo.write(hdr + tabs)
            for fi in range(n_frames):
                raw = np.ascontiguousarray(s['test'][fi * frame_raw:(fi + 1) * frame_raw])
                sym, ctx, plane = lane_symbols(raw, mode, lanes, cm, blk)
                streams = encode(sym, ctx, F, CUM)
                if fi == 0:   # reference decode of the first frame (full byte compare)
                    assert np.array_equal(decode(streams, ctx, F, CUM, LUT), sym), f'{cls}: numpy round trip failed'
                img = frame_image(streams, plane)
                fo.write(struct.pack('<I', len(img)) + img + raw.tobytes())
                pay = sum(len(x) for x in streams) * 4
                ideal = float(-np.log2(F[ctx, sym] / M).sum() / 8)   # bytes at the quantised table
                st['frames'].append(dict(image=len(img), ratio=round(len(img) / frame_raw, 5),
                                         offsets=4 * (lanes + 1), states=4 * lanes, streams=pay,
                                         plane=0 if plane is None else len(plane), ideal_stream=round(ideal)))
                print(cls, fi, st['frames'][-1], file=sys.stderr, flush=True)
        fr = st['frames']
        st['ratio'] = round(sum(f['image'] for f in fr) / (frame_raw * n_frames), 5)
        st['overhead_vs_ideal'] = round(sum(f['image'] - f['ideal_stream'] - f['plane'] for f in fr) / (frame_raw * n_frames), 5)
        summary[cls] = st
    json.dump(summary, open(os.path.join(od, f'frames-L{lanes}.json'), 'w'), indent=1)
    print(json.dumps({k: (v['ratio'], v['overhead_vs_ideal']) for k, v in summary.items()}))

if __name__ == '__main__':
    main()
