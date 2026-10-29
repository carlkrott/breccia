"""F0b codec lab: what the dynamic classes (GDN state, QSA K/V) compress to.

Input: a `BRECCIA_DUMP_STATE` directory (p<pos>-b<block>-<part>.bin, raw device bytes).
Dynamic data ships its own table per frame, so a ratio here is plug-in entropy of the
frame plus 384 B per coded plane for the table (256 x 12-bit freqs). One frame = one file.
Usage: python3 -I statelab.py <dump_dir> > out.json
"""
import sys, os, re, json, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from codeclab import H, zr

TAB = 384   # bytes per shipped adaptive table
PAGE_CELLS, PAGE_BYTES = 16, 16384   # qsa::PAGE_CELLS, sizes::f16_pool_bytes(1)
ELEM = {'gdn_state': 4, 'gdn_conv': 4, 'qsa_k': 2, 'qsa_v': 2}

def coded(b):
    """bytes after an adaptive order-0 rANS over b (u8 array)."""
    return H(np.bincount(b, minlength=256)) * len(b) / 8 + TAB

def planes(b, e):
    """byte planes (byte k of every e-byte element), one table each."""
    return sum(coded(b[k::e]) for k in range(e))

def exp_split(b, e):
    """DFloat-style: exponent coded, sign+mantissa raw (f32: 8 of 32 bits, f16: 5 of 16)."""
    if e == 4:
        u = b.view('<u4'); ex = (u >> 23) & 0xFF; raw_bits = 24
    else:
        u = b.view('<u2'); ex = (u >> 10) & 0x1F; raw_bits = 11
    return H(np.bincount(ex.astype(np.int64), minlength=256)) * len(u) / 8 + TAB + raw_bits * len(u) / 8

def main():
    d = sys.argv[1]
    files = collections.defaultdict(dict)   # (block, part) -> {pos: path}
    for f in os.listdir(d):
        m = re.match(r'p(\d+)-b(\d+)-(\w+)\.bin$', f)
        if m and m[3] in ELEM:
            files[(int(m[2]), m[3])][int(m[1])] = os.path.join(d, f)
    acc = collections.defaultdict(lambda: collections.defaultdict(float))
    for (blk, part), snaps in sorted(files.items()):
        e = ELEM[part]; prev = prev_pos = None
        for pos in sorted(snaps):
            b = np.fromfile(snaps[pos], np.uint8)
            a = acc[part]; n = len(b)
            a['raw'] += n; a['frames'] += 1
            a['order0'] += coded(b)
            a['planes'] += planes(b, e)
            a['exp_split'] += exp_split(b, e)
            a['zstd3'] += zr(b.tobytes(), 3) * n
            if prev is not None:
                if part.startswith('qsa'):
                    # K/V cells are append-only: the shipped prefix is skipped by content ID,
                    # only the new pages move (planes-coded)
                    k = (prev_pos + 1) // PAGE_CELLS * PAGE_BYTES   # full pages only
                    a['d_raw'] += n
                    a['d_prefix_equal'] += float(np.array_equal(prev[:k], b[:k])) * k
                    a['d_coded'] += planes(b[k:], e) if n > k else 0
                else:
                    x = prev ^ b; a['d_raw'] += n
                    a['d_zero'] += float((x == 0).sum())
                    a['d_coded'] += planes(x, e)
            prev, prev_pos = b, pos
    out = {}
    for part, a in acc.items():
        r = {k: round(a[k] / a['raw'], 4) for k in ('order0', 'planes', 'exp_split', 'zstd3')}
        r.update(raw_mib=round(a['raw'] / 2**20, 1), frames=int(a['frames']))
        if a['d_raw']:
            r['delta_vs_prev'] = round(a['d_coded'] / a['d_raw'], 4)
            if part.startswith('qsa'):
                r['delta_prefix_equal'] = round(a['d_prefix_equal'] / a['d_raw'], 4)
            else:
                r['delta_zero_bytes'] = round(a['d_zero'] / a['d_raw'], 4)
        out[part] = r
    json.dump(out, sys.stdout, indent=1); print()

if __name__ == '__main__':
    main()
