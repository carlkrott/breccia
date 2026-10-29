"""Endpoint matrix: theoretical codec gain per sender->receiver pair from single-card runs.

Inputs: two `bench codec` reports (MI50, RX) run with BRECCIA_CODEC_LINK + RADV_PERFTEST=transfer_queue.
L = link rate (raw GB/s): H2D/D2H on the SDMA queue; GPU->GPU is a host bounce, L = min(D2H_A, H2D_B),
UNVERIFIED under concurrency (the one concurrent attempt hung both SDMA rings, 2026-10-29 20:14).
G = receiver decode rate (raw GB/s): GPU in-ring decode, or the 32-thread CPU decoder.
gain_serial = 1/(r + L/G) (decode after copy), gain_overlap = min(1/r, G/L) (pipelined ring).
Encode cost at the sender is excluded (free for pre-encoded static weights, not for live GPU data).
Usage: python3 -I endpoints.py <report-mi50.json> <report-rx.json>
"""
import sys, json

def load(p):
    d = json.load(open(p))['devices'][0]
    return {c['file'].split('/')[-1].split('-')[0]: c for c in d['classes']}

m, x = load(sys.argv[1]), load(sys.argv[2])
c0 = next(iter(m.values())); c1 = next(iter(x.values()))
link = {('cpu', 'mi50'): c0['h2d_gbps'], ('cpu', 'rx'): c1['h2d_gbps'],
        ('mi50', 'cpu'): c0['d2h_gbps'], ('rx', 'cpu'): c1['d2h_gbps'],
        ('mi50', 'rx'): min(c0['d2h_gbps'], c1['h2d_gbps']), ('rx', 'mi50'): min(c1['d2h_gbps'], c0['h2d_gbps'])}
rows = []
for cls in m:
    r = m[cls]['ratio']
    dec = {'mi50': m[cls]['ring_decode_gbps'], 'rx': x[cls]['ring_decode_gbps'],
           'cpu': (m[cls]['cpu_decode_gbps'] + x[cls]['cpu_decode_gbps']) / 2}
    for (a, b), L in link.items():
        G = dec[b]
        s, o = 1 / (r + L / G), min(1 / r, G / L)
        rows.append((cls, f'{a}->{b}', r, L, G, s, o, L * o))
print(f"{'class':7} {'pair':9} {'ratio':>6} {'L':>6} {'G':>6} {'serial':>7} {'overlap':>7} {'raw GB/s':>8}")
for row in sorted(rows, key=lambda t: (t[0], -t[6])):
    print('{:7} {:9} {:6.3f} {:6.2f} {:6.1f} {:6.3f}x {:6.3f}x {:8.2f}'.format(*row))
