#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""analyze.py RESULTS_DIR: tables for REPORT.md from one run.sh result folder (every CSV has a header row).

E0    per-shape kernel times (rocprofv3 GPU timestamps; median of the timed launches), ours warm next to iree#23507's
      tables (which ran warm: "thousands of iterations, so L2/Infinity Cache is warm"), ours cold for the floor
E1    a least-squares fit of t = t0 + bytes / BW over the cold sweep
sync  the persistent kernel's per-stage cost: empty stages (barrier only) and one-tile stages, per variant and grid
E2    per matvec: launches, a HIP graph of the launches, and stages of one persistent kernel (four variants)
E3    the 1,393-matvec token per launch mode (matvecs only: compared between modes, never with whole-token tok/s)
"""
import csv, glob, os, statistics, sys

# iree#23507, comment of 2026-04-15 (RX 7900 XTX, rocprofv3 kernel times, warm caches, µs): (N, K) -> time
IREE_F16 = {(8192, 2048): 74.6, (4096, 2048): 38.6, (2048, 4096): 34.8, (512, 2048): 5.1, (2048, 512): 6.2,
            (256, 2048): 5.4, (128, 2048): 5.3, (32, 2048): 9.0, (151936, 2048): 1413.0, (248320, 2048): 2570.0,
            (768, 2048): 5.0, (2048, 768): 8.9}
IREE_I4 = {(8192, 2048): 18.2, (4096, 2048): 10.3, (2048, 4096): 13.9, (512, 2048): 5.2, (2048, 512): 2.5,
           (256, 2048): 5.0, (128, 2048): 5.0, (32, 2048): 4.9, (151936, 2048): 231.8, (248320, 2048): 346.3,
           (768, 2048): 5.6, (2048, 768): 3.2}
GGML_F16 = {(8192, 2048): 31.5, (4096, 2048): 15.7, (2048, 4096): 12.7, (512, 2048): 3.1, (2048, 512): 5.1,
            (256, 2048): 2.3, (128, 2048): 2.1, (32, 2048): 2.0, (151936, 2048): 784.9, (248320, 2048): 1285.0,
            (768, 2048): 14.7, (2048, 768): 4.9}
WARMUP = 20   # decode_floor shapes / sweep: warmup launches per shape before the timed ones
PEAK = 960.0  # GB/s, the RX 7900 XTX's GDDR6


def rows(res, name):
    path = os.path.join(res, name)
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def trace(dirpath):
    """k_launch kernel durations (µs) in dispatch order from a rocprofv3 kernel-trace CSV, or None."""
    files = glob.glob(os.path.join(dirpath, '**', '*kernel_trace.csv'), recursive=True)
    if not files:
        return None
    recs = []
    for fn in files:
        with open(fn) as f:
            for r in csv.DictReader(f):
                if 'k_launch' in r.get('Kernel_Name', ''):
                    recs.append((int(r['Dispatch_Id']), (int(r['End_Timestamp']) - int(r['Start_Timestamp'])) / 1e3))
    recs.sort()
    return [d for _, d in recs]


def kernel_times(res, csvname, prof_fmt):
    """per shape row: the rocprofv3 median of its timed launches (rows of one fmt/cache run in dispatch order)."""
    out = []
    groups = {}
    for r in rows(res, csvname):
        groups.setdefault((r['fmt'], r['cache']), []).append(r)
    for (fmt, cache), rs in groups.items():
        durs = trace(os.path.join(res, prof_fmt.format(fmt=fmt, cache=cache)))
        per = (len(durs) // len(rs)) if durs else 0
        for i, r in enumerate(rs):
            k = None
            if durs and per > WARMUP:
                blk = durs[i * per + WARMUP:(i + 1) * per]
                k = statistics.median(blk) if blk else None
            out.append(dict(r, kernel_us=k, N=int(r['N']), K=int(r['K']), bytes=float(r['weight_bytes'])))
    return out


def fit(points):
    """least squares t = t0 + bytes * c; returns (t0 µs, BW GB/s)"""
    n = len(points)
    if n < 2:
        return None
    mx = sum(b for b, _ in points) / n
    my = sum(t for _, t in points) / n
    sxx = sum((b - mx) ** 2 for b, _ in points)
    c = sum((b - mx) * (t - my) for b, t in points) / sxx
    return my - c * mx, 1.0 / (c * 1e3)


def f(v, d=1):
    if v is None or v == '':
        return '-'
    try:
        return f'{float(v):.{d}f}'
    except ValueError:
        return str(v)


def main():
    res = sys.argv[1]
    out = []
    p = out.append

    e0 = kernel_times(res, 'e0.csv', 'prof-e0-{fmt}-{cache}')
    if e0:
        p('## E0: their shapes, one kernel per launch (µs; rocprofv3 kernel time where traced, else event time *)\n')
        p('IREE and llama.cpp columns: iree#23507, 2026-04-15, same card model, warm caches ("thousands of iterations").\n'
          'Compare them with "ours warm"; "ours cold" is the bandwidth-floor measurement.\n')
        p('| fmt | shape (1×N×K) | MB | ours warm | IREE (warm) | llama.cpp f16 (warm) | ours cold | GB/s cold | copies / reuse MB (cold) |')
        p('|---|---|---|---|---|---|---|---|---|')
        by = {}
        for r in e0:
            by.setdefault((r['fmt'], r['N'], r['K'], r['label']), {})[r['cache']] = r
        for (fmt, N, K, label), d in by.items():
            t = lambda r: None if not r else (r['kernel_us'] if r['kernel_us'] is not None else str(f(r['event_us_median'], 2)) + ' *')
            w, c = d.get('warm'), d.get('cold')
            tc = c['kernel_us'] if c and c['kernel_us'] is not None else None
            b = (c or w)['bytes']
            ref = (IREE_F16 if fmt == 'f16' else IREE_I4).get((N, K))
            gg = GGML_F16.get((N, K)) if fmt == 'f16' else None
            p(f"| {fmt} | 1×{N}×{K} ({label}) | {b / 1e6:.2f} | {f(t(w))} | {f(ref)} | {f(gg)} | {f(t(c))} | "
              f"{f(b / (tc * 1e3)) if tc else '-'} | {c['copies'] + ' / ' + c['reuse_MB'] if c else '-'} |")
        p('')

    e1 = kernel_times(res, 'e1.csv', 'prof-e1-{fmt}')
    for fmt in ('i4', 'f16'):
        pts = [(r['bytes'], r['kernel_us'] if r['kernel_us'] is not None else float(r['event_us_median'])) for r in e1 if r['fmt'] == fmt]
        if len(pts) >= 2:
            t0, bw = fit(pts)
            small = [t for b, t in pts if b < 1e6]
            p(f'## E1 ({fmt}, cold): t = t0 + bytes / BW over {len(pts)} shapes at K=2048: t0 = {t0:.2f} µs, BW = {bw:.0f} GB/s '
              f'({100 * bw / PEAK:.0f} % of {PEAK:.0f})' + (f'; shapes under 1 MB: {min(small):.2f}-{max(small):.2f} µs' if small else '') + '\n')
            p('| N | MB | µs | GB/s |\n|---|---|---|---|')
            for r in [r for r in e1 if r['fmt'] == fmt]:
                t = r['kernel_us'] if r['kernel_us'] is not None else float(r['event_us_median'])
                p(f"| {r['N']} | {r['bytes'] / 1e6:.3f} | {t:.2f} | {r['bytes'] / (t * 1e3):.0f} |")
            p('')

    sy = rows(res, 'sync.csv')
    if sy:
        p('## Synchronization alone: per-stage cost of one persistent launch (µs; GPU stamps, median, p90)\n')
        p('| variant | tiles per stage | grid (blocks per WGP) | stage median | p90 | status |\n|---|---|---|---|---|---|')
        for r in sy:
            p(f"| {r['variant']} | {r['tiles_per_stage']} | {r['grid']} ({r['blocks_per_WGP']}) | {f(r['stage_us_median'], 2)} | {f(r['stage_us_p90'], 2)} | {r['status']} |")
        p('')

    e2 = rows(res, 'e2.csv')
    if e2:
        S = e2[0]['stages']
        p(f'## E2: per matvec, {S} matvecs as launches, as a HIP graph, and as {S} stages of one persistent kernel (µs each, event time amortized)\n')
        p('| fmt | cache | shape | MB | grid | launches | graph | barrier | barrier-pf | dataflow | dataflow-pf |\n|---|---|---|---|---|---|---|---|---|---|---|')
        for r in e2:
            p(f"| {r['fmt']} | {r['cache']} | 1×{r['N']}×{r['K']} ({r['label']}) | {float(r['weight_bytes']) / 1e6:.2f} | {r['grid']} | {f(r['launch_us'], 2)} | {f(r['graph_us'], 2)} | "
              f"{f(r['barrier_us'], 2)} | {f(r['barrier_pf_us'], 2)} | {f(r['dataflow_us'], 2)} | {f(r['dataflow_pf_us'], 2)} |")
        p('')

    e3 = rows(res, 'e3.csv')
    if e3:
        groups = {}
        for r in e3:
            groups.setdefault((r['fmt'], r['cache'], r['layers'], r['persistent_grid'] if r['strategy'].startswith('persistent') else ''), []).append(r)
        p('## E3: the matvecs of a Qwen3-30B-A3B decode token, per launch mode (ms, event time, median of interleaved runs)\n')
        p('Matvecs only (attention, norms, routing and sampling are not run): compare the modes with each other, not with whole-token speeds.\n')
        base = {}
        for r in e3:
            if r['strategy'] == 'per-op' and r['status'] == 'ok':
                base[(r['fmt'], r['cache'], r['layers'])] = float(r['ms_median'])
        p('| fmt | cache | layers / matvecs | GB per token | mode | grid | ms (p10-p90) | effective GB/s | vs per-op |\n|---|---|---|---|---|---|---|---|---|')
        for r in e3:
            key = (r['fmt'], r['cache'], r['layers'])
            if r['status'] != 'ok':
                p(f"| {r['fmt']} | {r['cache']} | {r['layers']} / {r['matvecs']} | {float(r['weight_bytes']) / 1e9:.3f} | {r['strategy']} | {r['persistent_grid']} | fail | | |")
                continue
            ms = float(r['ms_median'])
            rel = f"{base[key] / ms:.2f}×" if key in base else '-'
            p(f"| {r['fmt']} | {r['cache']} | {r['layers']} / {r['matvecs']} | {float(r['weight_bytes']) / 1e9:.3f} | {r['strategy']} | "
              f"{r['persistent_grid'] if r['strategy'].startswith('persistent') else '-'} | {ms:.3f} ({float(r['ms_p10']):.3f}-{float(r['ms_p90']):.3f}) | {float(r['effective_GBps']):.0f} | {rel} |")
        p('')

    text = '\n'.join(out)
    print(text)
    with open(os.path.join(res, 'tables.md'), 'w') as fh:
        fh.write(text + '\n')


if __name__ == '__main__':
    main()
