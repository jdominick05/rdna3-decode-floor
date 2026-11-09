#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Tables for REPORT.md from run.sh result folders (every CSV has a header row).

  analyze.py RESULTS_DIR             tables.md for one result folder
  analyze.py --spread OUT.md DIR...  the same quantities across several result folders of one build: median of the
                                     per-run values and their range (min-max)

E0     per-shape kernel times (rocprofv3 GPU timestamps; median of the timed launches), ours warm next to iree#23507's
       tables (which ran warm: "thousands of iterations, so L2/Infinity Cache is warm"), ours cold for the floor
floor  empty kernels and one- / two-round-trip INT4 matvecs (rocprofv3): what a small kernel's time is made of
E1     a least-squares fit of t = t0 + bytes / BW over the cold sweep
sync   the persistent kernel's per-stage cost: empty stages (barrier only) and one-tile stages, per variant and grid
E2     per matvec: launches, a HIP graph of the launches, and stages of one persistent kernel (four variants)
E3     the 1,393-matvec token per launch mode (matvecs only: compared between modes, never with whole-token tok/s)
stages where a token's time goes: grouped kernels (rocprofv3, the traced run) and persistent stages (GPU stamps)
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
WARMUP = 20   # decode_floor shapes / sweep / floor: warmup launches per row before the timed ones
PEAK = 960.0  # GB/s, the RX 7900 XTX's GDDR6
KINDS = ['q|k|v', 'o', 'router', 'gate|up x8', 'down x8', 'lm_head']
# the grouped token's kernels by grid (blocks of 256): one of each per layer, the vocab head once; the 128-block class
# holds o_proj (4.5 MB) and the per-op modes' expert down (0.8 MB), told apart by time (bimodal: ~3 and ~10 µs)
GROUPED = {320: 'q|k|v', 8: 'router', 768: 'gate|up x8', 1024: 'down x8', 9496: 'lm_head'}
O_SPLIT_US = 6.0
EXTRA_E3 = [('e3.csv', ''), ('e3-occ.csv', 'own occupancy'), ('e3-chunk1.csv', 'one-tile claims, K ≤ 768')]
EXTRA_ST = [('stages.csv', ''), ('stages-occ.csv', 'own occupancy'), ('stages-chunk1.csv', 'one-tile claims, K ≤ 768')]


def rows(res, name):
    path = os.path.join(res, name)
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def prof(dirpath, name):
    """(duration µs, blocks of 256) of the kernels whose name contains `name`, in dispatch order; [] without a trace."""
    recs = []
    for fn in glob.glob(os.path.join(dirpath, '**', '*kernel_trace.csv'), recursive=True):
        with open(fn) as f:
            for r in csv.DictReader(f):
                if name in r.get('Kernel_Name', ''):
                    recs.append((int(r['Dispatch_Id']), (int(r['End_Timestamp']) - int(r['Start_Timestamp'])) / 1e3,
                                 int(r['Grid_Size_X']) // 256))
    recs.sort()
    return [(d, b) for _, d, b in recs]


def med(v):
    v = [x for x in v if x is not None]
    return statistics.median(v) if v else None


def mean(v):
    v = [x for x in v if x is not None]
    return sum(v) / len(v) if v else None


def fnum(v, d=1):
    if v is None or v == '':
        return '-'
    try:
        return f'{float(v):.{d}f}'
    except ValueError:
        return str(v)


def flt(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# ---- per-run data ----------------------------------------------------------------------------------------------------

def kernel_rows(res, csvname, prof_fmt):
    """shape rows with the rocprofv3 median of their timed launches (rows of one fmt/cache run in dispatch order)."""
    out, groups = [], {}
    for r in rows(res, csvname):
        groups.setdefault((r['fmt'], r['cache']), []).append(r)
    for (fmt, cache), rs in groups.items():
        durs = [d for d, _ in prof(os.path.join(res, prof_fmt.format(fmt=fmt, cache=cache)), 'k_launch')]
        per = (len(durs) // len(rs)) if durs else 0
        for i, r in enumerate(rs):
            k = med(durs[i * per + WARMUP:(i + 1) * per]) if durs and per > WARMUP else None
            out.append(dict(r, kernel_us=k, N=int(r['N']), K=int(r['K']), bytes=float(r['weight_bytes'])))
    return out


def e1_fits(res):
    """fmt -> (t0 µs, BW GB/s, [(bytes, µs, N)])"""
    out = {}
    e1 = kernel_rows(res, 'e1.csv', 'prof-e1-{fmt}')
    for fmt in ('i4', 'f16'):
        pts = [(r['bytes'], r['kernel_us'] if r['kernel_us'] is not None else float(r['event_us_median']), r['N'])
               for r in e1 if r['fmt'] == fmt]
        if len(pts) >= 2:
            n = len(pts)
            mx = sum(b for b, _, _ in pts) / n
            my = sum(t for _, t, _ in pts) / n
            c = sum((b - mx) * (t - my) for b, t, _ in pts) / sum((b - mx) ** 2 for b, _, _ in pts)
            out[fmt] = (my - c * mx, 1.0 / (c * 1e3), pts)
    return out


def floor_rows(res):
    """[(kind, K, blocks, cache, µs)]: one floor invocation per prof-floor-<cache> folder; an invocation's rows are its
    empty-kernel rows followed by its matvec rows, in launch order."""
    fl, inv = rows(res, 'floor.csv'), []
    for r in fl:
        if not inv or (r['kind'] == 'empty' and inv[-1][-1]['kind'] == 'matvec'):
            inv.append([])
        inv[-1].append(r)
    out = []
    for chunk in inv:
        cache = next((r['cache'] for r in chunk if r['kind'] == 'matvec'), 'cold')
        d = os.path.join(res, f'prof-floor-{cache}')
        src = {'empty': [x for x, _ in prof(d, 'k_empty')], 'matvec': [x for x, _ in prof(d, 'k_launch')]}
        idx = {'empty': 0, 'matvec': 0}
        for r in chunk:
            reps = int(r['reps']); blk = WARMUP + reps; i = idx[r['kind']]
            part = src[r['kind']][i * blk + WARMUP:(i + 1) * blk]
            idx[r['kind']] += 1
            out.append((r['kind'], int(r['K']), int(r['blocks']), cache if r['kind'] == 'matvec' else '-',
                        med(part) if len(part) == reps else None))
    return out


def e3_rows(res):
    out = []
    for name, note in EXTRA_E3:
        for r in rows(res, name):
            if r['status'] == 'skip' or (note and not r['strategy'].startswith('persistent')):
                continue
            out.append(dict(r, note=note, grid=r['persistent_grid'] if r['strategy'].startswith('persistent') else '-'))
    return out


def e3_key(r):
    return (r['fmt'], r['cache'], r['layers'], r['strategy'], r['grid'], r['note'])


def grouped_breakdown(res):
    """the grouped token's kernels (rocprofv3, the traced run): kind -> (n, median µs, mean µs), and per-token sums."""
    ks = prof(os.path.join(res, 'prof-e3-i4'), 'k_launch')
    if not ks:
        return None
    by = {k: [] for k in KINDS}
    for d, b in ks:
        if b == 128:
            if d > O_SPLIT_US:
                by['o'].append(d)
        elif b in GROUPED:
            by[GROUPED[b]].append(d)
    if not all(by.values()):
        return None
    st = {k: (len(v), med(v), mean(v)) for k, v in by.items()}
    layer_mean = sum(st[k][2] for k in KINDS[:5]); layer_med = sum(st[k][1] for k in KINDS[:5])
    return st, 48 * layer_mean + st['lm_head'][2], 48 * layer_med + st['lm_head'][1]


def persistent_breakdown(res):
    """(note, fmt, cache, variant, grid) -> {kind: mean µs of the stage medians, 'sum': µs}"""
    out = {}
    for name, note in EXTRA_ST:
        for r in rows(res, name):
            key = (note, r['fmt'], r['cache'], r['variant'], r['grid'])
            d = out.setdefault(key, {'_v': {}, 'sum': 0.0})
            x = flt(r['us_median'])
            if x is not None:
                d['_v'].setdefault(r['kind'], []).append(x); d['sum'] += x
    for d in out.values():
        for k, v in d.pop('_v').items():
            d[k] = sum(v) / len(v)
    return out


# ---- one run: tables.md ----------------------------------------------------------------------------------------------

def tables(res):
    out = []
    p = out.append

    e0 = kernel_rows(res, 'e0.csv', 'prof-e0-{fmt}-{cache}')
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
            t = lambda r: None if not r else (r['kernel_us'] if r['kernel_us'] is not None else fnum(r['event_us_median'], 2) + ' *')
            w, c = d.get('warm'), d.get('cold')
            tc = c['kernel_us'] if c and c['kernel_us'] is not None else None
            b = (c or w)['bytes']
            ref = (IREE_F16 if fmt == 'f16' else IREE_I4).get((N, K))
            gg = GGML_F16.get((N, K)) if fmt == 'f16' else None
            p(f"| {fmt} | 1×{N}×{K} ({label}) | {b / 1e6:.2f} | {fnum(t(w))} | {fnum(ref)} | {fnum(gg)} | {fnum(t(c))} | "
              f"{fnum(b / (tc * 1e3)) if tc else '-'} | {c['copies'] + ' / ' + c['reuse_MB'] if c else '-'} |")
        p('')

    fl = floor_rows(res)
    if fl:
        p('## The floor, decomposed (rocprofv3 kernel time, µs; median of the timed launches)\n')
        p('Empty kernels: dispatch, wave launch and teardown with no loads. INT4 matvecs, 16 rows per block: K = 128 is one load '
          'round trip per lane, K = 2048 two.\n')
        p('| kind | K | blocks | cache | µs |\n|---|---|---|---|---|')
        for kind, K, blocks, cache, t in fl:
            p(f"| {kind} | {K if kind == 'matvec' else '-'} | {blocks} | {cache} | {fnum(t, 2)} |")
        p('')

    for fmt, (t0, bw, pts) in e1_fits(res).items():
        small = [t for b, t, _ in pts if b < 1e6]
        p(f'## E1 ({fmt}, cold): t = t0 + bytes / BW over {len(pts)} shapes at K=2048: t0 = {t0:.2f} µs, BW = {bw:.0f} GB/s '
          f'({100 * bw / PEAK:.0f} % of {PEAK:.0f})' + (f'; shapes under 1 MB: {min(small):.2f}-{max(small):.2f} µs' if small else '') + '\n')
        p('| N | MB | µs | GB/s |\n|---|---|---|---|')
        for b, t, N in pts:
            p(f"| {N} | {b / 1e6:.3f} | {t:.2f} | {b / (t * 1e3):.0f} |")
        p('')

    sy = rows(res, 'sync.csv')
    if sy:
        p('## Synchronization alone: per-stage cost of one persistent launch (µs; GPU stamps, median, p90)\n')
        p('| variant | tiles per stage | grid (blocks per WGP) | stage median | p90 | status |\n|---|---|---|---|---|---|')
        for r in sy:
            p(f"| {r['variant']} | {r['tiles_per_stage']} | {r['grid']} ({r['blocks_per_WGP']}) | {fnum(r['stage_us_median'], 2)} | {fnum(r['stage_us_p90'], 2)} | {r['status']} |")
        p('')

    e2 = rows(res, 'e2.csv')
    if e2:
        S = e2[0]['stages']
        p(f'## E2: per matvec, {S} matvecs as launches, as a HIP graph, and as {S} stages of one persistent kernel (µs each, event time amortized)\n')
        p('| fmt | cache | shape | MB | grid | launches | graph | barrier | barrier-pf | dataflow | dataflow-pf |\n|---|---|---|---|---|---|---|---|---|---|---|')
        for r in e2:
            p(f"| {r['fmt']} | {r['cache']} | 1×{r['N']}×{r['K']} ({r['label']}) | {float(r['weight_bytes']) / 1e6:.2f} | {r['grid']} | {fnum(r['launch_us'], 2)} | {fnum(r['graph_us'], 2)} | "
              f"{fnum(r['barrier_us'], 2)} | {fnum(r['barrier_pf_us'], 2)} | {fnum(r['dataflow_us'], 2)} | {fnum(r['dataflow_pf_us'], 2)} |")
        p('')

    e3 = e3_rows(res)
    if e3:
        p('## E3: the matvecs of a Qwen3-30B-A3B decode token, per launch mode (ms, event time, median of interleaved runs)\n')
        p('Matvecs only (attention, norms, routing and sampling are not run): compare the modes with each other, not with whole-token speeds.\n')
        base = {(r['fmt'], r['cache'], r['layers']): float(r['ms_median']) for r in e3 if r['strategy'] == 'per-op' and r['status'] == 'ok'}
        p('| fmt | cache | layers / matvecs | GB per token | mode | grid | ms (p10-p90) | effective GB/s | vs per-op |\n|---|---|---|---|---|---|---|---|---|')
        for r in e3:
            mode = r['strategy'] + (f" ({r['note']})" if r['note'] else '')
            head = f"| {r['fmt']} | {r['cache']} | {r['layers']} / {r['matvecs']} | {float(r['weight_bytes']) / 1e9:.3f} | {mode} | {r['grid']} |"
            if r['status'] != 'ok':
                p(head + ' fail | | |')
                continue
            ms = float(r['ms_median'])
            rel = base.get((r['fmt'], r['cache'], r['layers']))
            p(head + f" {ms:.3f} ({float(r['ms_p10']):.3f}-{float(r['ms_p90']):.3f}) | {float(r['effective_GBps']):.0f} | {f'{rel / ms:.2f}×' if rel else '-'} |")
        p('')

    gb = grouped_breakdown(res)
    pb = persistent_breakdown(res)
    if gb or pb:
        p('## Where the INT4 cold token goes, per stage kind (µs)\n')
        p('Grouped: rocprofv3 kernel time of the traced run (mean; median in brackets). Persistent: GPU stamps, the mean over '
          'the token\'s stages of each stage\'s median over 10 runs; a persistent stage includes its synchronization, a '
          'kernel time does not include the gap to the next launch.\n')
        p('| mode | grid | ' + ' | '.join(KINDS) + ' | sum of stages (ms) | token, event time (ms) |\n|---|---|' + '---|' * (len(KINDS) + 2))
        e3ok = {e3_key(r): float(r['ms_median']) for r in e3 if r['status'] == 'ok'}
        tok_g = next((v for k, v in e3ok.items() if k[:4] == ('i4', 'cold', '48', 'grouped-graph') and not k[5]), None)
        if gb:
            st, s_mean, _ = gb
            p('| grouped (kernels) | - | ' + ' | '.join(f'{st[k][2]:.2f} ({st[k][1]:.2f})' for k in KINDS) +
              f' | {s_mean / 1e3:.3f} | {fnum(tok_g, 3)} (graph) |')
        for (note, fmt, cache, var, grid), d in sorted(pb.items(), key=lambda x: (x[0][0], x[0][4], x[0][3])):
            if (fmt, cache) != ('i4', 'cold'):
                continue
            tok = e3ok.get(('i4', 'cold', '48', 'persistent-' + var, grid, note))
            p(f"| {var}{' (' + note + ')' if note else ''} | {grid} | " + ' | '.join(fnum(d.get(k), 2) for k in KINDS) +
              f" | {d['sum'] / 1e3:.3f} | {fnum(tok, 3)} |")
        if gb and tok_g:
            st, s_mean, s_med = gb
            p(f'\nGrouped graph: kernels {s_mean / 1e3:.3f} ms per token (sum of means; {s_med / 1e3:.3f} with medians), '
              f'token {tok_g:.3f} ms, so {(tok_g * 1e3 - s_mean):.0f} µs between kernels: {(tok_g * 1e3 - s_mean) / 241:.2f} µs per stage (241 stages).')
        p('')
    return '\n'.join(out)


# ---- several runs: spread.md -----------------------------------------------------------------------------------------

def mr(vals, d=2):
    v = [x for x in vals if x is not None]
    if not v:
        return '-'
    m = statistics.median(v)
    return f'{m:.{d}f}' if len(v) == 1 or max(v) == min(v) else f'{m:.{d}f} ({min(v):.{d}f}-{max(v):.{d}f})'


def spread(dest, runs):
    out = []
    p = out.append
    p(f'# Across {len(runs)} runs: median of the per-run values (min-max)\n')
    p('Runs: ' + ', '.join(os.path.basename(r.rstrip('/')) for r in runs) + '\n')

    keyed = {}
    for i, res in enumerate(runs):
        for r in e3_rows(res):
            if r['status'] == 'ok':
                keyed.setdefault(e3_key(r), [None] * len(runs))[i] = float(r['ms_median'])
    if keyed:
        p('## E3: the 1,393-matvec token (ms, event time; each run\'s value is its median of interleaved rounds)\n')
        p('| fmt | cache | mode | grid | ms |\n|---|---|---|---|---|')
        for (fmt, cache, layers, strat, grid, note), v in keyed.items():
            p(f"| {fmt} | {cache} | {strat}{' (' + note + ')' if note else ''} | {grid} | {mr(v, 3)} |")
        p('')

    fits = [e1_fits(r) for r in runs]
    p('## E1 fits (cold)\n\n| fmt | t0 (µs) | BW (GB/s) | under 1 MB (µs) |\n|---|---|---|---|')
    for fmt in ('i4', 'f16'):
        t0 = [f[fmt][0] if fmt in f else None for f in fits]; bw = [f[fmt][1] if fmt in f else None for f in fits]
        sm = [min((t for b, t, _ in f[fmt][2] if b < 1e6), default=None) if fmt in f else None for f in fits]
        sx = [max((t for b, t, _ in f[fmt][2] if b < 1e6), default=None) if fmt in f else None for f in fits]
        p(f'| {fmt} | {mr(t0)} | {mr(bw, 0)} | {mr(sm)} to {mr(sx)} |')
    p('')

    fk = {}
    for i, res in enumerate(runs):
        for kind, K, blocks, cache, t in floor_rows(res):
            fk.setdefault((kind, K, blocks, cache), [None] * len(runs))[i] = t
    if fk:
        p('## The floor, decomposed (rocprofv3 kernel time, µs)\n\n| kind | K | blocks | cache | µs |\n|---|---|---|---|---|')
        for (kind, K, blocks, cache), v in fk.items():
            p(f"| {kind} | {K if kind == 'matvec' else '-'} | {blocks} | {cache} | {mr(v)} |")
        p('')

    sk = {}
    for i, res in enumerate(runs):
        for r in rows(res, 'sync.csv'):
            sk.setdefault((r['variant'], r['tiles_per_stage'], r['grid'], r['blocks_per_WGP']), [None] * len(runs))[i] = flt(r['stage_us_median'])
    if sk:
        p('## Synchronization alone (µs per stage, GPU stamps)\n\n| variant | tiles | grid (per WGP) | stage |\n|---|---|---|---|')
        for (var, nt, grid, bpw), v in sk.items():
            p(f'| {var} | {nt} | {grid} ({bpw}) | {mr(v)} |')
        p('')

    ek = {}
    for i, res in enumerate(runs):
        for r in kernel_rows(res, 'e0.csv', 'prof-e0-{fmt}-{cache}'):
            ek.setdefault((r['fmt'], r['N'], r['K'], r['label']), {}).setdefault(r['cache'], [None] * len(runs))[i] = r['kernel_us']
    if ek:
        p('## E0 (rocprofv3 kernel time, µs)\n\n| fmt | shape | ours warm | IREE (warm) | llama.cpp f16 (warm) | ours cold |\n|---|---|---|---|---|---|')
        for (fmt, N, K, label), d in ek.items():
            ref = (IREE_F16 if fmt == 'f16' else IREE_I4).get((N, K)); gg = GGML_F16.get((N, K)) if fmt == 'f16' else None
            p(f"| {fmt} | 1×{N}×{K} ({label}) | {mr(d.get('warm', []), 1)} | {fnum(ref)} | {fnum(gg)} | {mr(d.get('cold', []), 1)} |")
        p('')

    cols = ['launch_us', 'graph_us', 'barrier_us', 'barrier_pf_us', 'dataflow_us', 'dataflow_pf_us']
    e2k = {}
    for i, res in enumerate(runs):
        for r in rows(res, 'e2.csv'):
            d = e2k.setdefault((r['fmt'], r['cache'], r['N'], r['K'], r['label'], r['grid']), {c: [None] * len(runs) for c in cols})
            for c in cols:
                d[c][i] = flt(r[c])
    if e2k:
        p('## E2 (µs per matvec; medians across runs, and the largest run-to-run range in the row as % of its median)\n')
        p('| fmt | cache | shape | grid | launches | graph | barrier | barrier-pf | dataflow | dataflow-pf | max range |\n|---|---|---|---|---|---|---|---|---|---|---|')
        for (fmt, cache, N, K, label, grid), d in e2k.items():
            rng = max(((max(v) - min(v)) / statistics.median(v) for v in ([x for x in d[c] if x is not None] for c in cols) if v), default=0)
            p(f'| {fmt} | {cache} | 1×{N}×{K} ({label}) | {grid} | ' + ' | '.join(fnum(med(d[c]), 2) for c in cols) + f' | {100 * rng:.0f} % |')
        p('')

    gbs = [grouped_breakdown(r) for r in runs]
    pbs = [persistent_breakdown(r) for r in runs]
    p('## Where the INT4 cold token goes, per stage kind (µs; persistent stages include their synchronization)\n')
    p('| mode | grid | ' + ' | '.join(KINDS) + ' | sum (ms) |\n|---|---|' + '---|' * (len(KINDS) + 1))
    if all(gbs):
        p('| grouped (kernel means) | - | ' + ' | '.join(mr([g[0][k][2] for g in gbs]) for k in KINDS) + f' | {mr([g[1] / 1e3 for g in gbs], 3)} |')
    keys = sorted({k for pb in pbs for k in pb if k[1:3] == ('i4', 'cold')}, key=lambda k: (k[0], k[4], k[3]))
    for key in keys:
        note, _, _, var, grid = key
        p(f"| {var}{' (' + note + ')' if note else ''} | {grid} | " + ' | '.join(mr([pb.get(key, {}).get(k) for pb in pbs]) for k in KINDS) +
          f" | {mr([pb[key]['sum'] / 1e3 if key in pb else None for pb in pbs], 3)} |")
    if all(gbs) and keyed:
        tok = next((v for k, v in keyed.items() if k[:4] == ('i4', 'cold', '48', 'grouped-graph') and not k[5]), None)
        if tok:
            gaps = [(t * 1e3 - g[1]) / 241 if t is not None else None for t, g in zip(tok, gbs)]
            p(f'\nGrouped graph: gap between kernels {mr(gaps)} µs per stage (token event time minus the kernel sum, over 241 stages).')
    p('')
    text = '\n'.join(out)
    with open(dest, 'w') as fh:
        fh.write(text + '\n')
    print(text)


def main():
    if len(sys.argv) >= 4 and sys.argv[1] == '--spread':
        spread(sys.argv[2], sys.argv[3:])
        return
    res = sys.argv[1]
    text = tables(res)
    print(text)
    with open(os.path.join(res, 'tables.md'), 'w') as fh:
        fh.write(text + '\n')


if __name__ == '__main__':
    main()
