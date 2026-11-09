import re, sys, glob, os, csv
run = sys.argv[1]
arms = []
for line in open(f'{run}/arms.log'):
    m = re.match(r'\S+ (r\d-\S+) window rc=(\d+)', line)
    if m: arms.append(m.group(1))
state = {}; cur = None
for line in open(f'{run}/arms.log'):
    if 'state:' in line: cur = line.strip()
    m = re.match(r'\S+ (r\d-\S+) window rc', line)
    if m: state[m.group(1)] = cur
pat = re.compile(r'# window (\S+)\s+events minus stamps per run: p10 ([\d.]+), median ([\d.]+), p90 ([\d.]+), max ([\d.]+) us; (\d+) of (\d+) runs over 50 us; stamp to stamp p10 ([\d.]+) p90 ([\d.]+)')
sts = re.compile(r'# window (\S+)\s+event\s+([\d.]+) us \(host waits\)\s+([\d.]+) us \(host spins\); stamp to stamp\s+([\d.]+) us')
print('| arm | l1 01/03, policy, level, pcie | barrier-pf hidden med (over 50) | dataflow-pf | barrier | dataflow | grouped-graph | empty | b-pf stamp-to-stamp | E0 35B head ev | 30B head ev |')
print('|---|---|---|---|---|---|---|---|---|---|---|')
for a in arms:
    t = open(f'{run}/{a}-window.log').read()
    h = {m.group(1): (float(m.group(3)), int(m.group(6)), int(m.group(7))) for m in pat.finditer(t)}
    s2s = {m.group(1): float(m.group(4)) for m in sts.finditer(t)}
    e0 = {}
    f = f'{run}/{a}-e0.csv'
    if os.path.exists(f):
        for r in csv.reader(open(f)):
            if len(r) > 10 and r[5] in ('35B lm_head', '30B lm_head'): e0[r[5]] = float(r[9])
    st = state.get(a) or ''
    m = re.search(r"l1_aspm 01=(\d) 03=(\d) policy='([^']*)' level=(\S+)", st)
    pol = re.search(r'\[(\w+)\]', m.group(3)).group(1) if m else '?'
    stc = f"{m.group(1)}/{m.group(2)}, {pol}, {m.group(4)}" if m else '?'
    c = lambda k: f"{h[k][0]:.1f} ({h[k][1]}/{h[k][2]})" if k in h else '-'
    print(f"| {a} | {stc} | {c('persistent-barrier-pf')} | {c('persistent-dataflow-pf')} | {c('persistent-barrier')} | {c('persistent-dataflow')} | {c('grouped-graph')} | {c('empty-kernel') if 'empty-kernel' in h else c('empty')} | {s2s.get('persistent-barrier-pf', 0):.1f} | {e0.get('35B lm_head', 0):.1f} | {e0.get('30B lm_head', 0):.1f} |")
