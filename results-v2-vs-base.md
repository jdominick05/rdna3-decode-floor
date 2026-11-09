runs: {'base': ['20261109-134840-xtx-base-v2', '20261109-135039-xtx-base-v2', '20261109-135424-xtx-base-v2'], '1rt': ['20261109-135231-xtx-1rt-v2', '20261109-135617-xtx-1rt-v2']}

| quantity | base | 1rt | change |
|---|---|---|---|
| floor matvec K=128 1 block cold (µs) | 2.32 | 2.44 | +5.2 % |
| floor matvec K=2048 1 block cold (µs) | 3.24 | 2.92 | -9.9 % |
| floor matvec K=2048 48 block cold (µs) | 4.68 | 4.69 | +0.2 % |
| floor matvec K=128 1 block warm (µs) | 2.12 | 2.20 | +3.8 % |
| floor matvec K=2048 1 block warm (µs) | 2.52 | 2.52 | +0.0 % |
| floor empty K=0 1 block - (µs) | 1.08 | 1.10 | +1.9 % |
| E1 i4 t0 (µs) | 4.38 | 3.95 | -9.8 % |
| E1 i4 BW (GB/s) | 720 | 825 | +14.5 % |
| E1 f16 t0 (µs) | 6.27 | 6.32 | +0.8 % |
| E1 f16 BW (GB/s) | 852 | 850 | -0.2 % |
| E0 INT4 cold 1x4096x2048 (µs) | 12.68 | 10.68 | -15.8 % |
| E0 INT4 cold 1x2048x4096 (µs) | 10.20 | 12.61 | +23.6 % |
| E0 INT4 cold 1x512x2048 (µs) | 4.20 | 4.14 | -1.4 % |
| E0 INT4 cold 1x128x2048 (µs) | 3.68 | 3.34 | -9.2 % |
| E0 INT4 cold 1x768x2048 (µs) | 4.62 | 4.65 | +0.7 % |
| E0 INT4 cold 1x2048x768 (µs) | 3.56 | 3.78 | +6.2 % |
| E0 INT4 cold 1x151936x2048 (µs) | 192.86 | 187.68 | -2.7 % |
| E3 f16 cold grouped (ms) | 8.932 | 8.946 | +0.2 % |
| E3 f16 cold grouped-graph (ms) | 8.820 | 8.846 | +0.3 % |
| E3 f16 cold per-op (ms) | 14.812 | 14.886 | +0.5 % |
| E3 f16 cold per-op-graph (ms) | 13.997 | 14.063 | +0.5 % |
| E3 i4 cold grouped (ms) | 3.805 | 3.512 | -7.7 % |
| E3 i4 cold grouped-graph (ms) | 3.683 | 3.431 | -6.8 % |
| E3 i4 cold per-op (ms) | 8.992 | 9.063 | +0.8 % |
| E3 i4 cold per-op-graph (ms) | 8.201 | 8.274 | +0.9 % |
| E3 i4 cold persistent-barrier @96 (ms) | 5.899 | 5.688 | -3.6 % |
| E3 i4 cold persistent-barrier-pf @96 (ms) | 5.734 | 5.472 | -4.6 % |
| E3 i4 cold persistent-dataflow @96 (ms) | 6.681 | 6.265 | -6.2 % |
| E3 i4 cold persistent-dataflow-pf @96 (ms) | 6.450 | 6.086 | -5.6 % |
| E3 i4 warm grouped (ms) | 2.172 | 2.151 | -0.9 % |
| E3 i4 warm grouped-graph (ms) | 2.036 | 2.016 | -1.0 % |
| E3 i4 warm per-op (ms) | 6.939 | 6.958 | +0.3 % |
| E3 i4 warm per-op-graph (ms) | 6.101 | 6.162 | +1.0 % |
| E3 i4 cold grouped (4 per WGP, link kept awake) (ms) | 3.789 | 3.498 | -7.7 % |
| E3 i4 cold grouped-graph (4 per WGP, link kept awake) (ms) | 3.675 | 3.426 | -6.8 % |
| E3 i4 cold per-op (4 per WGP, link kept awake) (ms) | 8.829 | 8.889 | +0.7 % |
| E3 i4 cold per-op-graph (4 per WGP, link kept awake) (ms) | 8.135 | 8.212 | +0.9 % |
| E3 i4 cold persistent-barrier @192 (4 per WGP, link kept awake) (ms) | 4.080 | 4.034 | -1.1 % |
| E3 i4 cold persistent-barrier-pf @192 (4 per WGP, link kept awake) (ms) | 3.911 | 3.816 | -2.4 % |
| E3 i4 cold persistent-dataflow @192 (4 per WGP, link kept awake) (ms) | 4.604 | 4.502 | -2.2 % |
| E3 i4 cold persistent-dataflow-pf @192 (4 per WGP, link kept awake) (ms) | 4.346 | 4.190 | -3.6 % |
| E3 f16 cold grouped (link kept awake) (ms) | 8.901 | 8.938 | +0.4 % |
| E3 f16 cold grouped-graph (link kept awake) (ms) | 8.794 | 8.825 | +0.4 % |
| E3 f16 cold per-op (link kept awake) (ms) | 14.647 | 14.729 | +0.6 % |
| E3 f16 cold per-op-graph (link kept awake) (ms) | 13.969 | 14.049 | +0.6 % |
| E3 i4 cold grouped (link kept awake) (ms) | 3.790 | 3.502 | -7.6 % |
| E3 i4 cold grouped-graph (link kept awake) (ms) | 3.684 | 3.426 | -7.0 % |
| E3 i4 cold per-op (link kept awake) (ms) | 8.848 | 8.913 | +0.7 % |
| E3 i4 cold per-op-graph (link kept awake) (ms) | 8.166 | 8.241 | +0.9 % |
| E3 i4 warm grouped (link kept awake) (ms) | 2.150 | 2.139 | -0.5 % |
| E3 i4 warm grouped-graph (link kept awake) (ms) | 2.032 | 2.016 | -0.8 % |
| E3 i4 warm per-op (link kept awake) (ms) | 6.786 | 6.846 | +0.9 % |
| E3 i4 warm per-op-graph (link kept awake) (ms) | 6.088 | 6.151 | +1.0 % |
| E3 i4 cold persistent-dataflow @336 (own occupancy) (ms) | 4.663 | 4.415 | -5.3 % |
| E3 i4 cold persistent-dataflow @384 (own occupancy) (ms) | 4.004 | 4.679 | +16.9 % |
| grouped kernel q|k|v (µs) | 14.93 | 12.18 | -18.4 % |
| grouped kernel o (µs) | 9.76 | 12.20 | +25.0 % |
| grouped kernel router (µs) | 3.22 | 2.95 | -8.1 % |
| grouped kernel gate|up x8 (µs) | 27.08 | 20.43 | -24.5 % |
| grouped kernel down x8 (µs) | 10.30 | 11.83 | +14.9 % |
| grouped kernel lm_head (µs) | 192.94 | 187.79 | -2.7 % |
| grouped kernel sum (ms) | 3.325 | 3.048 | -8.3 % |
| stage barrier @4/WGP q|k|v (µs) | 14.69 | 12.65 | -13.8 % |
| stage barrier @4/WGP o (µs) | 9.94 | 12.28 | +23.6 % |
| stage barrier @4/WGP gate|up x8 (µs) | 30.61 | 27.55 | -10.0 % |
| stage barrier @4/WGP down x8 (µs) | 21.37 | 23.28 | +9.0 % |
| stage barrier @4/WGP sum (ms) | 4.064 | 4.008 | -1.4 % |
| stage barrier-pf @4/WGP q|k|v (µs) | 10.81 | 10.24 | -5.3 % |
| stage barrier-pf @4/WGP o (µs) | 8.94 | 8.79 | -1.7 % |
| stage barrier-pf @4/WGP gate|up x8 (µs) | 31.34 | 28.38 | -9.4 % |
| stage barrier-pf @4/WGP down x8 (µs) | 22.84 | 24.48 | +7.2 % |
| stage barrier-pf @4/WGP sum (ms) | 3.897 | 3.787 | -2.8 % |
| stage dataflow @4/WGP q|k|v (µs) | 13.49 | 11.39 | -15.6 % |
| stage dataflow @4/WGP o (µs) | 9.81 | 12.03 | +22.6 % |
| stage dataflow @4/WGP gate|up x8 (µs) | 35.72 | 31.90 | -10.7 % |
| stage dataflow @4/WGP down x8 (µs) | 29.18 | 30.66 | +5.1 % |
| stage dataflow @4/WGP sum (ms) | 4.603 | 4.482 | -2.6 % |
| stage dataflow-pf @4/WGP q|k|v (µs) | 9.74 | 8.94 | -8.2 % |
| stage dataflow-pf @4/WGP o (µs) | 9.72 | 9.83 | +1.1 % |
| stage dataflow-pf @4/WGP gate|up x8 (µs) | 36.02 | 32.20 | -10.6 % |
| stage dataflow-pf @4/WGP down x8 (µs) | 27.82 | 29.13 | +4.7 % |
| stage dataflow-pf @4/WGP sum (ms) | 4.333 | 4.173 | -3.7 % |
