# Performance

RTX 2060 6 GB, driver 592.27, `--bench`, annual Boston EPW, radiation + sun hours.

| sensors | triangles (grid + boxes) | directions / sensor | rays | GAS build | trace | GRays/s | wall total |
|---|---|---|---|---|---|---|---|
| 100 k | 0.3 M | 290 + 4430 suns | 0.47 G | 7 ms | 0.24 s | 2.0 | 0.6 s |
| 1 M | 3 M | 290 + 4430 suns | 4.7 G | 44 ms | 1.7 s | 2.8 | 2.6 s |
| 1 M | 12 M | 290 + 4430 suns | 4.7 G | 260 ms | 2.5 s | 1.9 | 4.2 s |
| 1 M | 12 M | 1154 (MF2) + 4430 suns | 5.6 G | 258 ms | 3.0 s | 1.9 | 4.8 s |

Wall total includes the Perez sky model for 8760 steps, GAS build, tracing, reductions and output. Launches are
chunked to ~150 ms (Windows TDR). Expect 5-10x on Ada/Blackwell GPUs.

Mesh transport (`validate/ingest.py`, 1000 x 1000 quad grid = 1 002 001 points / 2M triangles, one time step, same
machine, seconds, best of 2; the USD rows need the `vs2022-usd` build):

| path | seconds |
|---|---|
| core `.dmesh` read (44 MB) | 0.025 |
| core `.obj` read (64 MB, `std::from_chars` + Clinger fast path; was 0.95 with `strtod`) | 0.23 |
| core `.usdc` read (12 MB, LightUSD) / `.usda` (49 MB) | 0.10 / 0.45 |
| core values out, 1M rows: `--out v.npy` / `--out v.txt` | 0.04 / 1.0 (text: formatting is 0.06, the rest is this machine's on-close file scan) |
| core wall `.dmesh` in, `.npy` out (CUDA start, GAS, trace, exit included) | 1.04 |
| Python `.usdc` -> `.dmesh`: `Stage.Open` + `meshes()` + `to_dmesh` (was ~10.8 with per-face lists and OBJ text) | 0.03 + 0.06 + 0.14 |
| Python `.usda` `Stage.Open` (pxr text parser) | 2.1 - 2.7 |
| Python values back: `np.load` / `np.loadtxt` | 0.004 / 0.13 |
| TypeScript `.glb` (36 MB, 2M triangles): `fromGLB` / `toDmesh` / write | 0.04 / 0.03 / 0.18 |
| TypeScript values back, 1M rows: `.npy` (`parseNpy`) / text (`parseValues`) | 0.22 / 1.2 |
