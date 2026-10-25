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

Transmission (`--transmission on`): the GAS gets per-triangle SBT offsets (one hit record per optics slot) only when a
slot transmits; default runs keep the single-record layout (when it was chosen, offsets on every build cost 18 % GAS
memory and ~6 % GRays/s). 1 M sensors / 12 M triangles with one of the boxes' 16 slots porous (`--optics
bench_boxes/1=0.2,0.5,porous`): GAS 616 MB instead of 506, trace 2.9 s instead of 2.2 (1.65 GRays/s; transmittance
bytes, 18 chunks of 256 MiB), wall 5.2 s instead of 3.9. Rays go on through every translucent crossing: with every box
porous (100 k sensors, 0.3 M triangles) the trace runs at 0.5 GRays/s instead of 3.5. Byte-mode launches start at
2^13 threads and aim at 30 ms, leaving room below the TDR when the crossings vary from sensor to sensor.

Reflected light (`--bounces N`, `--bounce-samples 1024`): a second pass per chunk, one thread per sensor tracing its
paths from closest hit to closest hit. 1 M sensors, annual clear sky (Boston, `--clear-sky --lat 42.35 --lon -71.07`),
the GPU shared with another session (15-30 % load, 79 C: the visibility pass ran ~10 % below the table above):

| scene | sky density | bounces | reflected-light pass | paths/s | wall total |
|---|---|---|---|---|---|
| grid + 10 k boxes (`--bench 1000000 120000`) | 1 | 1 | 2.1 s | 494 M | 5.6 s |
| | 1 | 2 | 2.2 s | 455 M | 5.5 s |
| | 4 | 1 | 2.1 s | 485 M | 7.9 s |
| | 4 | 2 | 2.3 s | 453 M | 7.7 s |
| the same, `--transmission on --optics bench_boxes/1=0.2,0.5,porous` | 1 | 1 | 2.9 s | 357 M | 7.2 s |
| `--bench` (12 M triangles) | 1 | 1 | 69 s | 15 M | 76 s |
| | 1 | 2 | 103 s | 10 M | 111 s |
| | 4 | 1 | 175 s | 6 M | 194 s |
| | 4 | 2 | 203 s | 5 M | 220 s |

Closest hits have no first-hit exit, so the pass depends on the geometry far more than the visibility does: `--bench`
puts 833 k random boxes on 1 km^2, overlapping ~250 deep, the worst case for closest hits (30x slower per path than
the city-like 10 k boxes; the visibility pass does not notice). A second bounce adds 5-10 % in the city-like scene and
16-49 % on `--bench` (one thread runs a sensor's paths, so the ~4 % that survive the Russian roulette at the second
reflection keep most warps busy for another segment among the dense boxes); interiors, where paths stay long, pay more.
Time is linear in M. With bounces, slots that differ in rho or glass give the GAS a hit record per slot (as a
transmitting slot does: ~20 % more GAS memory, a few % slower visibility) under a resident-scene key of its own, so a
server that switches bounces on and off for such a model rebuilds it each time or parks both. At sky density 4 the
rows (2306 bins, 9 kB per sensor) cut chunks to 29 k sensors (35 chunks instead of 3), which costs the city-like scene
nothing but slowed both passes of `--bench` (launch tails, there and in the visibility's smaller launches). Without
`--bounces` the pass does not run and the visibility keeps its kernel (byte mode has its own raygen, the default bits
one is the original): `--bench` best of 3 against the build before reflected light, interleaved, 1.37 vs 1.36 GRays/s
(both throttled at 88 C).

Noise (canyon fixture, 400 facade sensors with a ground mesh, one bounce, against M = 65536): at M 1024 the annual
radiation is within 0.5 % (median sensor) and 2.3 % (95th percentile) of the reference, the reflected part itself
(~11 % of the total) within 4.5 % / 13 %; a single instant (06-21 12:00) within 2.3 % / 14 %, its reflected part
17 % / 48 %. Sums over the sensors are unbiased (< 0.07 %); everything scales as 1 / sqrt(M).

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
