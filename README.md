# daylight

GPU annual daylight / solar simulation for meshes. Generic face-vertex meshes in, one value list per element out (per
face or per vertex, input order). Ray casting on NVIDIA RT cores via OptiX 9; sky model is a port of Radiance
`gendaymtx` (Perez all-weather); results validated against Ladybug Tools, Radiance and pvlib.

The core is a format-agnostic CLI (and a C++ library) that takes raw vertex / face-vertex-index arrays: `.dmesh`
files (one `fread`, no library semantics) are the native input, `.obj` is text interop, `.usd*` is interop when the
CLI is built with `-DDAYLIGHT_USD=ON`. USD, glTF and three.js scenes otherwise go through the bindings
(`bindings/python`, `bindings/cpp`, `bindings/ts`), which write `.dmesh`, call the C++ API in-process, and put the
results back onto the meshes (USD primvars, three.js vertex colours).

```
daylight roof.obj --context city.obj --epw site.epw --out rad.txt                      # annual kWh/m2 per face
daylight sensors.dmesh --context city.dmesh --context trees.obj --epw site.epw --quantity radiation,sunhours --out v.npy
daylight facade.obj --lat 52.52 --lon 13.41 --quantity sunhours --from 06-21T00:00 --to 06-22T00:00 --step 10
daylight roof.obj --clear-sky --lat 52.52 --lon 13.41 --out clear.txt                  # no weather file: clear-sky year
daylight scene.usdc --context city.usdc --epw site.epw --on vertices --out v.txt       # vs2022-usd build
daylight roof.obj --context city.obj --epw site.epw --mode ladybug --no-self-occlude --out lb.txt
daylight roof.obj --epw site.epw --quantity illuminance --sky-density 2 --patch-samples 4 --timeseries roof.npy --out roof.txt
daylight --bench 1000000 10000000 --epw site.epw
```

## Usage

**CLI.** Positional files are analysed (sensor meshes), `--context FILE` files only block. Any mix of `.dmesh`, `.obj`
and (USD build) `.usd*`. `--out v.txt` writes one text line per element, `--out v.npy` a float64 `[N x C]` NumPy
array; both get a `.json` manifest next to them (columns, mesh ranges). Every flag: see [Flags](#flags).

```
daylight sensors.dmesh --context city.obj --epw site.epw --quantity radiation,sunhours --sides both --out v.npy
daylight --help
```

**C++, in-process (no files).** Fill `daylight::Mesh` (points xyz in metres, Z-up; vertices per face; 0-based
indices, right-handed winding) and call `daylight::run`. CMake: `add_subdirectory(<repo>)` +
`target_link_libraries(app PRIVATE daylightcore)`.

```cpp
#include "daylight.h"
#include <stdexcept>
daylight::Mesh m;                                  // one quad; m.sensor = false would make it context (blocks only)
m.name = "roof"; m.points = {0,0,3, 10,0,3, 10,8,3, 0,8,3}; m.counts = {4}; m.indices = {0,1,2,3};
daylight::Options opt; opt.quantities = {daylight::Quantity::Radiation, daylight::Quantity::SunHours};
daylight::Weather w; daylight::Result r; std::string err;
if (!daylight::read_epw("site.epw", w, err) || !daylight::run({m}, &w, opt, r, err)) throw std::runtime_error(err);
// r.values: N x r.columns.size() row-major (N = elements of all sensor meshes, input order; r.ranges = per mesh)
double roofRadiation = r.values[0], roofSunHours = r.values[1];
// no weather file: daylight::clear_sky_weather(52.52, 13.41, 1, 3.0, 0.0, w, err) instead of read_epw (lat, lon, tz, TL, altitude)
```

**C++ with USD.** `bindings/cpp` adds `read_usd` (LightUSD: sublayers, references, payloads, xforms, up axis,
units); configure with `-DDAYLIGHT_USD=ON` (preset `vs2022-usd`), link `daylight_usd`.

```cpp
#include "daylight_usd.h"
std::vector<daylight::Mesh> meshes; std::string err;
if (!daylight::read_usd("scene.usdc", 'A', -1, meshes, err)) throw std::runtime_error(err);  // 'A' = stage up axis, -1 = authored metersPerUnit
for (auto& m : meshes) m.sensor = m.name.rfind("/World/Bldg", 0) == 0;   // prim path prefix picks the sensors
daylight::Weather w; daylight::Result r;
if (!daylight::read_epw("site.epw", w, err) || !daylight::run(meshes, &w, daylight::Options{}, r, err)) throw std::runtime_error(err);
```

**Python.** `run_meshes` takes `(points, counts, indices)` triples (numpy or lists), writes `.dmesh` files, runs the
CLI and loads its `.npy` output; `meshes` / `split` turn a composed USD stage (`usd-core`) into the same flat arrays;
`write_results` puts the values back on the stage as primvars (see [Results on meshes](#results-on-meshes)).

```python
import numpy as np, daylight
P = np.array([[0, 0, 3], [10, 0, 3], [10, 8, 3], [0, 8, 3]], float)       # world metres, Z-up
values, manifest = daylight.run_meshes([(P, [4], [0, 1, 2, 3])], context=[(cityP, cityCounts, cityIdx)],
                                       epw='site.epw', args=['--quantity', 'radiation,sunhours'])
print(manifest['columns'], values)                                         # values: [N x C] float64, N = faces

sens, ctx = daylight.split(daylight.meshes('scene.usd'), sensors=['/World/Bldg'])   # [(prim, points, counts, indices)]
values, manifest = daylight.run_meshes([m[1:] for m in sens], [m[1:] for m in ctx], epw='site.epw')
daylight.write_results('scene.usd', values, manifest, [m[0] for m in sens], 'daylight.usda')   # primvars + displayColor
```

```
python -m daylight scene.usd --sensors /World/Bldg -o out --write-usd out/daylight.usda -- --epw site.epw   # USD in, USD out
```

**TypeScript.** `fromThree` / `fromGLB` produce meshes in the core frame (Y-up to Z-up, mirrored transforms
unflipped); `run` (Node) writes `.dmesh` files, runs the CLI and reads its `.npy`; `applyToThree` colours the mesh.

```ts
import { fromThree, fromGLB, run, applyToThree, toDmesh } from 'daylight-mesh';
import fs from 'node:fs/promises';
const sensors = fromThree([scene]);                                  // every descendant three.js Mesh
const context = await fromGLB(await fs.readFile('city.glb'));
const { values, columns, manifest } = await run('build/RelWithDebInfo/daylight.exe', { sensors, context, args: ['--epw', 'site.epw'] });
console.log(columns, values[0]);                                     // one Float64Array per sensor triangle
await applyToThree(roofMesh, values.slice(0, roofTriangles).map((v) => v[0]));   // vertex colours (viridis) on the mesh
const bytes = toDmesh(sensors[0]);                                   // browser: POST to a server that runs the CLI
```

**Build.** Core: VS 2022 (MSVC 14.44), CUDA 12.6, OptiX SDK 9.0 headers (`OPTIX_INSTALL_DIR`; runtime in the driver
R570+), CMake 3.27+, git (FetchContent pulls earcut.hpp, and LightUSD for the USD build). An RTX GPU is needed to run.

```
cmake --preset vs2022 && cmake --build --preset release && ctest --preset release                 # build/RelWithDebInfo/daylight.exe
cmake --preset vs2022-usd && cmake --build --preset release-usd && ctest --preset release-usd     # same exe + USD input
cd bindings/python && uv sync && uv run python -m daylight.check                                  # Python (usd-core, numpy)
cd bindings/ts && npm install && npm run build && npm test                                        # TypeScript (vitest)
```

The build compiles `src/trace_dev.cu` to OptiX-IR with `nvcc -optix-ir -arch=compute_75` and embeds it; the rest is
plain C++17 (`/W4 /WX`) plus one CUDA host file. The default build has no USD / glTF dependency.

## Flags

Complete CLI reference (`daylight --help` prints the short form). The C++ `daylight::Options` field for each flag
is given for in-process callers; the Python and TypeScript `run(..., args=[...])` pass flags through verbatim.

| flag | value | default | meaning | `Options` |
|---|---|---|---|---|
| `FILE...` | `.dmesh` `.obj` `.usd*` | required (or `--bench`) | sensor meshes: analysed, and blocking unless `--no-self-occlude` | `Mesh::sensor = true` |
| `--context FILE` | mesh file, repeatable | none | blocks rays only, never analysed | `Mesh::sensor = false` |
| `--on` | `faces` \| `vertices` | `faces` | element = polygon (sensor at the centroid, face normal) or point (area-weighted vertex normal) | `onVertices` |
| `--mode` | `physical` \| `ladybug` | `physical` | sky/sun model, see [Modes](#modes) | `mode` |
| `--quantity` | `Q[,Q...]` | `radiation` | one output column per quantity: `radiation` kWh/m2 over the run, `direct` (sun only), `diffuse` (sky + ground), `irradiance` mean W/m2 (instantaneous for a single `--time`), `illuminance` mean lux (physical mode), `sunhours` hours of unoccluded sun | `quantities` |
| `--sides` | `front` \| `back` \| `both` | `front` | receiving hemisphere of each element; `both` = two columns per quantity (`_front`, `_back`) | `sides` |
| `--cull-backfaces` | flag | off | occluder triangles block only from their front (winding) side; default both sides | `cullBackfaces` |
| `--no-self-occlude` | flag | off | sensor meshes do not block rays (only `--context` files do) | `selfOcclude = false` |
| `--epw FILE` | EPW path | none | hourly weather (DNI/DHI) and location from the header; 8760 or 8784 rows, hour-ending, sun evaluated at mid-hour | `read_epw` -> `Weather` |
| `--clear-sky` | flag | off | no weather file: synthesise an hourly clear-sky year (Ineichen-Perez model, pvlib formulation) for `--lat/--lon`; cloudless upper bound, typically 20-50 % above a TMY; exclusive with `--epw` | `clear_sky_weather(lat, lon, tz, TL, alt, w)` |
| `--turbidity TL` | Linke turbidity | `3` | atmosphere for `--clear-sky`: 2 very clear mountain, 3 clear, 4-5 hazy / urban, 6-7 polluted | argument of `clear_sky_weather` |
| `--altitude M` | metres | `0` | site elevation for `--clear-sky` (air pressure, model coefficients) | argument of `clear_sky_weather` |
| `--lat DEG` `--lon DEG` | degrees, N and E positive | EPW header | location; required without `--epw` (sun hours, clear sky); overrides the EPW header | `lat`, `lon` |
| `--tz H` | hours, E positive | EPW header, else `round(lon / 15)` | standard-time zone of all timestamps (no DST) | `tz` |
| `--time TS` | `[YYYY-]MM-DDTHH:MM` | | evaluate one instant (year ignored; Feb 29 only with a leap EPW) | `t0` |
| `--from TS --to TS` | timestamps | | evaluate the range `[from, to)` in `--step` increments | `t0`, `t1` |
| `--step MIN` | minutes | `60` | step of `--from/--to` (DNI/DHI of the containing hour) | `stepMin` |
| (no time flags) | | | annual: one step per weather row at the mid-hour | `t0 = t1 = NAN` |
| `--sky-density` | `1` \| `2` \| `4` | `1` | sky patches 145 / 577 / 2305 (Reinhart MF); ladybug mode: 1 or 2 | `mf` |
| `--patch-samples K` | square number | `1` | physical mode: K stratified rays per patch (4, 9, 16 smooth the patch discretisation) | `patchSamples` |
| `--ground RHO` | 0..1 | `0.2` | ground reflectance (Lambertian ground = mirrored sky) | `ground` |
| `--north DEG` | degrees, counter-clockwise from +Y | `0` | model north (ladybug convention; 90 = west) | `north` |
| `--offset M` | metres | `0` | sensors moved along their normal (Grasshopper uses 0.1) | `offset` |
| `--out FILE[.npy]` | path | stdout | values: text (one line per element, `%.10g` columns) or NumPy float64 `[N x C]`; manifest at `FILE.json` | `Result` |
| `--timeseries F.npy` | path | off | per-step values, float32 `[N x T]` (W/m2, or lux for illuminance) for the non-night steps; `F.npy.hoys.txt` lists the hours of year | `seriesPath` |
| `--dump-bits F.npy` | path | off | validation: visibility bits uint32 `[N x words]`, direction order sky \| ground \| sun | `dumpBitsPath` |
| `--dump-sky F` | path | off | validation (ladybug mode): the two `gendaymtx -A` blocks | `dumpSky` |
| `--dump-sun F` | path | off | validation: per step hoy, date, NOAA and sun.c positions, DNI, DHI | `dumpSun` |
| `--bench [NSENS NTRIS]` | counts | `1000000 10000000` | synthetic grid + boxes benchmark, prints timings, no output file | |
| `-v` | flag | off | timings and sizes on stderr (`info: read ...`, `info: run ...`, `info: output ...`) | `verbose` |
| `-h` `--help` | | | usage | |

Environment: `CUDA_VISIBLE_DEVICES` selects the GPU. Exit codes: 0 ok, 1 usage error (message + usage on stderr),
2 runtime error (file, weather, scene, CUDA/OptiX). Weather is needed for every quantity except `sunhours`; without
`--epw` or `--clear-sky` only `--quantity sunhours` runs (`--lat --lon` required, `--tz` optional).

## Output

- Rows: one per element of every sensor mesh, meshes in input order, elements in mesh order (`--on faces`: one row
  per input polygon in `counts` order, a quad is one row; `--on vertices`: one row per point in point order). Nothing
  is reordered or dropped, so `values[first:first+count]` from the manifest's `meshes` entry is exactly that mesh's
  face or point attribute.
- Columns: `columns` in the manifest, `<quantity>_<side>` (`radiation_front`, `sunhours_back`, ...), in `--quantity`
  order, sides inner.
- Manifest `FILE.json`: `{"columns": [...], "duration_h": H, "dirs": {"sky": P*K, "ground": P*K, "sun": S},
  "meshes": [{"name": "...", "first": 0, "count": N0}, ...]}`. Mesh names: `.dmesh` file stem, OBJ `o`/`g` group, USD
  prim path (C++ reader) or sanitised prim path (Python converter, see its `meshes.json`).
- Units: `radiation`/`direct`/`diffuse` kWh/m2 summed over the evaluated steps (x1000 = Wh/m2), `irradiance` W/m2
  averaged over the evaluated hours, `illuminance` lux averaged, `sunhours` hours.
- Library: `Result::values` row-major `[N x C]`, `Result::columns`, `Result::ranges` = `(name, count)` per sensor mesh
  in input order.

## Results on meshes

The values are plain per-face or per-point attributes, so writing them back is index arithmetic. The bindings do it:

- **USD (Python)**: `daylight.write_results(stage, values, manifest, prims, out, on='faces', name='daylight',
  vmin=None, vmax=None)` writes an `over` layer `out` (`.usda`/`.usdc`) with one `float[] primvars:<name>_<column>`
  per column (`interpolation = "uniform"` for faces, `"vertex"` for points) and `primvars:displayColor` (viridis of
  the first column, `vmin`/`vmax` default to the data range) on every sensor prim. The source stage is untouched; add
  the layer as a sublayer (Omniverse Kit, usdview, `Usd.Stage.Open('daylight.usda')`) and the meshes are coloured.
  `prims` = the prim paths of the sensor meshes in run order (what `split` gave you); the CLI one-liner does the same
  with `--write-usd out/daylight.usda` using its `meshes.json`. Faces dropped by the converter (USD `holeIndices`,
  faces with < 3 vertices) get `nan`; instance proxies and point-instancer prototypes are skipped with a warning
  (they cannot be overridden per instance).
- **three.js (TypeScript)**: `await applyToThree(mesh, values, {vmin, vmax, material})` sets a `color` attribute
  (per triangle when `values.length` equals the triangle count: the geometry becomes non-indexed; per vertex when it
  equals the point count) and a `MeshStandardMaterial({vertexColors: true})` unless `material: false`. Use the rows
  of the mesh from the manifest range; `fromThree`/`fromGLB` meshes are triangle lists, so faces map 1:1.
- **Anything else**: `daylight.colormap(values, vmin, vmax)` (Python, `[N x 3]` float RGB in 0..1) and
  `colormap(values, {vmin, vmax})` (TypeScript, `Float32Array` RGB) are the same 9-stop viridis with linear
  interpolation, `nan` -> grey; feed them to glTF `COLOR_0`, PLY vertex colours, a Kit extension or a shader. For a
  custom scale keep the raw numbers and colour them yourself.

## Core contract

| input | role | availability |
|---|---|---|
| `.dmesh` | native: raw arrays, one mesh per file, name = file stem (layout below) | always; the bindings write it |
| `.obj` | interop text: world metres, Z-up, right-handed winding, one `o`/`g` group per mesh, 1-based global `v` indices (negative = relative); `vt`/`vn` ignored | always |
| `.usd` `.usda` `.usdc` `.usdz` | interop via LightUSD (`read_usd`), every mesh of the stage, name = prim path | `-DDAYLIGHT_USD=ON`; the Python binding converts USD without it |

```
offset  0  char[4]   "DLM1"                 magic + version
offset  4  uint32    0                      reserved (keeps the arrays 8-byte aligned)
offset  8  uint64    n_points
offset 16  uint64    n_faces
offset 24  uint64    n_indices
offset 32  float64   xyz[3 * n_points]      world metres, Z-up
           int32     counts[n_faces]        vertices per face (>= 3), right-handed winding
           int32     indices[n_indices]     0-based, sum(counts) == n_indices
```

Positional files are analysed (sensors) and, by default, also block rays; `--context FILE` files block only (roles
are per file). Location and time: `--epw FILE` gives weather and location, `--clear-sky` synthesises weather for
`--lat --lon`, `--lat --lon --tz` alone serve sun-hour runs; `--time`, `--from/--to/--step` select instants, the
default is annual.

## Semantics

- **Separate arrays per mesh in, one scene inside.** Every mesh (one `.dmesh` / `daylight::Mesh`, one OBJ group,
  one USD prim) keeps its own points and indices. Internally all occluding triangles go into one OptiX acceleration
  structure and all sensors into one array; results come back in input order and the manifest `meshes` ranges
  (`Result::ranges`) map rows back to meshes.
- **Sensor meshes block by default.** A sensor mesh is also an occluder (self-shading, like the analysed geometry in
  a Radiance scene); `--no-self-occlude` leaves sensor meshes out of the acceleration structure. An element never
  blocks its own rays: the ray origin is nudged along the normal (Waechter-Binder) and hits closer than 1e-5 m are
  ignored.
- **Hemisphere cull.** Per sensor, directions with a non-positive cosine against its normal are never traced and
  contribute nothing. No sensor is dropped; every element gets a row, even one looking into a wall (then 0).
- **`--cull-backfaces`.** Occluder triangles whose front side faces away from the ray are ignored, decided per ray
  inside the RT-core traversal (OptiX ray flag); nothing is pre-filtered on the host. Default: triangles block from
  both sides, as Radiance black plastic.
- **`--on faces` vs `--on vertices`.** Faces: one element per polygon (quads and n-gons are one element each),
  sensor at the centroid, direction the face normal. Vertices: one element per point, direction the area-weighted
  mean of the adjacent face normals (a point shared by faces of several meshes belongs to each mesh's output).
- **`--sides both`** evaluates the front and the back hemisphere of each element (two columns per quantity).
- **C++ callers need no files**: build `daylight::Mesh` objects and call `daylight::run`; `.dmesh` exists so other
  languages and the CLI pass the same raw arrays without parsing.

## Modes

- `--mode physical` (default): patch-centre sky vectors (`--sky-density 1|2|4` = 145/577/2305 patches,
  `--patch-samples K` stratified rays per patch), exact sun vector per step (never binned), Lambertian ground
  (`--ground RHO`), NOAA sun position with refraction (Ladybug Sunpath semantics).
- `--mode ladybug`: emulates `ladybug_radiance.study.RadiationStudy` / `DirectSunStudy` bit for bit: Ladybug's
  trapezoid-normal patch vectors for rays and cosines, `gendaymtx -A -O1 -d/-s` float accumulation with `%.3g`
  rounding, sun spread over the 4 nearest patches, sun.c positions, ground = reflectance x mean sky patch.
  Use with `--no-self-occlude --offset 0` for library parity (the Grasshopper component uses 0.1 m and self-shading).

Weather sources, both modes: an EPW (`--epw`, measured/typical hourly DNI and DHI) or `--clear-sky`: for every hour
of a 365-day year the Ineichen-Perez clear-sky model (as in pvlib `clearsky.ineichen`: Kasten-Young airmass,
Spencer extraterrestrial irradiance, Linke turbidity `--turbidity`, site `--altitude`) at the mid-hour sun
position gives DNI and DHI, which then feed the same Perez sky. It is deterministic and needs no download, but it
has no clouds: use it for relative comparisons (which facade, which overhang), not for absolute yields.

## Bindings

Each binding lives in its own folder with its own tests and README; none changes the core.

- **Python + USD** (`bindings/python`, uv project, `usd-core`): `run_meshes` / `to_dmesh` for raw arrays,
  `meshes` / `split` for a composed stage (instancing, visibility, purpose, leftHanded, mirrored transforms,
  metersPerUnit, up axis; all vectorised, flat arrays out), `write_results` / `colormap` for results on the stage,
  `python -m daylight stage.usd --sensors /World/Bldg -o out [--write-usd out/daylight.usda] -- <core flags>` as the
  one-liner (writes one `.dmesh` per prim + `meshes.json`, results in `values.npy`). See `bindings/python/README.md`.
- **C++ + USD** (`bindings/cpp`, LightUSD): the `daylight_usd` library (`read_usd(path, upAxis, metersPerUnit,
  meshes, err)`) and its `check_usd` test; with `-DDAYLIGHT_USD=ON` the CLI reads `.usd*` directly. See
  `bindings/cpp/README.md`.
- **TypeScript + glTF / three.js** (`bindings/ts`, npm package `daylight-mesh`): `fromThree` / `fromGLB` into the
  core frame, `toDmesh` / `toObj` writers, `run(exe, {sensors, context, args})` drives the CLI from Node and reads
  its `.npy`, `applyToThree` / `colormap` for display. See `bindings/ts/README.md`.

## Validation

`validate/` is a uv project (Python 3.12) that downloads Radiance rad5R4 and EPW files, builds fixtures with
ladybug-geometry, computes references (gendaymtx, ladybug Sunpath, RadiationStudy, DirectSunStudy, pvlib) and compares
them with the CLI output layer by layer; `validate/ingest.py` times the mesh transport per format. See
`validate/README.md`.

## Performance (RTX 2060 6 GB, driver 592.27, `--bench`, annual Boston EPW, radiation + sun hours)

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

## Validation results (validate/compare.py, Boston + Chicago EPWs)

| layer | reference | result |
|---|---|---|
| sun position | ladybug Sunpath (8760 h, 5 sites) / gendaymtx -D / pvlib SPA | 5e-7 deg, 0 day/night flips / 5e-5 deg / 0.015 deg |
| sky matrix | gendaymtx -A -O1 ASCII (rad5R4) | byte-identical rows |
| cumulative radiation, `--mode ladybug` | ladybug-radiance RadiationStudy (faces and vertices, MF1/MF2, north 0/30) | ~1e-6 relative on every sensor after edge-ray decomposition |
| sun hours, `--mode ladybug` | ladybug-radiance DirectSunStudy | exact |
| visibility bits | rcontrib intersection matrix | all mismatches are edge-grazing or zero-cosine rays |
| `--mode physical` vs ladybug | RadiationStudy | 1-4 % mean difference (exact sun vs 4-patch binning, patch vectors), informational |
| USD (python binding) vs OBJ | same scene through both | identical except edge-grazing rays |
| `.dmesh` vs `.obj` | same meshes through both | byte-identical values |
| `--clear-sky` DNI/DHI | pvlib `Location.get_clearsky('ineichen')`, SPA positions, 8760 h, Boston TL 3 | 0.21 % max (sun > 5 deg), 4e-5 on the annual sums; model goldens 1e-6 |
| analytic | sum(DHI + DNI sin alt), sum(DNI cos) on a south wall | < 1e-6 |

## Layout

```
src/daylight.h   public API           src/internal.h  module contracts        src/params.h   OptiX launch params
src/mesh.cpp     OBJ + .dmesh readers, triangulation, sensors   src/solar.cpp   NOAA sun, EPW, clear sky, steps
src/sky.cpp      patches, gendaymtx + sun.c ports     src/run.cpp     composition, weights, chunk loop, dumps
src/gpu.cu       OptiX host + CUDA kernels            src/trace_dev.cu OptiX raygen (bits)
src/main.cpp     CLI (format dispatch, flags)         tests/check*.cpp assert-style checks (ctest)
bindings/python  USD (pxr) in and out, .dmesh writer, CLI wrapper   bindings/cpp    USD reader (LightUSD) for the CLI / C++
bindings/ts      three.js / glTF -> .dmesh, CLI wrapper, vertex colours   validate/   Python harness + ingest benchmark
```

Licence: MIT. Third-party notices: THIRD_PARTY_NOTICES.md.
