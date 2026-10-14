# Library and bindings

In-process use from C++, and the Python / TypeScript bindings that drive the CLI. CLI flags: [cli.md](cli.md).

## C++

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

Inputs from memory (uploads, no files): `read_dmesh(data, size, name, meshes, err)` parses one `.dmesh` image,
`read_epw(istream, label, w, err)` parses EPW text. A long-lived process (a server) sets `Options::sceneKey` to an
identity of its meshes + `selfOcclude`: when the key equals the previous run's, the occluder triangles and the OptiX
acceleration structure stay resident and only sensors, directions and the trace are redone; the GPU context is kept
for the whole process either way. `Options::residentScenes = N` keeps up to N keyed scenes in VRAM (least recently
used freed first, and while less than 1 GB is free), so users who alternate between models do not rebuild each other's;
`scene_resident(key)` tells whether a run would reuse one. `Options::cancel` points to an `std::atomic<bool>` another
thread may set: `run()` then stops at its next check (before the sky model, before the GPU, per trace chunk) and fails
with "cancelled". `run()` itself is one call at a time per process (one GPU, one owner thread). `src/cli.h` (`daylight::cli::parse` / `prepare` / `manifest_json`) exposes the CLI's
flag parsing, weather / time rules and manifest for front ends that accept CLI flags verbatim.

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

## Python

`run_meshes` takes `(points, counts, indices)` triples (numpy or lists), writes `.dmesh` files, runs the
CLI and loads its `.npy` output; `meshes` / `split` turn a composed USD stage (`usd-core`) into the same flat arrays;
`write_results` puts the values back on the stage as primvars (see [results-on-meshes.md](results-on-meshes.md)).

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

## TypeScript

`fromThree` / `fromGLB` produce meshes in the core frame (Y-up to Z-up, mirrored transforms
unflipped); `run` (Node) writes `.dmesh` files, runs the CLI and reads its `.npy`; `applyToThree` colours the mesh.

```ts
import { fromThree, fromGLB, run, applyToThree, toDmesh } from 'daylight-mesh';
import fs from 'node:fs/promises';
const sensors = fromThree([scene]);                                  // every descendant three.js Mesh
const context = await fromGLB(await fs.readFile('city.glb'));
const { values, columns, manifest } = await run('build/RelWithDebInfo/daylight.exe', { sensors, context, args: ['--epw', 'site.epw'] });
console.log(columns, values[0]);                                     // one Float64Array per sensor triangle
await applyToThree(roofMesh, values.slice(0, roofTriangles).map((v) => v[0]));   // vertex colours (viridis) on the mesh
const remote = await runRemote('http://gpu-box:8080', { sensors, context, epw: epwFile, args: ['--quantity', 'radiation'] }); // browser: daylight_server, meshes cached by id
```

## Bindings overview

Each binding lives in its own folder with its own tests and README; none changes the core.

- **Python + USD** (`bindings/python`, uv project, `usd-core`): `run_meshes` / `to_dmesh` for raw arrays,
  `meshes` / `split` for a composed stage (instancing, visibility, purpose, leftHanded, mirrored transforms,
  metersPerUnit, up axis; all vectorised, flat arrays out), `write_results` / `colormap` for results on the stage,
  `python -m daylight stage.usd --sensors /World/Bldg -o out [--write-usd out/daylight.usda] -- <core flags>` as the
  one-liner (writes one `.dmesh` per prim + `meshes.json`, results in `values.npy`). See [bindings/python/README.md](../bindings/python/README.md).
- **C++ + USD** (`bindings/cpp`, LightUSD): the `daylight_usd` library (`read_usd(path, upAxis, metersPerUnit,
  meshes, err)`) and its `check_usd` test; with `-DDAYLIGHT_USD=ON` the CLI reads `.usd*` directly. See
  [bindings/cpp/README.md](../bindings/cpp/README.md).
- **TypeScript + glTF / three.js** (`bindings/ts`, npm package `daylight-mesh`): `fromThree` / `fromGLB` into the
  core frame, `toDmesh` / `toObj` writers, `run(exe, {sensors, context, args})` drives the CLI from Node and reads
  its `.npy`, `applyToThree` / `colormap` for display. See [bindings/ts/README.md](../bindings/ts/README.md).
