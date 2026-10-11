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

## Quick start

**CLI.** Positional files are analysed (sensor meshes), `--context FILE` files only block. Any mix of `.dmesh`, `.obj`
and (USD build) `.usd*`. `--out v.txt` writes one text line per element, `--out v.npy` a float64 `[N x C]` NumPy
array; both get a `.json` manifest next to them (columns, mesh ranges). Every flag: [docs/cli.md](docs/cli.md).

```
daylight sensors.dmesh --context city.obj --epw site.epw --quantity radiation,sunhours --sides both --out v.npy
daylight --help
```

**C++** (in-process, no files). Fill `daylight::Mesh` (points xyz in metres, Z-up; 0-based indices, right-handed
winding) and call `daylight::run`. CMake: `add_subdirectory(<repo>)` + `target_link_libraries(app PRIVATE daylightcore)`.

```cpp
#include "daylight.h"
#include <stdexcept>
daylight::Mesh m;                                  // one quad; m.sensor = false would make it context (blocks only)
m.name = "roof"; m.points = {0,0,3, 10,0,3, 10,8,3, 0,8,3}; m.counts = {4}; m.indices = {0,1,2,3};
daylight::Options opt; opt.quantities = {daylight::Quantity::Radiation, daylight::Quantity::SunHours};
daylight::Weather w; daylight::Result r; std::string err;
if (!daylight::read_epw("site.epw", w, err) || !daylight::run({m}, &w, opt, r, err)) throw std::runtime_error(err);
double roofRadiation = r.values[0], roofSunHours = r.values[1];   // r.values: N x r.columns.size(), row-major
```

**Python** (`bindings/python`). Raw arrays in, `[N x C]` NumPy array out; USD stages in and out via the CLI wrapper.

```python
import numpy as np, daylight
P = np.array([[0, 0, 3], [10, 0, 3], [10, 8, 3], [0, 8, 3]], float)       # world metres, Z-up
values, manifest = daylight.run_meshes([(P, [4], [0, 1, 2, 3])], epw='site.epw', args=['--quantity', 'radiation,sunhours'])
print(manifest['columns'], values)                                         # values: [N x C] float64, N = faces
```

```
python -m daylight scene.usd --sensors /World/Bldg -o out --write-usd out/daylight.usda -- --epw site.epw   # USD in, USD out
```

**TypeScript** (`bindings/ts`, three.js / glTF). Meshes in, one `Float64Array` per triangle out, vertex colours back.

```ts
import { fromThree, fromGLB, run, applyToThree } from 'daylight-mesh';
import fs from 'node:fs/promises';
const sensors = fromThree([roofMesh]);                               // three.js Mesh and its descendants
const context = await fromGLB(await fs.readFile('city.glb'));
const { values, columns } = await run('build/RelWithDebInfo/daylight.exe', { sensors, context, args: ['--epw', 'site.epw'] });
await applyToThree(roofMesh, values.map((v) => v[0]));               // viridis vertex colours
```

USD from C++, writing results back onto meshes and the full binding APIs: [docs/api.md](docs/api.md),
[docs/results-on-meshes.md](docs/results-on-meshes.md).

## Build

Core: VS 2022 (MSVC 14.44), CUDA 12.6, OptiX SDK 9.0 headers (`OPTIX_INSTALL_DIR`; runtime in the driver
R570+), CMake 3.27+, git (FetchContent pulls earcut.hpp, and LightUSD for the USD build). An RTX GPU is needed to run.

```
cmake --preset vs2022 && cmake --build --preset release && ctest --preset release                 # build/RelWithDebInfo/daylight.exe
cmake --preset vs2022-usd && cmake --build --preset release-usd && ctest --preset release-usd     # same exe + USD input
cd bindings/python && uv sync && uv run python -m daylight.check                                  # Python (usd-core, numpy)
cd bindings/ts && npm install && npm run build && npm test                                        # TypeScript (vitest)
```

The build compiles `src/trace_dev.cu` to OptiX-IR with `nvcc -optix-ir -arch=compute_75` and embeds it; the rest is
plain C++17 (`/W4 /WX`) plus one CUDA host file. The default build has no USD / glTF dependency.

## Documentation

- [CLI reference](docs/cli.md): every flag, exit codes, output rows / columns / manifest / units
- [Library and bindings](docs/api.md): C++ API, C++ + USD, Python, TypeScript, bindings overview
- [Results on meshes](docs/results-on-meshes.md): writing values back to USD primvars, three.js colours, colormaps
- [Input and semantics](docs/input.md): `.dmesh` / `.obj` / `.usd*` formats, sensor vs context, culling, faces vs vertices
- [Modes and weather](docs/modes.md): `physical` vs `ladybug`, EPW vs `--clear-sky`
- [Validation](docs/validation.md): harness and results against Radiance, Ladybug and pvlib
- [Performance](docs/performance.md): ray tracing and mesh transport timings
- [Source layout](docs/layout.md)

Licence: Apache 2.0 with the Commons Clause, see [LICENSE](LICENSE): free to use, modify and share, not to sell (no paid
product, hosting or consulting service whose value comes substantially from this software). Third-party notices: [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
