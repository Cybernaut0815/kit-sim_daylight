# Source layout

```
src/daylight.h   public API           src/internal.h  module contracts        src/params.h   OptiX launch params
src/mesh.cpp     OBJ + .dmesh readers, triangulation, sensors   src/solar.cpp   NOAA sun, EPW, clear sky, steps
src/sky.cpp      patches, gendaymtx + sun.c ports     src/run.cpp     composition, weights, chunk loop, dumps
src/gpu.cu       OptiX host + CUDA kernels            src/trace_dev.cu OptiX raygen (bits)
src/cli.cpp      CLI flags, weather / time rules, manifest (shared with servers)   src/main.cpp  CLI (format dispatch, output)
tests/check*.cpp assert-style checks (ctest)
bindings/python  USD (pxr) in and out, .dmesh writer, CLI wrapper   bindings/cpp    USD reader (LightUSD) for the CLI / C++
bindings/ts      three.js / glTF -> .dmesh, CLI wrapper, vertex colours   validate/   Python harness + ingest benchmark
```
