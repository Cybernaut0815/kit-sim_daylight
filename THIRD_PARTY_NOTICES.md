# Third-party notices

This project (Apache 2.0 with the Commons Clause, see LICENSE) contains or uses the following third-party material.

## Radiance (gendaymtx.c, sun.c, color.h) -- ported code in src/sky.cpp

The Perez all-weather sky model, sky patch subdivision (Reinhart/Tregenza), sun-to-patch binning and the simple
solar position in src/sky.cpp are ported from Radiance 5.4 (tag rad5R4), files src/gen/gendaymtx.c, src/gen/sun.c
and src/common/color.h, under the Radiance Software License, Version 2.0 reproduced below. The Perez sky code in
gendaymtx.c additionally carries the notice of byHeart Consultants Limited (Ian Ashdown), which is reproduced at the
top of src/sky.cpp. Radiance is a registered trademark of The Regents of the University of California; this project
is not endorsed by or affiliated with LBNL or the University of California.

```
The Radiance Software License, Version 2.0

Radiance v5.4 Copyright (c) 1990 to 2022, The Regents of the University of
California, through Lawrence Berkeley National Laboratory (subject to receipt
of any required approvals from the U.S. Dept. of Energy).  All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

(1) Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.

(2) Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

(3) Neither the name of the University of California, Lawrence Berkeley
National Laboratory, U.S. Dept. of Energy nor the names of its contributors
may be used to endorse or promote products derived from this software
without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.

You are under no obligation whatsoever to provide any bug fixes, patches,
or upgrades to the features, functionality or performance of the source
code ("Enhancements") to anyone; however, if you choose to make your
Enhancements available either publicly, or directly to Lawrence Berkeley
National Laboratory, without imposing a separate written license agreement
for such Enhancements, then you hereby grant the following license: a
non-exclusive, royalty-free perpetual license to install, use, modify,
prepare derivative works, incorporate into other computer software,
distribute, and sublicense such enhancements or derivative works thereof,
in binary and source code form.
```

## LightUSD (formerly TinyUSDZ) -- bindings/cpp (optional, fetched only with DAYLIGHT_USD=ON, tag v1.0.0-rc4)

```
Apache 2.0 License 

Copyright (c) 2020-2023 Syoyo Fujita
Copyright (c) 2024-Present Light Transport Entertainment Inc.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
```

LightUSD bundles third-party components under their own licences (see LICENSE.3rdparty in its repository); they are
compiled into the static library used by bindings/cpp/daylight_usd.cpp. The core (src/) does not use LightUSD.

## earcut.hpp -- polygon triangulation (fetched at configure time, tag v2.2.4)

```
ISC License

Copyright (c) 2015, Mapbox

Permission to use, copy, modify, and/or distribute this software for any purpose
with or without fee is hereby granted, provided that the above copyright notice
and this permission notice appear in all copies.

THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH
REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND
FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT,
INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER
TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF
THIS SOFTWARE.
```

## NVIDIA OptiX SDK headers

The OptiX API headers are NVIDIA proprietary (DesignWorks licence) and are NOT redistributed with this project: the
build reads them from an installed OptiX SDK (OPTIX_INSTALL_DIR). The OptiX runtime (nvoptix.dll) is part of the
NVIDIA display driver.

## Run-time dependencies of the bindings (imported, not redistributed)

- bindings/python: usd-core (Pixar USD, Apache-2.0 / TOMORROW OPEN SOURCE TECHNOLOGY LICENSE), numpy (BSD-3-Clause),
  installed by uv from PyPI.
- bindings/ts: three.js (MIT, peer dependency) and @gltf-transform/core (MIT), installed by npm.

## Other references (no code copied)

- Ray origin offset: C. Waechter and N. Binder, "A Fast and Robust Method for Avoiding Self-Intersection",
  Ray Tracing Gems, 2019 (algorithm re-implemented from the paper in src/trace_dev.cu).
- Solar position: NOAA Solar Calculator equations (Jean Meeus, Astronomical Algorithms), public domain formulas,
  re-implemented in src/solar.cpp with the same conventions as Ladybug Tools so results can be compared.
- Ladybug Tools (AGPL-3.0) is used only as a validation oracle in validate/ (separate Python environment); it is
  not linked into this software and no Ladybug code was copied. Its patch-vector convention (flat trapezoid
  normals) is re-derived from geometry in src/sky.cpp.
- Weather files (EPW) are not redistributed; validate/setup.py downloads them from their public sources.
