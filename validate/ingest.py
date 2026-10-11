# SPDX-License-Identifier: MIT
"""Ingest benchmark on a 1000x1000 quad grid (1 002 001 points, 2M triangles) -> _out/ingest/grid.{dmesh,obj,usdc,usda}.

  uv run python -I ingest.py --exe ../build/RelWithDebInfo/daylight.exe

Times (best of 2, seconds): core wall per format plus its `-v` read / output lines, the Python converter
(Stage.Open, meshes(), to_dmesh) on usdc / usda, and np.load vs np.loadtxt of the 1M-row value file."""
import argparse
import re
import subprocess
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
from daylight import meshes, to_dmesh
from pxr import Usd, UsdGeom, Vt

OUT = Path(__file__).resolve().parent / '_out' / 'ingest'
FLAGS = ['--lat', '42', '--lon', '-71', '--tz', '-5', '--quantity', 'sunhours', '--time', '06-21T12:00', '-v']
N = 1000


def build():
    OUT.mkdir(parents=True, exist_ok=True)
    x, y = np.meshgrid(np.arange(N + 1, dtype=float), np.arange(N + 1, dtype=float))
    P = np.stack([x.ravel(), y.ravel(), np.zeros((N + 1) ** 2)], 1)
    v = (np.arange(N)[:, None] * (N + 1) + np.arange(N)).ravel()               # lower-left vertex of each quad
    F = np.stack([v, v + 1, v + N + 2, v + N + 1], 1).astype(np.int32)
    to_dmesh(P, np.full(len(F), 4, np.int32), F.ravel(), OUT / 'grid.dmesh')
    with open(OUT / 'grid.obj', 'w') as f:
        np.savetxt(f, P, fmt='v %.6f %.6f %.6f', header='o grid', comments='')
        np.savetxt(f, F + 1, fmt='f %d %d %d %d')
    for ext in ('usdc', 'usda'):
        st = Usd.Stage.CreateNew(str(OUT / f'grid.{ext}'))
        UsdGeom.SetStageUpAxis(st, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(st, 1.0)
        m = UsdGeom.Mesh.Define(st, '/grid')
        m.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(P.astype(np.float32)))
        m.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(F), 4, np.int32)))
        m.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(F.ravel()))
        st.GetRootLayer().Save()


def best(fn, n=2):
    ts = []
    for _ in range(n):
        t = perf_counter()
        r = fn()
        ts.append((perf_counter() - t, r))
    return min(ts, key=lambda x: x[0])


def core(exe, mesh, out):
    cmd = [str(exe), str(mesh), *FLAGS, '--out', str(out)]
    (wall, r) = best(lambda: subprocess.run(cmd, capture_output=True, text=True, cwd=OUT))
    if r.returncode:
        return [(f'core {mesh.name}', 'needs -DDAYLIGHT_USD=ON' if mesh.suffix.startswith('.usd') else r.stderr.strip()[-80:])]
    info = re.findall(r'^info: (read|output).*?([\d.]+) s$', r.stderr, re.M)
    tag = f'core {mesh.name} -> {out.suffix}'
    return [(f'{tag} wall', f'{wall:.3f}')] + [(f'{tag} {k}', v) for k, v in info]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exe', required=True, type=Path)
    exe = ap.parse_args().exe.resolve()
    if not (OUT / 'grid.usda').exists():
        build()
    rows = []
    for ext in ('dmesh', 'obj', 'usdc', 'usda'):
        rows += core(exe, OUT / f'grid.{ext}', OUT / 'values.npy')
    rows += core(exe, OUT / 'grid.dmesh', OUT / 'values.txt')
    for ext in ('usdc', 'usda'):
        t_open, st = best(lambda: Usd.Stage.Open(str(OUT / f'grid.{ext}')), 1)     # a 2nd Open hits the layer cache
        t_mesh, (m,) = best(lambda: meshes(st))
        t_dm, _ = best(lambda: to_dmesh(*m[1:], OUT / 'py.dmesh'))
        rows += [(f'python Stage.Open grid.{ext}', f'{t_open:.3f}'), (f'python meshes() grid.{ext}', f'{t_mesh:.3f}'), (f'python to_dmesh grid.{ext}', f'{t_dm:.3f}')]
    for name, fn in ('np.load values.npy', lambda: np.load(OUT / 'values.npy')), ('np.loadtxt values.txt', lambda: np.loadtxt(OUT / 'values.txt', ndmin=2)):
        rows.append((name, f'{best(fn)[0]:.3f}'))
    print('| path | seconds |\n|---|---|\n' + '\n'.join(f'| {k} | {v} |' for k, v in rows))


if __name__ == '__main__':
    sys.exit(main())
