# SPDX-License-Identifier: MIT
"""python -m daylight.check: `to_dmesh` round trip and hole / winding / degenerate-face masking on an in-memory
stage (no exe needed), then convert validate/_out/fixtures/usd_xform/usd_xform.usda (fixtures.py) and compare
counts, winding and world points with values derived from the fixture generator (scene_box authored on a Y-up cm
stage: nested rotations, leftHanded sensors, mirrored box_b, guide / inactive / invisible lids), then `run_meshes`
end to end when the exe exists (sun hours of one quad with / without a slab above it), then the colormap stops and
`write_results` (hole -> nan / grey, vertex mode, mismatch, skipped instance proxy / instancer name)."""
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from pxr import Sdf, Usd, UsdGeom, Vt

from .core import VIRIDIS, colormap, run_meshes, to_dmesh
from .usd import meshes, to_obj, write_results

FIX = Path(__file__).resolve().parents[3] / 'validate' / '_out' / 'fixtures' / 'usd_xform' / 'usd_xform.usda'
QUAD = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], float)


def check_dmesh():
    P = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [2, 0, 1]], float)
    counts, idx = [4, 3], [0, 1, 2, 3, 1, 4, 2]
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / 'm.dmesh'
        to_dmesh(P, counts, idx, f)
        b = f.read_bytes()
        assert struct.unpack_from('<4sIQQQ', b) == (b'DLM1', 0, 5, 2, 7) and len(b) == 32 + 24 * 5 + 4 * 2 + 4 * 7, len(b)
        assert np.array_equal(np.frombuffer(b, '<f8', 15, 32).reshape(-1, 3), P)
        assert np.frombuffer(b, '<i4', 2, 152).tolist() == counts and np.frombuffer(b, '<i4', 7, 160).tolist() == idx
        for bad in (counts, idx[:-1]), ([2, 2, 3], idx), (counts, idx[:-1] + [5]):    # sum, count < 3, index range
            try:
                to_dmesh(P, *bad, f)
                raise AssertionError(bad)
            except ValueError:
                pass
        to_obj([('e', P[:2], [], [])], f)
        assert f.read_text() == 'o e\nv 0 0 0\nv 1 0 0\n', f.read_text()       # 0 faces: no `f` line


def check_masking():
    """counts [4,3,2,3], face 1 a hole, face 2 degenerate, leftHanded on a Z-up metre stage -> 2 faces, reversed."""
    st = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(st, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(st, 1.0)
    m = UsdGeom.Mesh.Define(st, '/m')
    m.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.arange(21, dtype=np.float32).reshape(7, 3)))
    m.CreateFaceVertexCountsAttr(Vt.IntArray([4, 3, 2, 3]))
    m.CreateFaceVertexIndicesAttr(Vt.IntArray([0, 1, 2, 3, 1, 4, 2, 5, 6, 2, 4, 6]))
    m.CreateHoleIndicesAttr(Vt.IntArray([1, 9]))
    m.CreateOrientationAttr(UsdGeom.Tokens.leftHanded)
    (name, P, C, I), = meshes(st)
    assert name == '/m' and P.dtype == np.float64 and np.array_equal(P, np.arange(21).reshape(7, 3)), (name, P)
    assert C.tolist() == [4, 3] and I.tolist() == [3, 2, 1, 0, 6, 4, 2], (C, I)
    assert C.dtype == I.dtype == np.int32
    m.GetFaceVertexCountsAttr().Set(Vt.IntArray([4, -1, 6, 3]))     # negative count: ValueError, not a numpy crash
    try:
        meshes(st)
        raise AssertionError('negative count accepted')
    except ValueError as e:
        assert '/m' in str(e), e


def check_run(tries=5):
    args = ['--quantity', 'sunhours', '--time', '06-21T12:00', '--lat', '42', '--lon', '-71', '--tz', '-5']
    slab = (QUAD * 20 + [-10, -10, 2], [4], [0, 1, 2, 3])           # 20 x 20 m, 2 m above the sensor quad
    for k in range(tries):
        try:
            v, m = run_meshes([(QUAD, [4], [0, 1, 2, 3])], [slab], args=args)
            assert v.shape == (1, 1) and v[0, 0] == 0 and [x['name'] for x in m['meshes']] == ['sensor_0'], (v, m)
            v, _ = run_meshes((QUAD, [4], [0, 1, 2, 3]), args=args)
            assert v[0, 0] == 1, v
            return print('run_meshes OK')
        except (OSError, subprocess.CalledProcessError, ValueError) as e:     # missing / old / rebuilding exe
            print(f'exe not usable ({e}); retry {k + 1}/{tries} in 60 s', file=sys.stderr)
            time.sleep(60)
    print('skip: run_meshes (no usable exe)')


def check_colormap():
    rgb = colormap([0, 0.3, 1, np.nan])                                 # default range = finite min/max = 0..1
    assert np.allclose(rgb[[0, 2]], VIRIDIS[[0, 8]]) and (rgb[3] == 0.5).all(), rgb
    assert np.allclose(rgb[1], [0.206931, 0.372933, 0.550578], atol=1e-5), rgb[1]
    assert np.array_equal(colormap([2.0, 2.0]), VIRIDIS[[0, 0]]) and colormap([]).shape == (0, 3)
    assert np.allclose(colormap([5], vmin=0, vmax=10), colormap([0, 0.5, 1])[1])


def check_write_results():
    with tempfile.TemporaryDirectory() as d:
        st = Usd.Stage.CreateNew(f'{d}/src.usda')
        m = UsdGeom.Mesh.Define(st, '/m')                                 # 4 points, 3 faces, face 1 a hole
        m.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.arange(12, dtype=np.float32).reshape(4, 3)))
        m.CreateFaceVertexCountsAttr(Vt.IntArray([3, 3, 3]))
        m.CreateFaceVertexIndicesAttr(Vt.IntArray([0, 1, 2, 1, 2, 3, 0, 2, 3]))
        m.CreateHoleIndicesAttr(Vt.IntArray([1]))
        UsdGeom.Mesh.Define(st, '/proto/mesh')
        inst = st.DefinePrim('/inst')
        inst.GetReferences().AddInternalReference('/proto')
        inst.SetInstanceable(True)
        st.Save()
        out, man = f'{d}/sub/daylight.usda', {'columns': ['a', 'b'], 'meshes': [{'first': 0, 'count': 2}]}
        assert write_results(st, [[1, 10], [3, 30]], man, ['/m'], out) == ['/m']
        so = Usd.Stage.Open(out)
        assert Sdf.Layer.Find(out).subLayerPaths == ['../src.usda'] and len(UsdGeom.Mesh(so.GetPrimAtPath('/m')).GetPointsAttr().Get()) == 4
        pv = UsdGeom.PrimvarsAPI(so.GetPrimAtPath('/m'))
        a, b, col = (np.array(pv.GetPrimvar(k).Get()) for k in ('daylight_a', 'daylight_b', 'displayColor'))
        assert a.shape == b.shape == (3,) and a[[0, 2]].tolist() == [1, 3] and b[2] == 30 and np.isnan(a[1]) and np.isnan(b[1]), (a, b)
        assert pv.GetPrimvar('daylight_a').GetInterpolation() == 'uniform' and pv.GetPrimvar('displayColor').GetInterpolation() == 'uniform'
        assert col.shape == (3, 3) and np.allclose(col[1], 0.5) and np.allclose(col[[0, 2]], VIRIDIS[[0, 8]], atol=1e-6), col
        assert write_results(str(st.GetRootLayer().realPath), np.arange(4.0), {'columns': ['v'], 'meshes': [{'first': 0, 'count': 4}]},
                             ['/m'], out, on='vertices') == ['/m']
        pv = UsdGeom.PrimvarsAPI(so.GetPrimAtPath('/m')).GetPrimvar('daylight_v')     # `so` follows the rewritten layer
        assert len(pv.Get()) == 4 and pv.GetInterpolation() == 'vertex'
        try:
            write_results(st, np.zeros(5), {'columns': ['v'], 'meshes': [{'first': 0, 'count': 5}]}, ['/m'], out)
            raise AssertionError('mismatch accepted')
        except ValueError as e:
            assert '/m' in str(e), e
        assert st.GetPrimAtPath('/inst/mesh').IsInstanceProxy()
        man = {'columns': ['v'], 'meshes': [{'first': 0, 'count': 1}, {'first': 1, 'count': 1}, {'first': 2, 'count': 1}]}
        assert write_results(st, np.zeros(3), man, ['/inst/mesh', '/pi[0]/proto/mesh', '/nope'], out) == []


def main():
    check_dmesh()
    check_colormap()
    check_write_results()
    check_masking()
    if not FIX.exists():
        print(f'skip: {FIX} missing (run validate/fixtures.py)')
    else:
        ms = {name: (p, c, i) for name, p, c, i in meshes(str(FIX))}
        assert {k: len(v[1]) for k, v in ms.items()} == {'/World/Group/Sub/sensors_flat': 100, '/World/context/box_a': 5,
                                                         '/World/context/box_b_mirrored/mesh': 5}, list(ms)
        p, c, i = ms['/World/Group/Sub/sensors_flat']         # 10x10 grid from (-5,-5,0), authored leftHanded under rotations
        assert np.allclose(p[0], (-5, -5, 0), atol=1e-5) and np.allclose(p[120], (5, 5, 0), atol=1e-5), (p[0], p[120])
        assert c.tolist() == [4] * 100 and i[:4].tolist() == [0, 1, 12, 11], i[:4]   # flipped back to the right-handed +Z winding
        p = ms['/World/context/box_a'][0]                     # box_mesh(0, 6, 4, 4, 6): vertex 0 = (x0, y0, 0)
        assert np.allclose(p[0], (-2, 4, 0), atol=1e-5), p[0]
        p, _, i = ms['/World/context/box_b_mirrored/mesh']    # box_mesh(-8, 0, 4, 4, 6) through a scale(-1,1,1) xform
        assert np.allclose(p[0], (-10, -2, 0), atol=1e-5) and np.allclose(p[6], (-6, 2, 6), atol=1e-5), (p[0], p[6])
        assert i[:4].tolist() == [4, 5, 6, 7], i[:4]        # mirror flip undone: top face winds +Z (outward)
    check_run()
    print('check OK')
    return 0


if __name__ == '__main__':
    sys.exit(main())
