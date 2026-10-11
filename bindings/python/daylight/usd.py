# SPDX-License-Identifier: MIT
"""USD stage -> list of (name, points [N,3] float64 world metres Z-up, counts int32 [F], indices int32 [sum(counts)])
with right-handed winding (the `.dmesh` arrays).

Composed stage, instance proxies and PointInstancers expanded, invisible subtrees / guide / proxy / inactive prims
dropped, holes and faces with < 3 vertices dropped, authored normals ignored, the subdivision control cage taken as
the geometry."""
import os
import sys
from pathlib import Path

import numpy as np
from pxr import Sdf, Usd, UsdGeom, Vt

from .core import colormap

PI = UsdGeom.PointInstancer


def _keep(m, C, t):
    """Faces the core receives: count >= 3 (.dmesh rejects degenerate faces) and not in holeIndices."""
    keep = C >= 3
    h = np.asarray(m.GetHoleIndicesAttr().Get(t) or (), int)
    keep[h[(h >= 0) & (h < len(C))]] = False
    return keep


def meshes(stage, time=None):
    if isinstance(stage, str):
        stage = Usd.Stage.Open(stage)          # raises Tf.ErrorException on a missing/unreadable file
    t = Usd.TimeCode(stage.GetStartTimeCode() if time is None else time)
    mpu = UsdGeom.GetStageMetersPerUnit(stage)            # 0.01 when unauthored (USD fallback)
    base = np.diag([mpu, mpu, mpu, 1.0])
    if UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.y:  # Y-up (x, y, z) -> Z-up (x, -z, y), det +1
        base[:3, :3] = [[mpu, 0, 0], [0, 0, mpu], [0, -mpu, 0]]
    out = []
    _walk(stage, stage.GetPseudoRoot(), base, '', t, UsdGeom.XformCache(t), out)
    return out


def _walk(stage, root, base, prefix, t, xc, out):
    """Every mesh under root; its world matrix is (prim relative to root) x base, row-vector convention."""
    it = iter(Usd.PrimRange(root, Usd.TraverseInstanceProxies()))
    for prim in it:
        img = UsdGeom.Imageable(prim)
        if img and (img.GetVisibilityAttr().Get(t) == UsdGeom.Tokens.invisible
                    or img.GetPurposeAttr().Get() in (UsdGeom.Tokens.guide, UsdGeom.Tokens.proxy)):
            it.PruneChildren()
            continue
        W = np.array(xc.ComputeRelativeTransform(prim, root)[0]) @ base
        if prim.IsA(PI):
            it.PruneChildren()
            pi = PI(prim)
            xf = pi.ComputeInstanceTransformsAtTime(t, t, PI.IncludeProtoXform, PI.IgnoreMask)
            mask = pi.ComputeMaskAtTime(t)
            protos = [stage.GetPrimAtPath(p) for p in pi.GetPrototypesRel().GetTargets()]
            for k, (M, pidx) in enumerate(zip(xf, pi.GetProtoIndicesAttr().Get(t) or ())):
                if not mask or mask[k]:
                    _walk(stage, protos[pidx], np.array(M) @ W, f'{prefix}{prim.GetPath()}[{k}]', t, xc, out)
        elif prim.IsA(UsdGeom.Mesh):
            m = UsdGeom.Mesh(prim)
            pts, counts, idx = (a.Get(t) for a in (m.GetPointsAttr(), m.GetFaceVertexCountsAttr(), m.GetFaceVertexIndicesAttr()))
            if not pts or not counts:
                continue
            sub = m.GetSubdivisionSchemeAttr()
            if sub.HasAuthoredValue() and sub.Get() != UsdGeom.Tokens.none:
                print(f'warning: {prim.GetPath()}: subdivisionScheme {sub.Get()}, using the control cage', file=sys.stderr)
            P = np.asarray(pts) @ W[:3, :3] + W[3, :3]
            C, I = np.asarray(counts, np.int32), np.asarray(idx or (), np.int32)
            if C.sum() != len(I) or (C < 0).any() or (len(I) and (I.min() < 0 or I.max() >= len(P))):
                raise ValueError(f'{prim.GetPath()}: faceVertexCounts / faceVertexIndices inconsistent or out of range')
            keep = _keep(m, C, t)
            I, C = I[np.repeat(keep, C)], C[keep]
            if (m.GetOrientationAttr().Get() == UsdGeom.Tokens.leftHanded) != (np.linalg.det(W[:3, :3]) < 0):
                e = np.cumsum(C)                           # USD winding is local-space: a mirror flips it
                I = I[np.repeat(2 * e - C - 1, C) - np.arange(len(I))]   # reverse every face in place
            out.append((f'{prefix}{prim.GetPath()}', P, C, I))


def split(meshes, sensors=(), context=()):
    """-> (sensor meshes, context meshes) by prim-path prefix. No sensors: every unmatched mesh is a sensor;
    no context: every non-sensor mesh is context; both given: the rest is dropped."""
    hit = lambda m, ps: any(m[0].startswith(p) for p in ps)
    s, c = [], []
    for m in meshes:
        if hit(m, sensors) or (not sensors and not hit(m, context)):
            s.append(m)
        elif hit(m, context) or not context:
            c.append(m)
    return s, c


def to_obj(meshes, path):
    """Core OBJ contract (interop only): one `o` group per mesh, world metres Z-up, 1-based global indices."""
    lines, base = [], 1
    for name, p, c, i in meshes:
        lines.append(f'o {name}')
        lines += ['v %.17g %.17g %.17g' % tuple(q) for q in p]
        lines += ['f ' + ' '.join(str(k + base) for k in f) for f in np.split(np.asarray(i), np.cumsum(c))[:-1]]
        base += len(p)
    Path(path).write_text('\n'.join(lines) + '\n')


def write_results(stage, values, manifest, prims, out, on='faces', name='daylight', vmin=None, vmax=None):
    """Values [N x C] (rows per `manifest['meshes']`, one range per prim path in `prims`) -> new layer `out`
    (.usda/.usdc, overwritten) that sublayers the stage's root layer and overrides each prim with float primvars
    `<name>_<column>` (uniform / vertex per `on`) + `displayColor` = colormap(first column). Opening `out` alone
    shows the coloured scene; the source file is untouched. Faces dropped by `meshes()` (holes, < 3 vertices) get
    nan / grey; other count mismatches raise ValueError. Instance proxies and `[k]` instancer names are skipped
    with a warning. -> written prim paths."""
    if isinstance(stage, str):
        stage = Usd.Stage.Open(stage)
    V = np.asarray(values, float).reshape(len(values), -1)
    t = Usd.TimeCode(stage.GetStartTimeCode())
    src, out = stage.GetRootLayer(), str(out)
    lo, hi = (np.nanmin(V[:, 0]) if vmin is None else vmin), (np.nanmax(V[:, 0]) if vmax is None else vmax)
    lay = Sdf.Layer.Find(out) or Sdf.Layer.CreateNew(out)
    lay.Clear()
    rel = src.identifier                                                    # anon:... for an in-memory source
    if src.realPath:
        try:
            rel = os.path.relpath(src.realPath, Path(out).resolve().parent).replace(os.sep, '/')
        except ValueError:                                                  # other Windows drive: no relative path
            rel = src.realPath.replace(os.sep, '/')
    lay.subLayerPaths = [rel if rel.startswith('.') or ':' in rel else './' + rel]   # './' = anchored, not a search path
    done, st_out = [], Usd.Stage.Open(lay)
    for path, rng in zip(prims, manifest['meshes']):
        prim = None if '[' in path else stage.GetPrimAtPath(path)
        if not prim or prim.IsInstanceProxy():
            print(f'warning: {path}: not a prim on the stage or an instance proxy, skipped', file=sys.stderr)
            continue
        m = UsdGeom.Mesh(prim)
        rows = V[rng['first']:rng['first'] + rng['count']]
        if on == 'faces':
            C = np.asarray(m.GetFaceVertexCountsAttr().Get(t) or (), int)
            keep, n, interp = _keep(m, C, t), len(C), UsdGeom.Tokens.uniform
            if len(rows) == keep.sum() < n:
                full = np.full((n, V.shape[1]), np.nan)
                full[keep], rows = rows, full
        else:
            n, interp = len(m.GetPointsAttr().Get(t) or ()), UsdGeom.Tokens.vertex
        if len(rows) != n:
            raise ValueError(f'{path}: {len(rows)} values for {n} {on}')
        pv = UsdGeom.PrimvarsAPI(st_out.OverridePrim(path))
        for col, v in zip(manifest['columns'], rows.T):
            pv.CreatePrimvar(f'{name}_{col}', Sdf.ValueTypeNames.FloatArray, interp).Set(Vt.FloatArray.FromNumpy(v.astype(np.float32)))
        rgb = colormap(rows[:, 0], lo, hi).astype(np.float32)
        pv.CreatePrimvar('displayColor', Sdf.ValueTypeNames.Color3fArray, interp).Set(Vt.Vec3fArray.FromNumpy(rgb))
        done.append(path)
    lay.Save()
    return done
