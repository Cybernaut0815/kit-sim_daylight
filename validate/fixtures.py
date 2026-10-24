# SPDX-License-Identifier: Apache-2.0 AND LicenseRef-Commons-Clause-1.0
"""Scenes as ladybug-geometry Mesh3D -> validate/_out/fixtures/<scene>/{sensors.obj, context.obj, fixture.json}
plus usd_xform/usd_xform.usda (the box scene re-authored in USD for the Python binding, compare.py layer 6).
fixture.json holds the raw vertices / faces and the ladybug sensors (face_centroids / face_normals / vertex_normals)
that compare.py uses for the direction/edge checks (layers 4a, 4b, 6), and `optics` {mesh: [rho, tau, kind]} for the
meshes that are not the core default (opaque rho 0.2): the --optics rows and the Radiance materials (layer 8)."""
import json
import math
from pathlib import Path

import numpy as np
from daylight import to_obj
from ladybug_geometry.geometry3d import Mesh3D, Point3D
from pxr import Gf, Usd, UsdGeom, Vt

OUT = Path(__file__).resolve().parent / '_out' / 'fixtures'


def grid(origin, u, v, nu, nv, cell, outward=None, tri=False):
    """nu x nv quads (or 2 tris each) spanning origin + [0,nu*cell]u + [0,nv*cell]v; winding so the normal faces
    `outward` (default u x v)."""
    o, u, v = (np.asarray(a, float) for a in (origin, u, v))
    verts = [o + (i * cell) * u + (j * cell) * v for j in range(nv + 1) for i in range(nu + 1)]
    vid = lambda i, j: j * (nu + 1) + i
    faces = []
    for j in range(nv):
        for i in range(nu):
            q = (vid(i, j), vid(i + 1, j), vid(i + 1, j + 1), vid(i, j + 1))
            faces += [q[:3], (q[0], q[2], q[3])] if tri else [q]
    if outward is not None and np.dot(np.cross(u, v), outward) < 0:
        faces = [f[::-1] for f in faces]
    return verts, faces


def merge(parts):
    """Concatenate (verts, faces) parts into one Mesh3D, sharing vertices with equal coordinates."""
    key, pts, faces = {}, [], []
    for verts, fs in parts:
        ids = []
        for p in verts:
            k = tuple(np.round(p, 9))
            if k not in key:
                key[k] = len(pts)
                pts.append(Point3D(*p))
            ids.append(key[k])
        faces += [tuple(ids[i] for i in f) for f in fs]
    return Mesh3D(pts, faces)


def box_mesh(cx, cy, w, d, h):
    """Open box (4 sides + top) with outward normals.
    ponytail: no bottom face, so sensors at z=0 under the box are never coplanar with an occluder (Radiance and
    OptiX would disagree on t=0 hits); add it back if a closed volume is ever needed."""
    x0, x1, y0, y1 = cx - w / 2, cx + w / 2, cy - d / 2, cy + d / 2
    c = {k: (x, y, z) for k, (x, y, z) in enumerate([(x0, y0, 0), (x1, y0, 0), (x1, y1, 0), (x0, y1, 0),
                                                       (x0, y0, h), (x1, y0, h), (x1, y1, h), (x0, y1, h)])}
    faces = [(4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]   # top, S, E, N, W
    return Mesh3D([Point3D(*c[k]) for k in range(8)], faces)


def scene_flat(tri=False):
    return {'sensors': [('flat', merge([grid((-5, -5, 0), (1, 0, 0), (0, 1, 0), 10, 10, 1, (0, 0, 1), tri)]))],
            'context': []}


def scene_walls():
    walls = [('north', grid((-5, 5, 0), (1, 0, 0), (0, 0, 1), 20, 6, 0.5, (0, 1, 0))),
             ('east', grid((5, -5, 0), (0, 1, 0), (0, 0, 1), 20, 6, 0.5, (1, 0, 0))),
             ('south', grid((-5, -5, 0), (1, 0, 0), (0, 0, 1), 20, 6, 0.5, (0, -1, 0))),
             ('west', grid((-5, -5, 0), (0, 1, 0), (0, 0, 1), 20, 6, 0.5, (-1, 0, 0)))]
    return {'sensors': [(n, merge([g])) for n, g in walls], 'context': []}


def scene_box():
    s = scene_flat()
    s['context'] = [('box_a', box_mesh(0, 6, 4, 4, 6)), ('box_b', box_mesh(-8, 0, 4, 4, 6))]
    return s


def scene_tilt30():
    c, s = math.cos(math.radians(30)), math.sin(math.radians(30))
    g = grid((-3, -3 * c, 0), (1, 0, 0), (0, c, s), 12, 12, 0.5, (0, -s, c))   # normal (0, -sin30, cos30): south-up
    return {'sensors': [('tilt30', merge([g]))], 'context': []}


def scene_overhang():
    wall = grid((-3, 0, 0), (1, 0, 0), (0, 0, 1), 24, 12, 0.25, (0, -1, 0))        # 6 x 3 m, faces south
    slab = grid((-3, -1.5, 3), (1, 0, 0), (0, 1, 0), 24, 6, 0.25, (0, 0, 1))      # 1.5 m deep at the top, faces up
    return {'sensors': [('overhang', merge([wall, slab]))], 'context': []}


def scene_canopy(optics):
    """flat's ground sensors under a 6 x 6 m single sheet at 3 m, off centre (x -2..4, y -3..3), 1 m quads facing down;
    optics = (rho, tau, kind) of the sheet."""
    s = scene_flat()
    s['context'] = [('canopy', merge([grid((-2, -3, 3), (1, 0, 0), (0, 1, 0), 6, 6, 1, (0, 0, -1))]))]
    s['optics'] = {'canopy': optics}
    return s


def scene_canyon():
    """Street canyon, H/W 1: two facing 20 x 10 m walls 10 m apart and the ground between them (context, opaque rho 0.2),
    a sensor grid on each wall (1 m cells, facing the street; runs use --no-self-occlude)."""
    walls = [('canyon_s', grid((-10, -5, 0), (1, 0, 0), (0, 0, 1), 20, 10, 1, (0, 1, 0))),
             ('canyon_n', grid((-10, 5, 0), (1, 0, 0), (0, 0, 1), 20, 10, 1, (0, -1, 0)))]
    ground = grid((-10, -5, 0), (1, 0, 0), (0, 1, 0), 20, 10, 1, (0, 0, 1))
    return {'sensors': [(n, merge([g])) for n, g in walls],
            'context': [(f'{n}_wall', merge([g])) for n, g in walls] + [('ground', merge([ground]))]}


# canopy: compare.py transmit; glass_canopy and canyon: for the reflected-light layer
SCENES = {'flat': scene_flat, 'flat_tri': lambda: scene_flat(True), 'walls': scene_walls, 'box': scene_box,
          'tilt30': scene_tilt30, 'overhang': scene_overhang, 'canopy': lambda: scene_canopy((0.2, 0.5, 'porous')),
          'glass_canopy': lambda: scene_canopy((0, 0.64, 'glass')), 'canyon': scene_canyon}


def write_obj(path, meshes):
    to_obj([(n, m.vertices, np.fromiter(map(len, m.faces), np.int32, len(m.faces)),
             np.fromiter((i for f in m.faces for i in f), np.int32)) for n, m in meshes], path)


def mesh_json(name, m):
    return {'name': name, 'vertices': [list(p) for p in m.vertices], 'faces': [list(f) for f in m.faces],
            'face_centroids': [list(p) for p in m.face_centroids], 'face_normals': [list(n) for n in m.face_normals],
            'vertex_normals': [list(n) for n in m.vertex_normals]}


def yup_cm(m):
    """Mesh3D world points (metres, Z-up) -> the usd_xform stage units: Y-up (x, z, -y), centimetres."""
    return np.array([list(p) for p in m.vertices])[:, [0, 2, 1]] * [1, 1, -1] / 0.01


def define_mesh(stage, path, m, xf=None, left=False):
    """Author m (world metres, Z-up) at prim path in the Y-up cm stage so that reading it back through xf gives m.
    USD winding is local-space, so the faces are reversed iff left != (det3(xf) < 0): a mirroring xf turns the
    outward world winding inside-out locally, and leftHanded flips it once more."""
    pts, faces = yup_cm(m), m.faces
    if xf is not None:
        pts = np.array([Gf.Matrix4d(xf).GetInverse().Transform(Gf.Vec3d(*p)) for p in pts])
    if left != (xf is not None and Gf.Matrix4d(xf).GetDeterminant3() < 0):
        faces = [f[::-1] for f in faces]
    prim = UsdGeom.Mesh.Define(stage, path)
    if left:
        prim.CreateOrientationAttr(UsdGeom.Tokens.leftHanded)
    prim.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*map(float, p)) for p in pts]))
    prim.CreateFaceVertexCountsAttr(Vt.IntArray([len(f) for f in faces]))
    prim.CreateFaceVertexIndicesAttr(Vt.IntArray([int(i) for f in faces for i in f]))
    return prim


def write_usd_xform(path):
    """box scene, Y-up, cm, nested rotations, leftHanded sensors, mirrored (negative scale) box_b, one guide, one
    inactive and one invisible lid (all three must be pruned: a lid facing down blocks even under --cull-backfaces)."""
    scene = scene_box()
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(stage, 0.01)
    rotA = Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0, 1, 0), 37.0)) * Gf.Matrix4d().SetTranslate(Gf.Vec3d(120, 0, -40))
    rotB = Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(1, 0, 0), 90.0))
    UsdGeom.Xform.Define(stage, '/World')
    UsdGeom.Xform.Define(stage, '/World/context')
    UsdGeom.Xform.Define(stage, '/World/Group').AddTransformOp().Set(rotA)
    UsdGeom.Xform.Define(stage, '/World/Group/Sub').AddTransformOp().Set(rotB)
    define_mesh(stage, '/World/Group/Sub/sensors_flat', scene['sensors'][0][1], rotB * rotA, left=True)
    (_, box_a), (_, box_b) = scene['context']
    define_mesh(stage, '/World/context/box_a', box_a)
    # box_b = mirror of box_a: world_b = world_a * S(-1,1,1) * T; author box_b under that xform
    mir = Gf.Matrix4d().SetScale(Gf.Vec3d(-1, 1, 1)) * Gf.Matrix4d().SetTranslate(Gf.Vec3d(-800, 0, 600))
    world_b = np.array([mir.Transform(Gf.Vec3d(*p)) for p in yup_cm(box_a)])
    assert np.allclose(sorted(map(tuple, world_b)), sorted(map(tuple, yup_cm(box_b))), atol=1e-6), 'mirror does not land on box_b'
    UsdGeom.Xform.Define(stage, '/World/context/box_b_mirrored').AddTransformOp().Set(mir)
    define_mesh(stage, '/World/context/box_b_mirrored/mesh', box_b, mir)
    # a big slab 1 m above the grid, facing down: would zero every sensor if not pruned
    lid = merge([grid((-20, -20, 1), (1, 0, 0), (0, 1, 0), 1, 1, 40, (0, 0, -1))])
    define_mesh(stage, '/World/context/guide_lid', lid).CreatePurposeAttr(UsdGeom.Tokens.guide)
    define_mesh(stage, '/World/context/inactive_lid', lid).GetPrim().SetActive(False)
    UsdGeom.Xform.Define(stage, '/World/context/hidden').CreateVisibilityAttr(UsdGeom.Tokens.invisible)
    define_mesh(stage, '/World/context/hidden/lid', lid)
    stage.GetRootLayer().Save()


def main():
    for name, build in SCENES.items():
        scene = build()
        d = OUT / name
        d.mkdir(parents=True, exist_ok=True)
        write_obj(d / 'sensors.obj', scene['sensors'])
        if scene['context']:
            write_obj(d / 'context.obj', scene['context'])
        js = {'sensors': [mesh_json(n, m) for n, m in scene['sensors']],
              'context': [mesh_json(n, m) for n, m in scene['context']], 'optics': scene.get('optics', {})}
        (d / 'fixture.json').write_text(json.dumps(js))
        ns = sum(len(m.faces) for _, m in scene['sensors'])
        nc = sum(len(m.faces) for _, m in scene['context'])
        print(f'{name}: {ns} sensor faces, {nc} context faces')
    (OUT / 'usd_xform').mkdir(parents=True, exist_ok=True)
    write_usd_xform(OUT / 'usd_xform' / 'usd_xform.usda')
    print('usd_xform.usda written')


if __name__ == '__main__':
    main()
