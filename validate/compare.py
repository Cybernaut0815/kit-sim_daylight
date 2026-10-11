# SPDX-License-Identifier: MIT
"""Run the daylight CLI against the references (plan section 8) and print a per-case table.

  uv run compare.py <sky|sun|bits|e2e|usd|analytic|clearsky|all|selftest> --exe PATH [--epw STEM ...] [--scene NAME ...]

Exit 1 on hard tolerance failures; informative rows never fail. `selftest` exercises the parsers and the residual
decomposition on synthetic data and needs no CLI. Layer `usd` converts the usd_xform fixture with the Python binding
(bindings/python, imported from this venv as `daylight`). CLI dump formats this script reads (contract with main.cpp):
  --out F         : one line per sensor, one `%.10g` per column; F.json manifest {"columns": [...], "duration_h",
                    "dirs": {"sky", "ground", "sun"}, "meshes": [{"name", "first", "count"}]}
  --dump-bits F   : .npy uint32 [N x nWords]; direction d = word d//32, bit d%32; order [sky | ground | sun]
  --dump-sky F    : ladybug mode only; a line '# gendaymtx -s -A ...' followed by one `R G B` (%.3g) line per
                    gendaymtx row (ground first), then '# gendaymtx -d -A ...' and the same for the direct part
  --dump-sun F    : '#' comment; one line per step, 12 columns: hoy month day hour alt azi up alt_rad azi_rad up_rad
                    dni dhi (deg, 0/1; columns 5-7 NOAA, 8-10 Radiance sun.c, 11-12 W/m2 of the step's weather)
"""
import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
from ladybug.viewsphere import view_sphere

HERE = Path(__file__).resolve().parent
DL, OUT = HERE / '_dl', HERE / '_out'
REF, FIX, RUN = OUT / 'ref', OUT / 'fixtures', OUT / 'run'
DEFAULT_EPWS = ['USA_MA_Boston-City.WSO_TMY', 'USA_IL_Chicago-OHare.Intl.AP.725300_TMY3']
SCENES = ['flat', 'flat_tri', 'walls', 'box', 'tilt30', 'overhang']
LADYBUG = ['--mode', 'ladybug', '--no-self-occlude', '--offset', '0']
TOL = {'sky_ascii_rows': 0, 'sun_noaa_deg': 1e-6, 'sun_flips': 0, 'sun_radiance_deg': 1e-4, 'sun_pvlib_deg': 0.02,
       'bits_mismatch': 1e-3, 'bits_edge_m': 1e-4, 'e2e_rel': 1e-3, 'e2e_residual_rel': 1e-3, 'sunhours_edge_h': 1.0,
       'analytic_flat': 0.01, 'analytic_direct': 1e-3, 'analytic_north': 1e-6, 'usd_rel': 1e-5,
       'clearsky_rel': 5e-3, 'clearsky_sum_rel': 3e-3}
ROWS, FAILS = [], []


def report(layer, case, metric, value, limit, hard=True):
    ok = value <= limit
    ROWS.append((layer, case, metric, value, limit, 'ok' if ok else ('FAIL' if hard else 'info')))
    if hard and not ok:
        FAILS.append(f'{layer} {case} {metric} = {value:.3g} > {limit:.3g}')


# ---------------------------------------------------------------- parsers (self-tested)
def read_values(path):
    return np.loadtxt(path, ndmin=2)


def read_bits(npy_path, n_dir):
    """uint32 [N x nWords] -> bool [N x n_dir]; direction d = word d // 32, bit d % 32."""
    words = np.load(npy_path)
    assert words.dtype == np.uint32 and words.shape[1] * 32 >= n_dir, (words.dtype, words.shape, n_dir)
    bits = np.unpackbits(words.view(np.uint8), axis=1, bitorder='little')
    return bits[:, :n_dir].astype(bool)


def read_dump_sky(path):
    """-> (dif rows, dir rows) as 'R G B' strings, ground row first: the `R G B` lines after the '# gendaymtx -s -A'
    and '# gendaymtx -d -A' comment lines."""
    sec, cur = {'s': [], 'd': []}, None
    for l in Path(path).read_text().splitlines():
        if l.startswith('#'):
            cur = 's' if '-s -A' in l else 'd' if '-d -A' in l else None
        elif cur and l.strip():
            sec[cur].append(' '.join(l.split()))
    return sec['s'], sec['d']


def read_dump_sun(path):
    a = np.loadtxt(path, comments='#', ndmin=2)
    return {'hoy': a[:, 0], 'alt_noaa': a[:, 4], 'azi_noaa': a[:, 5], 'up_noaa': a[:, 6] > 0.5,
            'alt_rad': a[:, 7], 'azi_rad': a[:, 8], 'up_rad': a[:, 9] > 0.5}


def gendaymtx_A_rows(text):
    """`gendaymtx -A` ASCII: 8 header lines (7 + blank), then one `R G B` line per row (ground first)."""
    lines = text.split('\n')
    assert lines[7] == '' and 'NAVERAGED' not in text, 'not the rad5R4 8-line header'
    return [l.strip() for l in lines[8:] if l.strip()]


def read_wea(path):
    rows = [l.split() for l in Path(path).read_text().splitlines()[6:] if l.strip()]
    return np.array([[float(r[3]), float(r[4])] for r in rows])      # dni, dhi per row


def lb_dirs(mf, north):
    """Ladybug direction order: [sky patch vectors | the same reversed (ground)], rotated by north (ccw, deg)."""
    vecs = view_sphere.reinhart_dome_vectors if mf == 2 else view_sphere.tregenza_dome_vectors
    if north:
        vecs = [v.rotate_xy(math.radians(north)) for v in vecs]
    sky = np.array([list(v) for v in vecs])
    return np.vstack([sky, -sky])


def lb_weights(skymtx, n_sky):
    """W per direction from ladybug SkyMatrix data: sky = dir + dif (kWh/m2), ground = rho * mean(sky)."""
    sky = np.asarray(skymtx['dir']) + np.asarray(skymtx['dif'])
    assert sky.size == n_sky
    return np.concatenate([sky, np.full(n_sky, float(skymtx['rho']) * sky.mean())])


def residual(ours, lb, b_ours, b_lb, cos, W):
    """ours - lb - sum over differing directions of (b_ours - b_lb) * cos * W, per sensor."""
    return ours - lb - ((b_ours.astype(float) - b_lb.astype(float)) * cos * W[None, :]).sum(axis=1)


def fixture_tris(scene, include_sensors):
    js = json.loads((FIX / scene / 'fixture.json').read_text())
    tris = []
    for m in js['context'] + (js['sensors'] if include_sensors else []):
        v = np.array(m['vertices'])
        for f in m['faces']:
            for k in range(1, len(f) - 1):
                tris.append([v[f[0]], v[f[k]], v[f[k + 1]]])
    return np.array(tris).reshape(-1, 3, 3)


def edge_distance(p, d, tris):
    """Min over triangles hit by the ray (p, d) of the distance from the hit point to the triangle's edges;
    inf if no plane in front. Rays that pass within 1e-4 m of an edge are legitimate precision disagreements."""
    if len(tris) == 0:
        return math.inf
    a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]
    n = np.cross(b - a, c - a)
    denom = n @ d
    with np.errstate(divide='ignore', invalid='ignore'):
        t = np.einsum('ij,ij->i', a - p, n) / denom
    ok = np.isfinite(t) & (t > -1e-6)
    q = p + t[:, None] * d
    best = math.inf
    for e0, e1 in ((a, b), (b, c), (c, a)):
        seg = e1 - e0
        s = np.clip(np.einsum('ij,ij->i', q - e0, seg) / np.einsum('ij,ij->i', seg, seg), 0, 1)
        dist = np.linalg.norm(q - (e0 + s[:, None] * seg), axis=1)
        dist[~ok] = math.inf
        best = min(best, dist.min())
    return best


def fixture_sensors(scene, on_vertices):
    js = json.loads((FIX / scene / 'fixture.json').read_text())
    key_p, key_n = ('vertices', 'vertex_normals') if on_vertices else ('face_centroids', 'face_normals')
    p = np.vstack([np.array(m[key_p]) for m in js['sensors']])
    n = np.vstack([np.array(m[key_n]) for m in js['sensors']])
    return p, n


# ---------------------------------------------------------------- CLI driver
RUNS = {}


def cli(exe, scene_files, args, name):
    """-> (run dir, values, manifest); cached per (name, args) so layer_e2e reuses layer_bits runs."""
    key = (name, tuple(args))
    if key in RUNS:
        return RUNS[key]
    d = RUN / name
    d.mkdir(parents=True, exist_ok=True)
    out = d / 'out.txt'
    cmd = [str(exe)] + scene_files + args + ['--out', str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=d, check=False)
    if r.returncode != 0:
        raise RuntimeError(f'daylight exit {r.returncode}: {" ".join(cmd)}\n{r.stderr[-2000:]}')
    RUNS[key] = d, read_values(out), json.loads((out.parent / (out.name + '.json')).read_text())
    return RUNS[key]


def scene_files(scene):
    """meshes.json (python -m daylight output: .dmesh per prim, sensors first) if present, else the OBJ pair."""
    d = FIX / scene
    if (d / 'meshes.json').exists():
        ms = json.loads((d / 'meshes.json').read_text())
        return [str(d / m['file']) for m in ms if m['role'] == 'sensor'] \
            + [a for m in ms if m['role'] == 'context' for a in ('--context', str(d / m['file']))]
    f = [str(d / 'sensors.obj')]
    if (d / 'context.obj').exists():
        f += ['--context', str(d / 'context.obj')]
    return f


def epw_arg(stem):
    return ['--epw', str(DL / 'epw' / f'{stem}.epw')]


# ---------------------------------------------------------------- layers
def layer_sky(exe, epws, scenes):
    for stem in epws:
        for mf in (1, 2):
            d, _, _ = cli(exe, scene_files('flat'), epw_arg(stem) + LADYBUG + ['--sky-density', str(mf),
                          '--dump-sky', 'sky.txt'], f'sky_{stem}_m{mf}')
            dif, dir_ = read_dump_sky(d / 'sky.txt')
            ref_d = REF / 'epw' / stem
            bad = 0
            for ours, ref in ((dif, 's'), (dir_, 'd')):
                rows = gendaymtx_A_rows((ref_d / f'sky_m{mf}_{ref}_O1_A.txt').read_text())
                bad += sum(o != r for o, r in zip(ours, rows)) + abs(len(ours) - len(rows))
            report('2 sky', f'{stem} m{mf}', 'ASCII rows differing', bad, TOL['sky_ascii_rows'])


def layer_sun(exe, epws, scenes):
    for stem in epws:
        d, _, _ = cli(exe, scene_files('flat'), epw_arg(stem) + LADYBUG + ['--quantity', 'sunhours',
                      '--dump-sun', 'sun.txt'], f'sun_{stem}')
        s = read_dump_sun(d / 'sun.txt')
        ref_d = REF / 'epw' / stem
        sp = np.load(ref_d / 'sunpath.npz')
        assert len(s['hoy']) == len(sp['hoy']), (len(s['hoy']), len(sp['hoy']))
        up = sp['up']
        dalt = np.abs(s['alt_noaa'] - sp['alt'])
        dazi = np.abs((s['azi_noaa'] - sp['azi'] + 180) % 360 - 180)
        report('3 sun', stem, 'NOAA vs Sunpath alt deg', dalt.max(), TOL['sun_noaa_deg'])
        report('3 sun', stem, 'NOAA vs Sunpath azi deg (sun up)', dazi[up].max(), TOL['sun_noaa_deg'])
        report('3 sun', stem, 'is_during_day flips', int((s['up_noaa'] != up).sum()), TOL['sun_flips'])
        sd = np.load(ref_d / 'suns_D.npz')
        rows = np.rint(sd['hoy'] - 0.5).astype(int)
        ours = np.stack([np.cos(np.radians(s['alt_rad'])) * np.sin(np.radians(s['azi_rad'])),
                         np.cos(np.radians(s['alt_rad'])) * np.cos(np.radians(s['azi_rad'])),
                         np.sin(np.radians(s['alt_rad']))], axis=1)[rows]
        ang = np.degrees(np.arccos(np.clip((ours * sd['vec']).sum(axis=1) / np.linalg.norm(sd['vec'], axis=1), -1, 1)))
        report('3 sun', stem, 'sun.c vs gendaymtx -D vector deg', ang.max(), TOL['sun_radiance_deg'])
        report('3 sun', stem, 'sun.c up count - gendaymtx -D count', abs(int(s['up_rad'].sum()) - len(rows)), 0)
        pv = np.load(ref_d / 'pvlib.npz')
        report('3 sun', stem, 'NOAA vs pvlib SPA alt deg (sun up)', np.abs(s['alt_noaa'] - pv['alt'])[up].max(),
               TOL['sun_pvlib_deg'], hard=False)


def bits_case(exe, scene, stem, mf, north, on_vertices, quantity):
    args = epw_arg(stem) + ['--mode', 'ladybug', '--offset', '0', '--sky-density', str(mf), '--north', str(north),
                            '--on', 'vertices' if on_vertices else 'faces', '--quantity', quantity, '--dump-bits', 'bits.npy']
    if scene != 'overhang':
        args.append('--no-self-occlude')
    tag = f'm{mf}_n{north}_{"vertex" if on_vertices else "face"}'
    d, vals, man = cli(exe, scene_files(scene), args, f'{quantity}_{scene}_{stem}_{tag}')
    n_dir = man['dirs']['sky'] + man['dirs']['ground'] + man['dirs']['sun']
    return d, vals, man, read_bits(d / 'bits.npy', n_dir), tag


def layer_bits(exe, epws, scenes):
    stem = epws[0]
    for scene in scenes:
        tris = fixture_tris(scene, scene == 'overhang')
        for mf in (1, 2):
            for north in (0, 30):
                for on_v in (False, True):
                    d, _, man, bits, tag = bits_case(exe, scene, stem, mf, north, on_v, 'radiation')
                    n_sky = man['dirs']['sky']
                    imtx = np.load(REF / 'scene' / scene / f'imtx_{tag}.npy')
                    lb = imtx > 0
                    ours = bits[:, :2 * n_sky]
                    assert ours.shape == lb.shape, (ours.shape, lb.shape)
                    p, n = fixture_sensors(scene, on_v)
                    dirs = lb_dirs(mf, north)
                    graze = np.abs(n @ dirs.T) < 1e-6   # cos ~ 0: both tools contribute 0, the visible bit is sign noise
                    diff = (ours != lb) & ~graze
                    # a mismatch is explained when the ray passes within bits_edge_m of an occluder edge (undecidable tie)
                    unexplained, worst = 0, 0.0
                    for i, dd in zip(*np.nonzero(diff)):
                        ed = edge_distance(p[i], dirs[dd], tris)
                        if ed > TOL['bits_edge_m']:
                            unexplained += 1
                            worst = max(worst, ed)
                    report('4a bits', f'{scene} {tag}', 'mismatch fraction (non-grazing, edge-explained)', diff.mean(), TOL['bits_mismatch'], hard=False)
                    report('4a bits', f'{scene} {tag}', 'grazing bits excluded', int(graze.sum()), 10**9, hard=False)
                    report('4a bits', f'{scene} {tag}', 'unexplained mismatches (not within edge tol)', unexplained, 0)
                    report('4a bits', f'{scene} {tag}', 'max edge distance of unexplained (m)', worst, TOL['bits_edge_m'], hard=False)
        # DirectSunStudy bits (sun range of a sunhours run)
        for on_v in (False, True):
            d, _, man, bits, tag = bits_case(exe, scene, stem, 1, 0, on_v, 'sunhours')
            s0 = man['dirs']['sky'] + man['dirs']['ground']
            ref = np.load(REF / 'scene' / scene / f'sun_imtx_{stem}_{"vertex" if on_v else "face"}.npy')
            ours = bits[:, s0:s0 + man['dirs']['sun']]
            if ours.shape != ref.shape:
                report('4a bits', f'{scene} sun {tag}', f'shape {ours.shape} vs {ref.shape}', 1, 0)
                continue
            report('4a bits', f'{scene} sun {tag}', 'mismatch fraction', (ours != ref).mean(), TOL['bits_mismatch'])


def layer_e2e(exe, epws, scenes):
    for scene in scenes:
        for stem in epws:
            for mf in (1, 2):
                for north in (0, 30):
                    for on_v in (False, True):
                        d, vals, man, bits, tag = bits_case(exe, scene, stem, mf, north, on_v, 'radiation')
                        n_sky = man['dirs']['sky']
                        lb = np.load(REF / 'scene' / scene / f'rad_{stem}_{tag}.npy')
                        imtx = np.load(REF / 'scene' / scene / f'imtx_{tag}.npy')
                        b_lb, b_ours = imtx > 0, bits[:, :2 * n_sky]
                        _, n = fixture_sensors(scene, on_v)
                        cos = np.clip(n @ lb_dirs(mf, north).T, 0, None)
                        W = lb_weights(np.load(REF / 'epw' / stem / f'skymtx_m{mf}_n{north}.npz'), n_sky)
                        ours = vals[:, 0]
                        scale = np.maximum(np.abs(lb), 1e-3 * np.abs(lb).max())
                        case = f'{scene} {stem} {tag}'
                        # harness self-consistency: ladybug's own matrix times the reconstructed W must give its values
                        report('4b harness', case, 'imtx . W vs radiation_values rel', (np.abs(imtx @ W - lb) / scale).max(), 1e-9)
                        report('4b harness', case, 'analytic cos vs rcontrib cos, rel', (np.abs((b_lb * cos) @ W - lb) / scale).max(), 1e-3, hard=False)
                        edge = (b_ours != b_lb).any(axis=1)
                        rel = np.abs(ours - lb) / scale
                        if (~edge).any():
                            report('4b cumulative', case, f'non-edge rel err ({(~edge).sum()} sensors)', rel[~edge].max(), TOL['e2e_rel'])
                        res = np.abs(residual(ours, lb, b_ours, b_lb, cos, W)) / scale
                        report('4b cumulative', case, f'residual rel err (all, {edge.sum()} edge)', res.max(), TOL['e2e_residual_rel'])
            # sun hours vs DirectSunStudy (site of the first EPW)
            for on_v in (False, True):
                d, vals, man, bits, tag = bits_case(exe, scene, stem, 1, 0, on_v, 'sunhours')
                suffix = 'vertex' if on_v else 'face'
                ref = np.load(REF / 'scene' / scene / f'sunhours_{stem}_{suffix}.npy')
                ref_bits = np.load(REF / 'scene' / scene / f'sun_imtx_{stem}_{suffix}.npy')
                s0 = man['dirs']['sky'] + man['dirs']['ground']
                ours_bits = bits[:, s0:s0 + man['dirs']['sun']]
                dh = np.abs(vals[:, 0] - ref)
                if ours_bits.shape == ref_bits.shape:
                    edge = (ours_bits != ref_bits).any(axis=1)
                    report('4b sunhours', f'{scene} {stem} {suffix}', 'non-edge max |dh| (exact)', dh[~edge].max() if (~edge).any() else 0, 0)
                report('4b sunhours', f'{scene} {stem} {suffix}', 'all sensors max |dh| h', dh.max(), TOL['sunhours_edge_h'])
        # 4c physical vs ladybug, informative
        stem = epws[0]
        lb = np.load(REF / 'scene' / scene / f'rad_{stem}_m1_n0_face.npy')
        for k in (1, 4):
            args = epw_arg(stem) + ['--mode', 'physical', '--patch-samples', str(k), '--offset', '0']
            if scene != 'overhang':
                args.append('--no-self-occlude')
            _, vals, _ = cli(exe, scene_files(scene), args, f'phys_{scene}_{stem}_k{k}')
            rel = np.abs(vals[:, 0] - lb) / np.maximum(np.abs(lb), 1e-3 * np.abs(lb).max())
            report('4c physical', f'{scene} {stem} K={k}', 'mean rel diff vs ladybug', rel.mean(), 0.05, hard=False)


def layer_usd(exe, epws, scenes):
    """usd_xform.usda through the Python binding vs the box OBJ fixtures. Ladybug mode so the edge check's lb_dirs
    are the traced directions; --cull-backfaces makes a wrong winding (leftHanded / mirror) visible."""
    stem = epws[0]
    ux = FIX / 'usd_xform'
    subprocess.run([sys.executable, '-I', '-m', 'daylight', str(ux / 'usd_xform.usda'), '--sensors', '/World/Group',
                    '-o', str(ux)], cwd=HERE.parent / 'bindings' / 'python', check=True)
    extra = ['--mode', 'ladybug', '--cull-backfaces', '--quantity', 'radiation', '--dump-bits', 'bits.npy']
    db, box, mb = cli(exe, scene_files('box'), epw_arg(stem) + extra, f'usd_box_{stem}')
    du, usd, mu = cli(exe, scene_files('usd_xform'), epw_arg(stem) + extra, f'usd_xform_{stem}')
    report('6 usd', 'usd_xform vs box', 'sensor count difference', abs(len(usd) - len(box)), 0)
    if len(usd) != len(box):
        return
    rel = np.abs(usd - box) / np.maximum(np.abs(box), 1e-9)
    report('6 usd', 'usd_xform vs box', 'max rel diff', rel.max(), TOL['usd_rel'], hard=False)
    report('6 usd', 'usd_xform vs box', 'fraction of sensors differing > 1e-5', (rel > 1e-5).mean(), 0.05)
    # every differing visibility bit must be an edge-grazing ray (float32 cm points in the usda break exact ties)
    n_dir = mb['dirs']['sky'] + mb['dirs']['ground'] + mb['dirs']['sun']
    bb, bu = read_bits(db / 'bits.npy', n_dir), read_bits(du / 'bits.npy', n_dir)
    n_sg = mb['dirs']['sky'] + mb['dirs']['ground']
    diff = bb[:, :n_sg] != bu[:, :n_sg]
    p, n = fixture_sensors('box', False)
    dirs, tris, worst = lb_dirs(1, 0), fixture_tris('box', False), 0.0
    for i, dd in zip(*np.nonzero(diff)):
        if abs(n[i] @ dirs[dd]) >= 1e-6:
            worst = max(worst, edge_distance(p[i], dirs[dd], tris))
    report('6 usd', 'usd_xform vs box', 'sky/ground bits differing', int(diff.sum()), 10**9, hard=False)
    report('6 usd', 'usd_xform vs box', 'max edge distance of differing bits (m)', worst, TOL['bits_edge_m'])
    report('6 usd', 'usd_xform vs box', 'sun bits differing', int((bb[:, n_sg:] != bu[:, n_sg:]).sum()), 0)


def layer_analytic(exe, epws, scenes):
    stem = epws[0]
    wea = read_wea(REF / 'epw' / stem / 'wea.wea')
    d, vals, _ = cli(exe, scene_files('flat'), epw_arg(stem) + ['--mode', 'physical', '--dump-sun', 'sun.txt'], f'an_flat_{stem}')
    s = read_dump_sun(d / 'sun.txt')
    sin_alt = np.where(s['up_noaa'], np.sin(np.radians(s['alt_noaa'])), 0).clip(0)
    expect = (wea[:, 1] + wea[:, 0] * sin_alt).sum() / 1000           # kWh/m2, horizontal, unobstructed
    report('1a analytic', f'flat {stem}', 'horizontal annual vs sum(DHI + DNI sin alt) rel', abs(vals[:, 0].mean() / expect - 1), TOL['analytic_flat'])
    report('1a analytic', f'flat {stem}', 'spread across identical sensors rel', (vals[:, 0].max() - vals[:, 0].min()) / expect, 1e-6)
    _, wv, _ = cli(exe, scene_files('walls'), epw_arg(stem) + ['--mode', 'physical', '--quantity', 'direct'], f'an_walls_direct_{stem}')
    sun = np.stack([np.cos(np.radians(s['alt_noaa'])) * np.sin(np.radians(s['azi_noaa'])),
                    np.cos(np.radians(s['alt_noaa'])) * np.cos(np.radians(s['azi_noaa'])), np.sin(np.radians(s['alt_noaa']))], 1)
    cos_s = np.clip(-sun[:, 1], 0, None) * s['up_noaa']                # run.cpp: sun directions for up (refracted alt >= 0) steps
    expect = (wea[:, 0] * cos_s).sum() / 1000
    south = wv[240:360, 0]                                             # mesh order north, east, south, west
    report('1a analytic', f'walls {stem}', 'south wall direct vs sum(DNI max(0,cos)) rel', abs(south.mean() / expect - 1), TOL['analytic_direct'])
    _, w0, _ = cli(exe, scene_files('walls'), epw_arg(stem) + ['--mode', 'physical'], f'an_walls_n0_{stem}')
    _, w90, _ = cli(exe, scene_files('walls'), epw_arg(stem) + ['--mode', 'physical', '--north', '90'], f'an_walls_n90_{stem}')
    rot = np.vstack([w0[120:], w0[:120]])                              # north<-east, east<-south, south<-west, west<-north
    report('1a analytic', f'walls {stem}', '--north 90 == walls rotated by one, rel', (np.abs(w90 - rot) / np.abs(rot).max()).max(), TOL['analytic_north'])


def layer_clearsky(exe, epws, scenes):
    """--clear-sky (Ineichen, Linke turbidity 3, sea level) at Boston vs pvlib's Ineichen on the same mid-hours (pvlib
    uses SPA positions and the year 2017; the CLI's generic NOAA year is within 0.02 deg)."""
    import pandas as pd
    from pvlib.location import Location
    d, vals, _ = cli(exe, scene_files('flat'), ['--clear-sky', '--lat', '42.35', '--lon', '-71.07', '--tz', '-5',
                     '--quantity', 'irradiance', '--dump-sun', 'sun.txt'], 'clearsky_boston')
    a = np.loadtxt(d / 'sun.txt', comments='#', ndmin=2)
    assert a.shape == (8760, 12), f'sun.txt is {a.shape}, need 8760 x 12 (hoy .. up_rad dni dhi): old exe?'
    ours = {'dni': a[:, 10], 'dhi': a[:, 11]}
    loc = Location(42.35, -71.07, tz='Etc/GMT+5', altitude=0)
    times = pd.date_range('2017-01-01 00:30', periods=8760, freq='h', tz='Etc/GMT+5')
    ref = loc.get_clearsky(times, model='ineichen', linke_turbidity=3.0)
    up = loc.get_solarposition(times)['apparent_elevation'].to_numpy() > 5
    for k in ('dni', 'dhi'):
        r = ref[k].to_numpy()
        report('7 clearsky', 'boston', f'{k} max rel diff vs pvlib ineichen (elev > 5 deg)', (np.abs(ours[k] - r)[up] / r[up]).max(), TOL['clearsky_rel'])
        report('7 clearsky', 'boston', f'{k} annual sum rel diff', abs(ours[k].sum() / r.sum() - 1), TOL['clearsky_sum_rel'])
    r = ref['dni'].to_numpy()
    report('7 clearsky', 'boston', 'rows with dni 0 on one side, > 1 W/m2 on the other', int((((ours['dni'] == 0) & (r > 1)) | ((r == 0) & (ours['dni'] > 1))).sum()), 10, hard=False)
    expect = np.mean(ours['dhi'] + ours['dni'] * np.sin(np.radians(a[:, 4])).clip(0) * (a[:, 6] > 0.5))
    report('7 clearsky', 'boston', 'flat mean irradiance vs mean(dhi + dni sin alt) rel', abs(vals[:, 0].mean() / expect - 1), 0.02, hard=False)


LAYERS = {'sky': layer_sky, 'sun': layer_sun, 'bits': layer_bits, 'e2e': layer_e2e, 'usd': layer_usd, 'analytic': layer_analytic,
          'clearsky': layer_clearsky}


# ---------------------------------------------------------------- self-test of the parsers on synthetic data
def selftest():
    import tempfile
    tmp = Path(tempfile.mkdtemp())
    rng = np.random.default_rng(1)
    n_dir, N = 290, 7
    bits = rng.random((N, n_dir)) < 0.5
    words = np.zeros((N, (n_dir + 31) // 32), np.uint32)
    for i, d in zip(*np.nonzero(bits)):
        words[i, d // 32] |= np.uint32(1) << np.uint32(d % 32)
    np.save(tmp / 'bits.npy', words)
    assert np.array_equal(read_bits(tmp / 'bits.npy', n_dir), bits)
    (tmp / 'sky.txt').write_text(chr(10).join([
        '# mode ladybug mf 1', '# patch alt azi omega W', '0 5.9 0 0.0435 1 2 3', '# ground_weight 1',
        '# gendaymtx -s -A ascii rows (ground row first), %.3g', '0.1 0.2 0.3', '1e+03  2.5 3',
        '# gendaymtx -d -A ascii rows (ground row first), %.3g', '0 0 0', '4 5 6', '']))
    dif, dir_ = read_dump_sky(tmp / 'sky.txt')
    assert dif == ['0.1 0.2 0.3', '1e+03 2.5 3'] and dir_ == ['0 0 0', '4 5 6']
    (tmp / 'sun.txt').write_text(chr(10).join([
        '# hoy month day hour alt azi up alt_rad azi_rad up_rad (model: noaa; north 0)',
        '0.5 1 1 0.5 -10 30 0 -10.5 30.1 0', '12.5 1 1 12.5 24.03 190.57 1 23.9 190.4 1', '']))
    s = read_dump_sun(tmp / 'sun.txt')
    assert s['hoy'].tolist() == [0.5, 12.5] and s['up_noaa'].tolist() == [False, True] and s['azi_rad'][1] == 190.4
    assert s['alt_noaa'][1] == 24.03 and s['up_rad'].tolist() == [False, True]
    head = '#?RADIANCE\ngendaymtx -m 1 -s -O1 -A x.wea\nLATLONG= 42.35 -71.07\nNROWS=146\nNCOLS=1\nNCOMP=3\nFORMAT=ascii\n\n'
    rows = gendaymtx_A_rows(head + '0.1 0.1 0.1\n2.5 2.6 3\n')
    assert rows == ['0.1 0.1 0.1', '2.5 2.6 3']
    (tmp / 'w.wea').write_text('place x\nlatitude 1\nlongitude 2\ntime_zone 3\nsite_elevation 4\nweather_data_file_units 1\n1 1 0.500 10 20\n1 1 1.500 0 5\n')
    assert read_wea(tmp / 'w.wea').tolist() == [[10, 20], [0, 5]]
    (tmp / 'v.txt').write_text('1.5 2\n3 4\n')
    assert read_values(tmp / 'v.txt').tolist() == [[1.5, 2], [3, 4]]
    # ladybug direction table: 145 / 577 sky + mirrored ground; patch 0 points north at the Tregenza altitude
    d1, d2 = lb_dirs(1, 0), lb_dirs(2, 0)
    assert d1.shape == (290, 3) and d2.shape == (1154, 3)
    assert abs(math.degrees(math.asin(d1[0, 2])) - 5.967369) < 1e-5 and abs(d1[0, 0]) < 1e-12 and d1[0, 1] > 0
    assert np.allclose(d1[145:], -d1[:145]) and np.allclose(d1[144], [0, 0, 1])
    r = lb_dirs(1, 90)
    assert np.allclose(r[0], [-d1[0, 1], d1[0, 0], d1[0, 2]])     # +90 ccw: north -> west
    # W reconstruction + residual identity: ours and lb built from the same W and cos must give residual == 0
    sky = rng.random(145)
    W = lb_weights({'dir': sky * 0.4, 'dif': sky * 0.6, 'rho': 0.2}, 145)
    assert np.allclose(W[:145], sky) and np.allclose(W[145:], 0.2 * sky.mean())
    cos = np.clip(rng.standard_normal((N, 290)), 0, None)
    b_lb = rng.random((N, 290)) < 0.7
    b_ours = b_lb ^ (rng.random((N, 290)) < 0.05)
    ours = (b_ours * cos * W).sum(1)
    lb = (b_lb * cos * W).sum(1)
    assert (b_ours != b_lb).any() and np.abs(residual(ours, lb, b_ours, b_lb, cos, W)).max() < 1e-12
    assert np.abs(ours - lb).max() > 1e-3     # the plain difference is NOT zero, so the decomposition did the work
    # edge distance: ray through the centre of a unit right triangle vs ray grazing its edge
    tri = np.array([[[0, 0, 1], [1, 0, 1], [0, 1, 1]]], float)
    assert abs(edge_distance(np.array([0.25, 0.25, 0]), np.array([0, 0, 1.0]), tri) - 0.25) < 1e-12
    assert edge_distance(np.array([0.5, 0.0, 0]), np.array([0, 0, 1.0]), tri) < 1e-12
    assert edge_distance(np.array([0.25, 0.25, 2]), np.array([0, 0, 1.0]), tri) == math.inf   # plane behind the ray
    assert edge_distance(np.array([0.25, 0.25, 0]), np.array([0, 0, 1.0]), tris=np.zeros((0, 3, 3))) == math.inf
    # fixture helpers on the real fixtures when present
    if (FIX / 'box' / 'fixture.json').exists():
        assert fixture_tris('box', False).shape == (20, 3, 3) and fixture_tris('box', True).shape == (220, 3, 3)
        p, n = fixture_sensors('walls', False)
        assert p.shape == (480, 3) and np.allclose(n[240:360], [0, -1, 0])
    print('selftest OK')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('layer', choices=list(LAYERS) + ['all', 'selftest'])
    ap.add_argument('--exe')
    ap.add_argument('--epw', nargs='*', default=DEFAULT_EPWS)
    ap.add_argument('--scene', nargs='*', default=SCENES)
    a = ap.parse_args()
    if a.layer == 'selftest':
        return selftest()
    if not a.exe or not Path(a.exe).exists():
        sys.exit(f'error: --exe {a.exe!r} not found: build the CLI first (cmake --preset vs2022 && cmake --build --preset release)')
    missing = [s for s in a.epw if not (REF / 'epw' / s).is_dir()]
    if missing:
        sys.exit(f'error: no references for {missing}: run reference.py --epw ...')
    for name in (LAYERS if a.layer == 'all' else [a.layer]):
        try:
            LAYERS[name](Path(a.exe).resolve(), a.epw, a.scene)
        except (RuntimeError, FileNotFoundError, AssertionError, subprocess.CalledProcessError) as e:
            FAILS.append(f'{name}: {e}')
            print(f'{name}: {e}', file=sys.stderr)
    w = max(len(r[1]) for r in ROWS) if ROWS else 10
    for layer, case, metric, value, limit, status in ROWS:
        print(f'{layer:14s} {case:{w}s} {metric:48s} {value:12.4g} <= {limit:<8.3g} {status}')
    print(f'\n{len(FAILS)} hard failure(s)')
    for f in FAILS:
        print('  ' + f)
    return 1 if FAILS else 0


if __name__ == '__main__':
    sys.exit(main())
