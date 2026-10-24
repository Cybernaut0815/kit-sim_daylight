# SPDX-License-Identifier: Apache-2.0 AND LicenseRef-Commons-Clause-1.0
"""Byte-compare goldens of the daylight CLI. Every case writes --out out.npy + out.npy.json, some also --timeseries
series.npy + series.npy.hoys.txt and --dump-bits bits.npy; `check` must reproduce every file byte for byte.

  uv run python -I golden.py record [--exe PATH] [--force]   # once, on main: -> _out/golden/<case>/
  uv run python -I golden.py check [--exe PATH]              # rerun into a temp dir, list every differing file, exit 1

Cases: the fixture scenes (fixtures.py runs first when its output is missing; usd_xform is converted by the Python
binding as in compare.py layer 6) in physical mode with all six quantities and in ladybug mode with what it accepts
(flags as compare.py uses them), a default run, a --from/--to range with --hours, a --time instant, the flag variants
and a plain .dmesh input. VARIANTS must give the bytes of their base case; one runs when `daylight --help` lists the
flags it needs, else it is reported as skipped. The slot-block .dmesh (header word 1, uint16 face % 3 after the
indices) needs no flag: older readers ignore the word and the trailing bytes."""
import argparse
import json
import shutil
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
FIX, GOLD = HERE / '_out' / 'fixtures', HERE / '_out' / 'golden'
BOSTON = HERE / '_dl' / 'epw' / 'USA_MA_Boston-City.WSO_TMY.epw'
SCENES = {'flat': 'obj', 'flat_tri': 'obj', 'walls': 'obj', 'box': 'obj', 'tilt30': 'obj', 'overhang': 'obj',
          'usd_xform': 'usd'}                                          # scene -> its fixture's mesh format
EPW = ['--epw', str(BOSTON)]
PHYS = EPW + ['--quantity', 'radiation,direct,diffuse,irradiance,illuminance,sunhours']
DUMPS = ['--timeseries', 'series.npy', '--dump-bits', 'bits.npy']


def ladybug(scene):
    """compare.py's ladybug runs: no illuminance, no self-occlusion except on the self-shading overhang."""
    return EPW + ['--mode', 'ladybug', '--offset', '0', '--quantity', 'radiation,direct,diffuse,irradiance,sunhours'] \
        + ([] if scene == 'overhang' else ['--no-self-occlude'])


# name -> (scene, mesh format obj | usd | dmesh | slots, flags)
# ponytail: every case fits one GPU chunk (<= 480 sensors); add a case past the 256 MiB chunk cap if chunking changes
CASES = {**{f'{s}_phys': (s, f, PHYS + (DUMPS if s == 'box' else [])) for s, f in SCENES.items()},
         **{f'{s}_lb': (s, f, ladybug(s) + (DUMPS if s == 'walls' else [])) for s, f in SCENES.items()},
         'box_default': ('box', 'obj', EPW),
         'box_range_hours': ('box', 'obj', PHYS + ['--from', '06-01T00:00', '--to', '06-15T00:00', '--step', '30',
                                                  '--hours', '09:00-17:00'] + DUMPS),
         'box_time': ('box', 'obj', PHYS + ['--time', '06-21T14:30']),
         'box_series_vis': ('box', 'obj', EPW + ['--quantity', 'illuminance,radiation', '--timeseries', 'series.npy']),
         'flat_sunhours_noepw': ('flat', 'obj', ['--quantity', 'sunhours', '--lat', '42.35', '--lon', '-71.07', '--from',
                                                 '03-21T00:00', '--to', '03-22T00:00', '--step', '10', '--dump-bits', 'bits.npy']),
         'box_cull_backfaces': ('box', 'obj', PHYS + ['--cull-backfaces']),
         'overhang_no_self_occlude': ('overhang', 'obj', PHYS + ['--no-self-occlude']),
         'overhang_sides_both': ('overhang', 'obj', PHYS + ['--sides', 'both']),
         'box_vertices': ('box', 'obj', PHYS + ['--on', 'vertices']),
         'box_patch_samples4': ('box', 'obj', PHYS + ['--patch-samples', '4']),
         'box_sky_density2': ('box', 'obj', PHYS + ['--sky-density', '2']),
         'walls_north90': ('walls', 'obj', PHYS + ['--north', '90']),
         'box_lb_sky_density2': ('box', 'obj', ladybug('box') + ['--sky-density', '2']),
         'box_lb_north30': ('box', 'obj', ladybug('box') + ['--north', '30']),
         'box_lb_vertices': ('box', 'obj', ladybug('box') + ['--on', 'vertices']),
         'box_dmesh': ('box', 'dmesh', PHYS + DUMPS)}
# (name, base case, mesh format (None = the base's), extra flags, flags `daylight --help` must list)
VARIANTS = [('box_dmesh_slots', 'box_dmesh', 'slots', [], []),
            ('box_dmesh_optics_opaque', 'box_dmesh', 'slots', ['--optics', '*=0.2;flat/1=0.3;box_a=0.25;box_b/2=0.1'],
             ['--optics']),
            ('box_dmesh_optics_transmission_off', 'box_dmesh', 'slots',
             ['--optics', 'box_a/1=0.1,0.5,porous;box_b=0,0.64,glass;flat/2=0.3,0.2', '--transmission', 'off'],
             ['--optics', '--transmission']),
            ('box_phys_bounces0', 'box_phys', None, ['--bounces', '0'], ['--bounces']),
            ('walls_lb_bounces0', 'walls_lb', None, ['--bounces', '0'], ['--bounces'])]


def write_dmesh(path, m, slots):
    """fixture.json mesh -> .dmesh (layout in src/daylight.h); slots: header word 1 + uint16 slot[n_faces] = face % 3."""
    p = np.asarray(m['vertices'], '<f8')
    c = np.array([len(f) for f in m['faces']], '<i4')
    i = np.array([k for f in m['faces'] for k in f], '<i4')
    tail = (np.arange(len(c)) % 3).astype('<u2').tobytes() if slots else b''
    path.write_bytes(struct.pack('<4sIQQQ', b'DLM1', int(slots), len(p), len(c), len(i)) + p.tobytes() + c.tobytes()
                     + i.tobytes() + tail)


def mesh_args(scene, fmt, tmp):
    """Mesh arguments of a case; converted / written inputs go to tmp/<fmt>/<scene>/."""
    d, out = FIX / scene, tmp / fmt / scene
    if fmt == 'obj':
        return [str(d / 'sensors.obj')] + (['--context', str(d / 'context.obj')] if (d / 'context.obj').exists() else [])
    if fmt == 'usd':
        if not out.exists():
            subprocess.run([sys.executable, '-I', '-m', 'daylight', str(d / 'usd_xform.usda'), '--sensors', '/World/Group',
                            '-o', str(out)], check=True, stdout=subprocess.DEVNULL)
        return [a for m in json.loads((out / 'meshes.json').read_text())
                for a in (['--context'] if m['role'] == 'context' else []) + [str(out / m['file'])]]
    out.mkdir(parents=True, exist_ok=True)
    js, args = json.loads((d / 'fixture.json').read_text()), []
    for key in 'sensors', 'context':
        for m in js[key]:
            write_dmesh(out / f"{m['name']}.dmesh", m, fmt == 'slots')
            args += (['--context'] if key == 'context' else []) + [str(out / f"{m['name']}.dmesh")]
    return args


def run(exe, args, d):
    """daylight <args> --out out.npy with d (created) as the working directory -> '' or the error."""
    d.mkdir(parents=True)
    r = subprocess.run([str(exe)] + args + ['--out', 'out.npy'], cwd=d, capture_output=True, text=True, errors='replace',
                       check=False)
    return f'daylight exit {r.returncode}: {r.stderr.strip()[-500:]}' if r.returncode else ''


def why(a, b):
    """Short description of how two differing files differ."""
    if a.suffix != '.npy':
        la, lb = a.read_text().splitlines(), b.read_text().splitlines()
        return f'text differs from line {next((k for k, (p, q) in enumerate(zip(la, lb)) if p != q), min(len(la), len(lb))) + 1}'
    x, y = np.load(a), np.load(b)
    if x.shape != y.shape or x.dtype != y.dtype:
        return f'{x.dtype} {x.shape} -> {y.dtype} {y.shape}'
    n = int((x != y).sum())
    return f'{n} of {x.size} values differ' + (f', max |diff| {np.abs(x - y).max():.3g}' if x.dtype.kind == 'f' else '')


def compare(gold, new):
    """-> one line per file that differs between a golden case directory and a new run (missing files included)."""
    if not gold.is_dir():
        return [f'no golden {gold.name}: record first']
    out = []
    for f in sorted({p.name for p in gold.iterdir()} | {p.name for p in new.iterdir()}):
        a, b = gold / f, new / f
        if not (a.exists() and b.exists()):
            out.append(f'{f}: only in {"the golden" if a.exists() else "the new run"}')
        elif a.read_bytes() != b.read_bytes():
            out.append(f'{f}: {why(a, b)}')
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('cmd', choices=['record', 'check'])
    ap.add_argument('--exe', default=str(HERE.parent / 'build' / 'RelWithDebInfo' / 'daylight.exe'))
    ap.add_argument('--force', action='store_true', help='record: replace existing goldens')
    a = ap.parse_args()
    exe = Path(a.exe).resolve()
    if not exe.exists():
        sys.exit(f'error: --exe {exe} not found: build the CLI first (cmake --build --preset release)')
    if not BOSTON.exists():
        sys.exit(f'error: {BOSTON} missing: run setup.py')
    if a.cmd == 'record' and GOLD.exists() and not a.force:
        sys.exit(f'error: {GOLD} exists; goldens are recorded once, on main (--force replaces them)')
    if not all((FIX / s / ('usd_xform.usda' if f == 'usd' else 'fixture.json')).exists() for s, f in SCENES.items()):
        subprocess.run([sys.executable, '-I', str(HERE / 'fixtures.py')], check=True)
    tmp, t0 = Path(tempfile.mkdtemp(prefix='golden_')), time.perf_counter()
    if a.cmd == 'record':
        shutil.rmtree(GOLD, ignore_errors=True)
        for name, (scene, fmt, args) in CASES.items():
            if err := run(exe, mesh_args(scene, fmt, tmp) + args, GOLD / name):
                shutil.rmtree(GOLD)
                sys.exit(f'{name}: {err}')
            print(f'recorded {name}')
        head = subprocess.run(['git', '-C', str(HERE), 'describe', '--always', '--dirty'], capture_output=True, text=True,
                              check=False)
        (GOLD / 'recorded.txt').write_text(f'{head.stdout.strip()} {exe}\n')
        shutil.rmtree(tmp)
        print(f'{len(CASES)} cases -> {GOLD}, {time.perf_counter() - t0:.0f} s')
        return 0
    if (GOLD / 'recorded.txt').exists():
        print(f'goldens of {(GOLD / "recorded.txt").read_text().strip()}')
    usage = subprocess.run([str(exe), '--help'], capture_output=True, text=True, errors='replace', check=False).stdout
    jobs = [(name, name, scene, fmt, args) for name, (scene, fmt, args) in CASES.items()]
    for name, base, fmt, extra, needs in VARIANTS:
        if lacking := [f for f in needs if f not in usage]:
            print(f'skip {name}: daylight --help lists no {" ".join(lacking)}')
        else:
            scene, bfmt, args = CASES[base]
            jobs.append((name, base, scene, fmt or bfmt, args + extra))
    bad = 0
    for name, base, scene, fmt, args in jobs:
        err = run(exe, mesh_args(scene, fmt, tmp) + args, tmp / name)
        diffs = [err] if err else compare(GOLD / base, tmp / name)
        bad += bool(diffs)
        print(f'{"DIFF" if diffs else "ok  "} {name}' + (f' (= {base})' if base != name else ''))
        for line in diffs:
            print(f'       {line}')
    print(f'{len(jobs)} runs, {bad} differing, {time.perf_counter() - t0:.0f} s')
    if bad:
        print(f'new outputs kept in {tmp}')
        return 1
    shutil.rmtree(tmp)
    return 0


if __name__ == '__main__':
    sys.exit(main())
