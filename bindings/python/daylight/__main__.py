# SPDX-License-Identifier: Apache-2.0 AND LicenseRef-Commons-Clause-1.0
"""python -m daylight STAGE [--sensors PREFIX]... [--context PREFIX]... [--time T] -o DIR [--write-usd PATH] [-- <core flags>]

Writes DIR/<stem>.dmesh per mesh (stem = prim path with [^A-Za-z0-9._-] -> '_', leading '_' stripped, made unique
with _2, _3 ...) and DIR/meshes.json = [{"file", "prim", "role": "sensor"|"context"}] in core order (sensors
first); with core flags after `--` also runs the core -> DIR/values.npy(.json), rows in meshes.json order, and
--write-usd PATH puts them back on the sensor prims as a sublayer file (primvars + displayColor, see usd.write_results)."""
import argparse
import json
import re
import sys
from pathlib import Path

from .core import run, to_dmesh
from .usd import meshes, split, write_results


def main(argv):
    core = []
    if '--' in argv:
        i = argv.index('--')
        argv, core = argv[:i], argv[i + 1:]
    ap = argparse.ArgumentParser(prog='python -m daylight', description=__doc__.split('\n')[2])
    ap.add_argument('stage')
    ap.add_argument('--sensors', action='append', default=[], metavar='PREFIX', help='prim path prefix of sensor meshes (repeatable)')
    ap.add_argument('--context', action='append', default=[], metavar='PREFIX', help='prim path prefix of blocking-only meshes (repeatable)')
    ap.add_argument('--time', type=float, help='USD time code (default: the stage start time)')
    ap.add_argument('-o', '--out', required=True, metavar='DIR')
    ap.add_argument('--write-usd', metavar='PATH', help='.usda/.usdc sublayer with the values as primvars + displayColor (needs core flags after --)')
    a = ap.parse_args(argv)
    if a.write_usd and not core:
        sys.exit('error: --write-usd needs core flags after -- (nothing to write without a run)')
    d = Path(a.out)
    d.mkdir(parents=True, exist_ok=True)
    sens, ctx = split(meshes(a.stage, a.time), a.sensors, a.context)
    if not sens:
        sys.exit('error: no sensor mesh (check --sensors against the prim paths; invisible/guide/inactive prims are dropped)')
    files, index, used = {'sensor': [], 'context': []}, [], set()
    for role, ms in ('sensor', sens), ('context', ctx):
        for name, *arrays in ms:
            stem = base = re.sub('[^A-Za-z0-9._-]', '_', name).lstrip('_')
            k = 1
            while stem.lower() in used:                    # NTFS file names are case-insensitive
                k += 1
                stem = f'{base}_{k}'
            used.add(stem.lower())
            to_dmesh(*arrays, d / f'{stem}.dmesh')
            files[role].append(d / f'{stem}.dmesh')
            index.append({'file': f'{stem}.dmesh', 'prim': name, 'role': role})
    (d / 'meshes.json').write_text(json.dumps(index, indent=1))
    msg = f'{len(sens)} sensor and {len(ctx)} context meshes -> {d}'
    if core:
        values, man = run(files['sensor'], files['context'], args=core, out=d / 'values.npy')
        msg += f' + {d / "values.npy"}'
        if a.write_usd:
            write_results(a.stage, values, man, [x['prim'] for x in index if x['role'] == 'sensor'], a.write_usd,
                          on='vertices' if '--on vertices' in ' '.join(core) else 'faces')
            msg += f' + {a.write_usd}'
    print(msg)


if __name__ == '__main__':
    main(sys.argv[1:])
