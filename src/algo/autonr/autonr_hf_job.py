import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback
import zipfile

BUNDLES = {
    'raw': ('autonr_colab_inputs.zip', 'f694804064a575b7d85057f8835c01f0c4ff43f91192ec4cc3b2e6d454ad91bc'),
    'l2': ('autonr_hf_l2_inputs.zip', 'e37550dfee440b1a2d85d614278f85bf2dba7fb7f118e1190bd1e56ca132d88e'),
}
PINS = ['numpy==2.2.6', 'scipy==1.15.3', 'scikit-learn==1.6.1',
        'pandas==2.2.3', 'matplotlib==3.10.3', 'h5py==3.13.0',
        'pyarrow==19.0.1', 'psutil==7.0.0', 'threadpoolctl==3.6.0', 'tqdm==4.67.1']


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def unpack(bundle, project, expected_hash):
    if sha(bundle) != expected_hash:
        raise ValueError('Bundle does not match the frozen hash for this geometry')
    with zipfile.ZipFile(bundle) as z:
        manifest = json.loads(z.read('transfer_manifest.json'))
        names = set(manifest['files']) | {'transfer_manifest.json'}
        if set(z.namelist()) != names or len(z.namelist()) != len(names):
            raise ValueError('Unexpected or duplicate archive entries')
        for name, expected in manifest['files'].items():
            if not (project / name).resolve().is_relative_to(project.resolve()):
                raise ValueError('Unsafe input path')
            if hashlib.sha256(z.read(name)).hexdigest() != expected:
                raise ValueError('Input checksum failed: ' + name)
        z.extractall(project)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--geometry', choices=list(BUNDLES), default='raw',
                        help='raw or row-wise L2-normalised raw; both 1,024 dimensions')
    parser.add_argument('--bundle', type=Path, help='Defaults to /input/<bundle for geometry>')
    parser.add_argument('--output', type=Path, default=Path('/output'))
    parser.add_argument('--run', action='store_true', help='Without this, perform checks only')
    parser.add_argument('--records', choices=['full', 'dedup'], default='full',
                        help='Keep all records or the verified representative subset')
    args = parser.parse_args()
    bundle_name, bundle_hash = BUNDLES[args.geometry]
    bundle = args.bundle or Path('/input') / bundle_name
    geometry_args = [] if args.geometry == 'raw' else ['--geometry', args.geometry]
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    run_id = f'hf_{args.geometry}_{args.records}_seed0_' + stamp
    destination = args.output / run_id
    destination.mkdir(parents=True, exist_ok=False)
    project = Path('/tmp') / run_id
    project.mkdir(exist_ok=False)
    env = os.environ.copy()
    env['MPLBACKEND'] = 'Agg'
    env['MPLCONFIGDIR'] = str(project / '.mpl')

    def call(command, name):
        print('Starting:', name, flush=True)
        with (destination / name).open('x') as log:
            process = subprocess.Popen(command, cwd=project, env=env,
                                       stdout=log, stderr=subprocess.STDOUT)
            while True:
                try:
                    code = process.wait(timeout=30)
                    break
                except subprocess.TimeoutExpired:
                    attempt = destination / 'experiment/attempts' / run_id
                    for filename in ('stdout.log', 'resources.csv'):
                        p = attempt / filename
                        if p.exists():
                            print(filename, p.read_text(errors='replace').splitlines()[-2:], flush=True)
            if code:
                print((destination / name).read_text(errors='replace')[-3000:], flush=True)
                raise RuntimeError(f'{name} failed, exit {code}; retained in output volume')

    try:
        manifest = unpack(bundle, project, bundle_hash)
        (destination / 'transfer_manifest.json').write_text(json.dumps(manifest, indent=2))
        shutil.copy2(__file__, destination / 'autonr_hf_job.py')
        for folder in ('src', 'tests'):
            shutil.copytree(project / folder, destination / folder)
        (destination / 'job.json').write_text(json.dumps({
            'job_id': os.environ.get('JOB_ID'), 'cpu_cores': os.environ.get('CPU_CORES'),
            'memory': os.environ.get('MEMORY'), 'python': sys.version,
            'requested_packages': PINS + ['clustpy==0.0.2'],
            'threads': {k: os.environ.get(k) for k in
                        ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')},
            'records': args.records, 'geometry': args.geometry,
            'run_requested': args.run, 'input_zip_sha256': bundle_hash}, indent=2))
        if shutil.which('git') is None:
            raise RuntimeError('Docker image must include git; use python:3.12-bookworm')
        call([sys.executable, '-m', 'pip', 'install'] + PINS, 'install.log')
        call([sys.executable, '-m', 'pip', 'install', '--no-deps', 'clustpy==0.0.2'], 'install_clustpy.log')
        call([sys.executable, '-m', 'pip', 'freeze'], 'packages.txt')
        call(['git', 'init', '-q'], 'git_init.log')
        call(['git', 'add', 'src', 'tests', 'transfer_manifest.json'], 'git_add.log')
        call(['git', '-c', 'user.name=AutoNR transfer', '-c', 'user.email=transfer@localhost',
              'commit', '-qm', 'Record raw AutoNR transfer snapshot'], 'git_commit.log')
        runner = project / 'src/algo/autonr/autonr_standalone.py'
        call([sys.executable, str(runner), 'preflight', '--records', args.records] + geometry_args,
             'preflight.log')
        call([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests',
              '-p', 'test_autonr_standalone.py', '-v'], 'tests.log')
        if args.run:
            output = destination / 'experiment'
            call([sys.executable, '-u', str(runner), 'run', '--root', str(output),
                  '--run-id', run_id, '--seed', '0', '--records', args.records] + geometry_args,
                 'driver.log')
            call([sys.executable, str(runner), 'validate', '--root', str(output),
                  '--run-id', run_id], 'validation.log')
        print('Finished. Retrieve persistent outputs:', destination, flush=True)
    except BaseException:
        (destination / 'job_failure.txt').write_text(traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
