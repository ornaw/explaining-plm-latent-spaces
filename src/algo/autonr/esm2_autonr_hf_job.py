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

BUNDLE_NAME = 'esm2_autonr_hf_inputs.zip'
BUNDLE_SHA256 = 'eca48f8a9653ad093d3903296523b418ef5b1622ebe7832295bdc639abf30642'
PINS = ['numpy==2.5.2', 'scipy==1.18.0', 'scikit-learn==1.9.0', 'pandas==3.0.5',
        'h5py==3.16.0', 'pyarrow==25.0.1', 'psutil==7.2.2', 'threadpoolctl==3.6.0',
        'joblib==1.5.3', 'matplotlib==3.11.1', 'tqdm==4.70.0']
RUNNER = 'src/algo/autonr/esm2_autonr_standalone.py'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def unpack(bundle, project):
    if sha(bundle) != BUNDLE_SHA256:
        raise ValueError('Bundle does not match the frozen hash')
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
    parser = argparse.ArgumentParser(description=(__doc__ or "").partition("\n")[0])
    parser.add_argument('--bundle', type=Path, default=Path('/input') / BUNDLE_NAME)
    parser.add_argument('--output', type=Path, default=Path('/output'))
    parser.add_argument('--run', action='store_true', help='Without this, perform checks only')
    parser.add_argument('--workers', type=int, default=10)
    parser.add_argument('--threads', type=int, default=3)
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    run_id = 'hf_esm2_3b_pca90_full_seeds00to09_' + stamp
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
                    code = process.wait(timeout=60)
                    break
                except subprocess.TimeoutExpired:
                    lines = (destination / name).read_text(errors='replace').splitlines()
                    print(name, lines[-2:], flush=True)
            if code:
                print((destination / name).read_text(errors='replace')[-3000:], flush=True)
                raise RuntimeError(f'{name} failed, exit {code}; retained in output volume')

    try:
        manifest = unpack(args.bundle, project)
        (destination / 'transfer_manifest.json').write_text(json.dumps(manifest, indent=2))
        shutil.copy2(__file__, destination / 'esm2_autonr_hf_job.py')
        for folder in ('src', 'tests'):
            shutil.copytree(project / folder, destination / folder)
        (destination / 'job.json').write_text(json.dumps({
            'job_id': os.environ.get('JOB_ID'), 'cpu_cores': os.environ.get('CPU_CORES'),
            'memory': os.environ.get('MEMORY'), 'python': sys.version,
            'requested_packages': PINS + ['clustpy==0.0.2'],
            'workers': args.workers, 'threads_per_worker': args.threads,
            'run_requested': args.run, 'input_zip_sha256': BUNDLE_SHA256}, indent=2))
        if shutil.which('git') is None:
            raise RuntimeError('Docker image must include git; use python:3.12-bookworm')
        call([sys.executable, '-m', 'pip', 'install'] + PINS, 'install.log')
        call([sys.executable, '-m', 'pip', 'install', '--no-deps', 'clustpy==0.0.2'], 'install_clustpy.log')
        call([sys.executable, '-m', 'pip', 'freeze'], 'packages.txt')
        (project / '.gitignore').write_text('__pycache__/\n.mpl/\n')
        call(['git', 'init', '-q'], 'git_init.log')
        call(['git', 'add', '.'], 'git_add.log')
        call(['git', '-c', 'user.name=AutoNR transfer', '-c', 'user.email=transfer@localhost',
              'commit', '-qm', 'Record ESM-2 AutoNR transfer snapshot'], 'git_commit.log')
        call([sys.executable, RUNNER, 'preflight'], 'preflight.log')
        call([sys.executable, '-m', 'unittest', 'tests.test_esm2_autonr_standalone', '-v'], 'tests.log')
        if args.run:
            output = destination / 'experiment'
            call([sys.executable, '-u', RUNNER, 'run', '--root', str(output),
                  '--workers', str(args.workers), '--threads', str(args.threads)], 'driver.log')
            call([sys.executable, RUNNER, 'validate', '--root', str(output)], 'validation.log')
        print('Finished. Retrieve persistent outputs:', destination, flush=True)
    except BaseException:
        (destination / 'job_failure.txt').write_text(traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
