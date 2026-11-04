"""Relocatable, input-fingerprinted entry point for the maintained rate driver."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def resolve(value):
    p = Path(value).expanduser()
    return p.resolve() if p.is_absolute() else (ROOT / p).resolve()

def main():
    parser = argparse.ArgumentParser(description='FARSim event-rate launcher; Python >=3.11')
    parser.add_argument('config', nargs='?', default='configs/gp80_rate.json', help='JSON配置文件，相对本程序目录')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    cfg = json.loads(resolve(args.config).read_text(encoding='utf-8-sig'))
    if set(cfg) - {'description', 'response_dir', 'output_dir', 'driver_args'}:
        raise ValueError('Unknown top-level configuration keys')
    response = resolve(cfg['response_dir'])
    output = resolve(cfg['output_dir'])
    subprocess.run([sys.executable, str(ROOT/'validate_response.py'), str(response)], check=True)
    options = dict(cfg['driver_args'])
    if {'output-dir', 'resume'} & set(options):
        raise ValueError('Use top-level output_dir and launcher --resume')
    if not (0 <= options['theta-start'] <= options['theta-stop'] <= 88):
        raise ValueError('theta must be 0..88; each integer labels a one-degree integration bin')
    if not (0 <= options['phi-start'] <= options['phi-stop'] < 360):
        raise ValueError('phi must avoid duplicated 360-degree endpoint')
    step = options['phi-step']
    if step <= 0 or (options['phi-stop']-options['phi-start']) % step or options['phi-stop']+step > 360:
        raise ValueError('Invalid phi integration cells')
    if options['required-antennas'] < 1 or options['noise-trials'] < 1:
        raise ValueError('Invalid trigger/noise settings')
    for key in ('layout-path', 'l1-station-jitter-file'):
        if key in options:
            options[key] = str(resolve(options[key]))
    env = os.environ.copy()
    # Explicit settings prevent inherited project-specific environment variables.
    env.update(FARSIM_ROOT=str(ROOT), FARSIM_DATA_ROOT=str(ROOT/'data'),
        FARSIM_NATIVE_RFCHAIN_FORWARD=str(ROOT/'native_rfchain/batch_forward_rfchain_save_mat.py'),
        FARSIM_NATIVE_RFCHAIN_PARAMS=str(response),
        FARSIM_KERNEL_TEMPLATE_MODEL=str(ROOT/'farsim_waveform_py/models/kernel_trace_template_model_raw_full2500_1999x05ns.npz'),
        FARSIM_RING_FOCUSED_KERNEL_TEMPLATE_MODEL=str(ROOT/'farsim_waveform_py/models/kernel_trace_template_model_ring_focused_full2500_1999x05ns.npz'),
        FARSIM_RING_METRICS=str(ROOT/'data/pulse_width_metrics_raw.csv'),
        MPLBACKEND='Agg', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    env['PYTHONPATH'] = os.pathsep.join(str(ROOT/p) for p in
        ('farsim_py/src', 'farsim_waveform_py/src'))
    roots = [response, ROOT/'data', ROOT/'farsim_waveform_py/models', ROOT/'native_rfchain',
             ROOT/'drivers', ROOT/'farsim_py/src', ROOT/'farsim_waveform_py/src']
    inputs = sorted({p.resolve() for r in roots for p in r.rglob('*')
        if p.is_file() and not p.name.startswith('._') and '__pycache__' not in p.parts})
    inputs.extend(ROOT/p for p in ('run_rate.py', 'validate_response.py', 'requirements.txt',
                                   'farsim_py/pyproject.toml', 'farsim_waveform_py/pyproject.toml'))
    if 'l1-station-jitter-file' in options:
        p = Path(options['l1-station-jitter-file'])
        if not p.is_file():
            raise ValueError('Supply an existing station jitter file or omit this option')
        inputs.append(p)
    layout = Path(options['layout-path'])
    if layout not in inputs:
        inputs.append(layout)
    record = {'config': cfg, 'resolved_options': options,
              'inputs': {str(p): digest(p) for p in inputs}}
    command = [sys.executable, str(ROOT/'drivers/adaptive_rate.py'), '--output-dir', str(output)]
    for key, value in options.items():
        if value is True:
            command.append('--'+key)
        elif value is not False and value is not None:
            command += ['--'+key, str(value)]
    print('Configuration and input hashes checked.', flush=True)
    if args.check_only:
        print(' '.join(command))
        return
    stamp = output/'run_inputs.json'
    if args.resume:
        if not stamp.exists() or json.loads(stamp.read_text()) != record:
            raise ValueError('Resume rejected: configuration/input files changed or stamp absent')
        command.append('--resume')
    else:
        if output.exists() and any(output.iterdir()):
            raise FileExistsError('Output is not empty; select a new output_dir or use --resume')
        output.mkdir(parents=True, exist_ok=True)
        stamp.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Running:', command, flush=True)
    with (output/'console.log').open('a', encoding='utf-8') as log:
        process = subprocess.Popen(command, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in process.stdout:
            print(line, end='', flush=True)
            log.write(line); log.flush()
        result = process.wait()
    if result:
        raise SystemExit(result)

if __name__ == '__main__':
    main()
