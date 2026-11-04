"""Normalize a conventional single-line Touchstone v1 file for native RFChain."""
import argparse
from pathlib import Path
import numpy as np
from validate_response import SPECS

def main():
    p = argparse.ArgumentParser(description='Convert Hz/kHz/MHz/GHz and RI/MA/DB to the required stage format')
    p.add_argument('source'); p.add_argument('destination')
    a = p.parse_args()
    dest = Path(a.destination)
    if dest.exists():
        raise FileExistsError('Refusing to overwrite an existing response')
    if dest.name not in SPECS:
        raise ValueError('Destination basename must be one of the eight required S-parameter files')
    rows, header = [], None
    for line in Path(a.source).read_text().splitlines():
        line = line.split('!')[0].strip()
        if not line: continue
        if line.startswith('['):
            raise ValueError('Touchstone v2 is not supported; export v1')
        if line.startswith('#'):
            header = line.upper().split()[1:]
        else:
            rows.append([float(x) for x in line.split()])
    if header is None or len(header)!=5 or header[1]!='S' or header[3]!='R' or float(header[4])!=50:
        raise ValueError('An explicit S-parameter header with R=50 is required')
    table = np.asarray(rows, dtype=float)
    expected = 3 if dest.suffix.lower()=='.s1p' else 9
    if table.ndim!=2 or table.shape[1]!=expected or not np.isfinite(table).all():
        raise ValueError('Require one full frequency record per line; .s1p=3 columns, .s2p=9 columns')
    table[:,0] *= {'HZ':1, 'KHZ':1e3, 'MHZ':1e6, 'GHZ':1e9}[header[0]]
    pair = table[:,1:].reshape(len(table), -1, 2)
    mode = header[2]
    if mode=='RI':
        values = pair[:,:,0]+1j*pair[:,:,1]
    elif mode in {'MA','DB'}:
        magnitude = pair[:,:,0] if mode=='MA' else 10**(pair[:,:,0]/20)
        values = magnitude*np.exp(1j*np.deg2rad(pair[:,:,1]))
    else:
        raise ValueError('Require RI, MA or DB')
    amplitude = np.abs(values)
    if SPECS[dest.name]=='DB':
        if np.any(amplitude<=0): raise ValueError('Zero magnitude cannot be represented as finite DB')
        amplitude = 20*np.log10(amplitude)
    result = np.column_stack((table[:,0], np.stack((amplitude,np.rad2deg(np.angle(values))),axis=-1).reshape(len(table),-1)))
    if not np.all(np.diff(result[:,0])>0): raise ValueError('Frequencies must be strictly ascending')
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open('x', encoding='utf-8') as stream:
        stream.write(f'# Hz S {SPECS[dest.name]} R 50\n')
        np.savetxt(stream, result, fmt='%.14g')
    print(dest)

if __name__ == '__main__': main()
