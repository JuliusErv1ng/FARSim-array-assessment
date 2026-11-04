"""Check file names, array grids, and hard-coded native Touchstone conventions."""
import sys
from pathlib import Path
import numpy as np
from scipy.io import loadmat, whosmat

SPECS = {
 'balun13in20230612.s2p': 'MA', 'balun_before_ad.s2p': 'MA',
 'MatchingNetworkX.s2p': 'MA', 'MatchingNetworkZ.s2p': 'MA',
 'LNA-X.s2p': 'DB', 'cable+Connector.s2p': 'DB',
 'feb+amfitler+biast.s2p': 'DB', 'S_balun_AD.s1p': 'DB'}

def check(folder):
    grid = np.arange(30, 251)
    shapes = dict((name, shape) for name, shape, _ in whosmat(folder/'EFL_data.mat'))
    assert shapes.get('EFL_all') == (221, 3, 3, 181, 361), shapes
    mat = loadmat(folder/'EFL_data.mat', variable_names=['Theta', 'Phi', 'freq_ALL'])
    for key, expected in [('Theta', np.arange(181)), ('Phi', np.arange(361)), ('freq_ALL', grid/1000)]:
        actual = np.asarray(mat[key]).ravel()
        assert actual.shape == expected.shape and np.allclose(actual, expected), key
    z = np.loadtxt(folder/'Z_ant_3.2m.csv', delimiter=',', skiprows=1)
    assert z.shape == (221, 7) and np.allclose(z[:, 0], grid) and np.isfinite(z).all(), 'Z_ant'
    for file, mode in SPECS.items():
        headers, rows = [], []
        for line in (folder/file).read_text().splitlines():
            line = line.split('!')[0].strip()
            if not line:
                continue
            if line.startswith('#'):
                headers.append(line.upper().split())
            else:
                rows.append([float(v) for v in line.split()])
        assert len(headers) == 1 and headers[0][1:5] == ['HZ', 'S', mode, 'R'] and float(headers[0][5]) == 50, (file, headers)
        table = np.asarray(rows)
        width = 3 if file.endswith('.s1p') else 9
        assert table.ndim == 2 and table.shape[1] == width and np.isfinite(table).all(), file
        f = table[:,0]
        assert len(f) >= 4 and np.all(np.diff(f)>0) and f[0]<=30e6 and f[-1]>=250e6, file
        if width == 9 and mode == 'MA':
            assert np.all(table[:,3] > 0), (file, 'zero/negative S21')
    shapes = dict((name, shape) for name, shape, _ in whosmat(folder/'PG_ALL_jifen_perminute.mat'))
    assert shapes.get('PG_ALL_jifen') == (221,3,24,60), shapes
    shapes = dict((name, shape) for name, shape, _ in whosmat(folder/'Data_save_all_withAM.mat'))
    shape = shapes.get('Data_total_seperate', ())
    assert len(shape)==4 and shape[0]==1 and shape[1]>0 and shape[2:]==(4,1024), shapes
    print('PASS: response file structure, frequency/angle grids and S-parameter encodings.')
    print('A numerical waveform smoke test is also required; units/reference planes remain measurement metadata.')

if __name__ == '__main__':
    check(Path(sys.argv[1]).resolve())
