"""Export rate contributions in energy, zenith and azimuth from the driver MAT."""
import argparse
import csv
from pathlib import Path
import numpy as np
from scipy.io import loadmat

p = argparse.ArgumentParser()
p.add_argument('mat_file')
a = p.parse_args()
source = Path(a.mat_file)
m = loadmat(source)
rates = [np.asarray(m[k]) for k in ('trg_rate_voltage_threshold','trg_rate_l1_noisy')]
axes = [np.asarray(m[k]).ravel() for k in ('energy_log10_eV','theta_values','phi_values')]
expected = tuple(len(x) for x in axes)
assert all(r.shape==expected and np.isfinite(r).all() for r in rates)
print('Signal-voltage threshold:', float(rates[0].sum()), 'day^-1')
print('Noisy L1:', float(rates[1].sum()), 'day^-1')
for axis, label in enumerate(('energy_log10_eV','zenith_lower_deg','azimuth_lower_deg')):
    dest = source.with_name(source.stem+'_'+label+'_rates.csv')
    other_axes = tuple(i for i in range(3) if i!=axis)
    columns = [r.sum(axis=other_axes) for r in rates]
    with dest.open('x', newline='', encoding='utf-8-sig') as stream:
        writer = csv.writer(stream)
        writer.writerow([label,'voltage_rate_per_day','l1_rate_per_day'])
        writer.writerows(zip(axes[axis], *columns))
    print(dest)
