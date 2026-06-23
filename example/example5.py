'''
In this example we try to visualize the thin layer system to target Lab value.
under different light condition
'''

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import ASSETS

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from differentiable_tmm import (
    TMM,
    Spectrum2XYZ,
    XYZ2Lab,
    XYZ2sRGB,
    illumination,
    spectrum_interpolation,
)
import torch


num_coating_layers = 5
default_struc = np.array([20.8, 11.6, 70.1, 34.1, 141.1], dtype=np.float64)
observer = 'CIE 1964 10 Degree Standard Observer'


#load metarial

glass_n_fn = spectrum_interpolation.constent_refrective(1.52)

temp = np.loadtxt(ASSETS / 'Si3N4.txt')
Si3N4_n_fn = spectrum_interpolation.interpolate_refrective(temp[:,0] * 1e3, temp[:,1])

temp = np.loadtxt(ASSETS / 'SiO2.txt')
SiO2_n_fn = spectrum_interpolation.interpolate_refrective(temp[:,0] * 1e3, temp[:,1] + 1j * temp[:,2])

air_n_fn = spectrum_interpolation.constent_refrective(1)


def build_layer_refractive_indices(wavelengths):
    layers = [glass_n_fn(wavelengths)]
    coating_n_fns = [SiO2_n_fn, Si3N4_n_fn]
    for i in range(num_coating_layers):
        layers.append(coating_n_fns[i % 2](wavelengths))
    return np.array(layers)


def build_c_list():
    return ['i'] + ['c'] * num_coating_layers


if len(default_struc) != num_coating_layers:
    raise ValueError(
        f'default_struc length ({len(default_struc)}) must match num_coating_layers ({num_coating_layers})'
    )


device = torch.device('cpu')
Spectrum2XYZ_ref = Spectrum2XYZ(
    'D65', observer, device, datatype='complex128', clip=True
)

wv_array = Spectrum2XYZ_ref.wavelength.cpu().numpy()
theta_array = np.arange(0, 1)

wv_array_len = len(wv_array)
theta_array_len = len(theta_array)
theta_array = np.repeat(theta_array, wv_array_len)
wv_array = np.tile(wv_array, theta_array_len)

n0 = torch.tensor(np.array([air_n_fn(wv_array), air_n_fn(wv_array)]), dtype=torch.complex128).T.to(device)
n = torch.tensor(build_layer_refractive_indices(wv_array).T, dtype=torch.complex128).to(device)
theta_array = torch.tensor(theta_array, dtype=torch.float64).to(device)
wv_array = torch.tensor(wv_array, dtype=torch.float64).to(device)
c_list = build_c_list()

simulator_p = TMM.tmm(pol='p', datatype='complex128', theta_array=theta_array, wv_array=wv_array)
simulator_s = TMM.tmm(pol='s', datatype='complex128', theta_array=theta_array, wv_array=wv_array)

struc = torch.tensor(default_struc, dtype=Spectrum2XYZ_ref.datatype_real).to(device)
meta_data = torch.full((wv_array_len, 1), 500000, dtype=Spectrum2XYZ_ref.datatype_real).to(device)
d = torch.cat([meta_data, struc.repeat(wv_array_len, 1)], -1)

result_p = simulator_p(n=n.type(torch.complex128), d=d.type(torch.float64), n0=n0, c_list=c_list)
result_s = simulator_s(n=n.type(torch.complex128), d=d.type(torch.float64), n0=n0, c_list=c_list)
reflectances = (result_p['R'] + result_s['R']) / 2
reflectances = reflectances.reshape(theta_array_len, -1)

Lab_list = []
RGB_list = []
light_source = list(illumination.keys())

for lightsource_name in light_source:
    Spectrum2XYZ_transform = Spectrum2XYZ(
        lightsource_name, observer, device, datatype='complex128', clip=True
    )
    XYZ2Lab_transform = XYZ2Lab(
        lightsource_name, observer, device, datatype='complex128'
    )
    XYZ2sRGB_transform = XYZ2sRGB(device, datatype='complex128')

    XYZ = Spectrum2XYZ_transform(reflectances)
    Lab = XYZ2Lab_transform(XYZ)
    RGB = XYZ2sRGB_transform(XYZ)

    Lab_list.append(Lab[0].detach().cpu().numpy())
    RGB_list.append(np.clip(np.real(RGB[0].detach().cpu().numpy()), 0, 1))

Lab_list = np.array(Lab_list)
RGB_list = np.array(RGB_list)

fig, ax = plt.subplots(figsize=(max(12, len(light_source) * 0.35), 2))

x = np.arange(len(light_source))
for xi, color in zip(x, RGB_list):
    ax.bar(xi, 1, color=color, edgecolor='none')

ax.set_xticks(x)
ax.set_xticklabels(light_source, rotation=90)
ax.set_yticks([])

for spine in ax.spines.values():
    spine.set_visible(False)

plt.tight_layout()
fig.savefig(ASSETS / 'example5.png', dpi=200, bbox_inches='tight')
plt.close(fig)

print('Saved color bar under', len(light_source), 'illuminants to', ASSETS / 'example5.png')
