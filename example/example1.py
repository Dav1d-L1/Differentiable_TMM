'''
This example we simualte colors for different observation angles coating structrue
'''

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import ASSETS

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from differentiable_tmm import TMM, Spectrum2XYZ, XYZ2Lab, XYZ2sRGB, spectrum_interpolation
import torch
import time

#load metarial


glass_n_fn=spectrum_interpolation.constent_refrective(1.52)

temp=np.loadtxt(ASSETS / 'Si3N4.txt')
Si3N4_n_fn=spectrum_interpolation.interpolate_refrective(temp[:,0]*1e3, temp[:,1])

temp=np.loadtxt(ASSETS / 'SiO2.txt')
SiO2_n_fn=spectrum_interpolation.interpolate_refrective(temp[:,0]*1e3, temp[:,1]+1j*temp[:,2])

air_n_fn = spectrum_interpolation.constent_refrective(1)



# meta information

device=torch.device('cpu')
Spectrum2XYZ=Spectrum2XYZ('D65', 'CIE 1931 2 Degree Standard Observer', device, datatype='complex128', clip=True)
XYZ2Lab=XYZ2Lab('D65', 'CIE 1931 2 Degree Standard Observer', device, datatype='complex128')
XYZ2sRGB=XYZ2sRGB(device, datatype='complex128')


wavelength_values=Spectrum2XYZ.wavelength.cpu().numpy()
theta_values=np.arange(0, 81)
thickness_values=np.linspace(16, 26, 81)

wv_array=wavelength_values.copy()
theta_array=theta_values.copy()
d=np.repeat(np.array([[500000, 20.8, 11.6, 70.1, 34.1, 141.1]]), len(thickness_values), axis=0)
d[:,1]=thickness_values


wv_array_len=len(wv_array)
theta_array_len=len(theta_array)
theta_array=np.repeat(theta_array, wv_array_len)
wv_array=np.tile(wv_array, theta_array_len)

wv_array_len=len(wv_array)
d_len=len(d)

theta_array=np.tile(theta_array, d_len)
wv_array=np.tile(wv_array, d_len)
d=np.repeat(d, wv_array_len, axis=0)

n0=torch.tensor(np.array([air_n_fn(wv_array),air_n_fn(wv_array)]),dtype=torch.complex128).T.to(device)
n=torch.tensor(np.array([glass_n_fn(wv_array),SiO2_n_fn(wv_array),Si3N4_n_fn(wv_array),SiO2_n_fn(wv_array),Si3N4_n_fn(wv_array),SiO2_n_fn(wv_array)]),dtype=torch.complex128).T.to(device) 
theta_array=torch.tensor(theta_array, dtype=torch.float64).to(device)
wv_array=torch.tensor(wv_array, dtype=torch.float64).to(device) 
d=torch.tensor(d, dtype=torch.float64).to(device) 
c_list= ['i','c','c','c','c','c']


# simulation
start=time.time()

simulator_p=TMM.tmm(pol='p', datatype='complex128', theta_array=theta_array, wv_array=wv_array)
simulator_s=TMM.tmm(pol='s', datatype='complex128', theta_array=theta_array, wv_array=wv_array)
result_p= simulator_p(n=n.type(torch.complex128), d=d.type(torch.float64), n0=n0, c_list=c_list)
result_s= simulator_s(n=n.type(torch.complex128), d=d.type(torch.float64),  n0=n0, c_list=c_list)


reflectances=(result_p['R']+result_s['R'])/2
reflectances=reflectances.reshape(d_len*theta_array_len, -1)

XYZ=Spectrum2XYZ(reflectances)

Lab=XYZ2Lab(XYZ)

RGB=XYZ2sRGB(XYZ)

RGB=RGB.reshape(d_len,theta_array_len,-1)

end=time.time()
print(end-start)

def _pixel_edges(values):
    values=np.asarray(values, dtype=float)
    if len(values) == 1:
        return np.array([values[0] - 0.5, values[0] + 0.5])

    centers=(values[:-1] + values[1:]) / 2
    first=values[0] - (centers[0] - values[0])
    last=values[-1] + (values[-1] - centers[-1])
    return np.concatenate(([first], centers, [last]))


def _plot_color_map(rgb, title, filename):
    image=np.real(rgb.detach().cpu().numpy())
    image=np.clip(image, 0, 1)

    x_edges=_pixel_edges(theta_values)
    y_edges=_pixel_edges(thickness_values)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.imshow(
        image,
        interpolation='nearest',
        origin='lower',
        aspect='auto',
        extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
    )
    ax.set_xlabel('Observation angle (degrees)')
    ax.set_ylabel('Film thickness (nm)')
    ax.set_title(title)
    ax.set_xticks(np.arange(theta_values[0], theta_values[-1] + 1, 10))
    ax.set_yticks(np.linspace(thickness_values[0], thickness_values[-1], 6))
    fig.tight_layout()
    fig.savefig(filename, dpi=200)
    plt.close(fig)


#plot result
_plot_color_map(RGB, 'Thin Film color Map (reflectance)', ASSETS / 'example1.png')



transmittance =(result_p['T']+result_s['T'])/2
transmittance =transmittance .reshape(d_len*theta_array_len, -1)

XYZ=Spectrum2XYZ(transmittance )

Lab=XYZ2Lab(XYZ)

RGB=XYZ2sRGB(XYZ)

RGB=RGB.reshape(d_len,theta_array_len,-1)

end=time.time()
print(end-start)

#plot result
_plot_color_map(RGB, 'Thin Film color Map (transmittance)', ASSETS / 'example1_1.png')













