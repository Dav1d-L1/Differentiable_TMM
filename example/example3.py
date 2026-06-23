'''
In this example we try to optimize the thin layer system to target Lab value for different observation angle
And analysis optimization procedure and result
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
from tqdm import trange

lab_target=np.array([38, 5, -40])
theta_values_opt=np.arange(0, 31, 10)
target=np.array([lab_target]*len(theta_values_opt))
lower_bound=1.0
upper_bound=2000.0

epochs=2000


#load metarial

glass_n_fn=spectrum_interpolation.constent_refrective(1.52)

temp=np.loadtxt(ASSETS / 'Si3N4.txt')
Si3N4_n_fn=spectrum_interpolation.interpolate_refrective(temp[:,0]*1e3, temp[:,1])

temp=np.loadtxt(ASSETS / 'SiO2.txt')
SiO2_n_fn=spectrum_interpolation.interpolate_refrective(temp[:,0]*1e3, temp[:,1]+1j*temp[:,2])

air_n_fn = spectrum_interpolation.constent_refrective(1)

num_coating_layers = 11


def build_layer_refractive_indices(wavelengths):
    layers = [glass_n_fn(wavelengths)]
    coating_n_fns = [SiO2_n_fn, Si3N4_n_fn]
    for i in range(num_coating_layers):
        layers.append(coating_n_fns[i % 2](wavelengths))
    return np.array(layers)


def build_c_list():
    return ['i'] + ['c'] * num_coating_layers




# meta information

device=torch.device('cpu')
Spectrum2XYZ=Spectrum2XYZ('D65', 'CIE 1931 2 Degree Standard Observer', device, datatype='complex128', clip=True)
XYZ2Lab=XYZ2Lab('D65', 'CIE 1931 2 Degree Standard Observer', device, datatype='complex128')
XYZ2sRGB=XYZ2sRGB(device, datatype='complex128')


###############################################################################


wv_array=Spectrum2XYZ.wavelength.cpu().numpy()
theta_array=theta_values_opt.copy()

wv_array_len=len(wv_array)
theta_array_len=len(theta_array)
theta_array=np.repeat(theta_array, wv_array_len)
wv_array=np.tile(wv_array, theta_array_len)


n0=torch.tensor(np.array([air_n_fn(wv_array),air_n_fn(wv_array)]),dtype=torch.complex128).T.to(device)
n=torch.tensor(build_layer_refractive_indices(wv_array).T, dtype=torch.complex128).to(device)
theta_array=torch.tensor(theta_array, dtype=torch.float64).to(device)
wv_array=torch.tensor(wv_array, dtype=torch.float64).to(device) 
c_list=build_c_list()

simulator_p=TMM.tmm(pol='p', datatype='complex128', theta_array=theta_array, wv_array=wv_array)
simulator_s=TMM.tmm(pol='s', datatype='complex128', theta_array=theta_array, wv_array=wv_array)

#optimization

struc_init=np.random.uniform(low=10, high=100, size=[num_coating_layers]).astype(np.float64)
struc_fraction=np.clip(
    (struc_init - lower_bound) / (upper_bound - lower_bound),
    1e-6,
    1 - 1e-6,
)
struc_logits=torch.tensor(
    np.log(struc_fraction / (1 - struc_fraction)),
    dtype=Spectrum2XYZ.datatype_real,
).to(device)
struc_logits.requires_grad=True
optimizer = torch.optim.Adam([struc_logits], lr=0.01, betas=(0.9, 0.999))
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[epochs//2], gamma=0.2)
loss_fun = torch.nn.MSELoss()


t = trange(0,epochs, desc='', leave=True)
meta_data=torch.full((wv_array_len*theta_array_len,1), 500000,dtype=Spectrum2XYZ.datatype_real).to(device)
target_lab=torch.tensor(target,dtype=Spectrum2XYZ.datatype_real).to(device)
for epoch in t:
    struc=lower_bound + (upper_bound - lower_bound) * torch.sigmoid(struc_logits)

    d=torch.cat([meta_data, struc.repeat(wv_array_len*theta_array_len,1)],-1)

    result_p= simulator_p(n=n.type(torch.complex128), d=d.type(torch.float64),  n0=n0, c_list=c_list)
    result_s= simulator_s(n=n.type(torch.complex128), d=d.type(torch.float64),  n0=n0, c_list=c_list)
    reflectances=(result_p['R']+result_s['R'])/2
    reflectances=reflectances.reshape(theta_array_len, -1)
    XYZ=Spectrum2XYZ(reflectances)
    Lab=XYZ2Lab(XYZ)

    loss_lab=loss_fun(Lab,target_lab)

    t.set_description("loss_lab: {:.6f}".format(loss_lab.item()))
    t.refresh()

    loss=loss_lab

    loss.backward()

    torch.nn.utils.clip_grad_norm_([struc_logits], max_norm=1)
    optimizer.step()
    optimizer.zero_grad()

    scheduler.step()


final_struc=struc.detach().cpu().numpy()

###############################################################################

wv_array=Spectrum2XYZ.wavelength.cpu().numpy()
theta_values=np.arange(0,theta_values_opt.max())
d=np.array([[500000]+list(final_struc)])




wv_array_len=len(wv_array)
theta_array_len=len(theta_values)
theta_array=np.repeat(theta_values, wv_array_len)
wv_array=np.tile(wv_array, theta_array_len)

wv_array_len=len(wv_array)
d_len=len(d)

theta_array=np.tile(theta_array, d_len)
wv_array=np.tile(wv_array, d_len)
d=np.repeat(d, wv_array_len, axis=0)


n0=torch.tensor(np.array([air_n_fn(wv_array),air_n_fn(wv_array)]),dtype=torch.complex128).T.to(device)
n=torch.tensor(build_layer_refractive_indices(wv_array).T, dtype=torch.complex128).to(device)
theta_array=torch.tensor(theta_array, dtype=torch.float64).to(device)
wv_array=torch.tensor(wv_array, dtype=torch.float64).to(device) 
d=torch.tensor(d, dtype=torch.float64).to(device) 

c_list=build_c_list()


simulator_p=TMM.tmm(pol='p', datatype='complex128', theta_array=theta_array, wv_array=wv_array)
simulator_s=TMM.tmm(pol='s', datatype='complex128', theta_array=theta_array, wv_array=wv_array)

result_p= simulator_p(n=n.type(torch.complex128), d=d.type(torch.float64), n0=n0, c_list=c_list)
result_s= simulator_s(n=n.type(torch.complex128), d=d.type(torch.float64), n0=n0, c_list=c_list)
reflectances=(result_p['R']+result_s['R'])/2
reflectances=reflectances.reshape(d_len*theta_array_len, -1)
XYZ=Spectrum2XYZ(reflectances)
Lab=XYZ2Lab(XYZ)
RGB=XYZ2sRGB(XYZ)
RGB=RGB.reshape(d_len,theta_array_len,-1)

Lab_np=Lab.detach().cpu().numpy()

fig, axes = plt.subplots(3, 1, figsize=(15, 10))
axes[0].plot(Lab_np[:,0], label='new')
axes[0].set_xlabel('observer angle (degree)')
axes[0].set_ylabel('L')
axes[0].axhline(y=target[0,0], color='red', linestyle='--', label='target')
axes[0].set_title('L Channel')
axes[0].legend()
axes[1].plot(Lab_np[:,1], label='new')
axes[1].set_xlabel('observer angle (degree)')
axes[1].set_ylabel('a')
axes[1].axhline(y=target[0,1], color='red', linestyle='--', label='target')
axes[1].set_title('a Channel')
axes[1].legend()
axes[2].plot(Lab_np[:,2], label='new')
axes[2].set_xlabel('observer angle (degree)')
axes[2].set_ylabel('b')
axes[2].axhline(y=target[0,2], color='red', linestyle='--', label='target')
axes[2].set_title('b Channel')
axes[2].legend()
plt.tight_layout()
fig.savefig(ASSETS / 'example3.png', dpi=200)
plt.close(fig)


color_bar_data=np.clip(RGB.detach().cpu().numpy(), 0, 1)
fig, ax = plt.subplots(figsize=(10, 2))
ax.imshow(color_bar_data, aspect='auto')
ax.set_title('Color Visualization (new design) (0-80 Degrees)')
ax.set_xlabel('Observation Angle (Degrees)')
ax.set_yticks([])
tick_positions = np.linspace(0, theta_values_opt.max()-1, num=9)
tick_labels = [f"{int(pos/79 * 80)}" for pos in tick_positions]
ax.set_xticks(tick_positions)
ax.set_xticklabels(tick_labels)
plt.tight_layout()
fig.savefig(ASSETS / 'example3_1.png', dpi=200)
plt.close(fig)

print('Final layer thicknesses (nm):', final_struc)
