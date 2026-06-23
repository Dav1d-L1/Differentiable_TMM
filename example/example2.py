'''
In this example we try to optimize the thin layer system to target Lab value.(normal incident)
'''

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import ASSETS

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from differentiable_tmm import TMM, Spectrum2XYZ, XYZ2Lab, spectrum_interpolation
import torch
from tqdm import trange


target=np.array([[40 , 5, -40]])
lower_bound=1.0 #nm
upper_bound=2000.0 #nm

epochs=800


#load metarial

glass_n_fn=spectrum_interpolation.constent_refrective(1.52)

temp=np.loadtxt(ASSETS / 'Si3N4.txt')
Si3N4_n_fn=spectrum_interpolation.interpolate_refrective(temp[:,0]*1e3, temp[:,1])

temp=np.loadtxt(ASSETS / 'SiO2.txt')
SiO2_n_fn=spectrum_interpolation.interpolate_refrective(temp[:,0]*1e3, temp[:,1]+1j*temp[:,2])

air_n_fn = spectrum_interpolation.constent_refrective(1)



# meta information

device=torch.device('cuda:0')
Spectrum2XYZ=Spectrum2XYZ('D65', 'CIE 1931 2 Degree Standard Observer', device, datatype='complex128', clip=True)
XYZ2Lab=XYZ2Lab('D65', 'CIE 1931 2 Degree Standard Observer', device, datatype='complex128')


###############################################################################

wv_array=Spectrum2XYZ.wavelength.cpu().numpy()
theta_array=np.arange(0,1)

wv_array_len=len(wv_array)
theta_array_len=len(theta_array)
theta_array=np.repeat(theta_array, wv_array_len)
wv_array=np.tile(wv_array, theta_array_len)


n0=torch.tensor(np.array([air_n_fn(wv_array),air_n_fn(wv_array)]),dtype=torch.complex128).T.to(device)
n=torch.tensor(np.array([glass_n_fn(wv_array),SiO2_n_fn(wv_array),Si3N4_n_fn(wv_array),SiO2_n_fn(wv_array),Si3N4_n_fn(wv_array),SiO2_n_fn(wv_array)]),dtype=torch.complex128).T.to(device) 
theta_array=torch.tensor(theta_array, dtype=torch.float64).to(device)
wv_array=torch.tensor(wv_array, dtype=torch.float64).to(device) 
c_list= ['i','c','c','c','c','c']

simulator_p=TMM.tmm(pol='p', datatype='complex128', theta_array=theta_array, wv_array=wv_array)
simulator_s=TMM.tmm(pol='s', datatype='complex128', theta_array=theta_array, wv_array=wv_array)

#optimization

struc_init=np.random.uniform(low=10, high=100, size=[5]).astype(np.float64)
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
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[epochs//2,epochs//4*3], gamma=0.2)
loss_fun = torch.nn.MSELoss()


t = trange(0,epochs, desc='', leave=True)
meta_data=torch.full((wv_array_len,1), 500000,dtype=Spectrum2XYZ.datatype_real).to(device)
target_lab=torch.tensor(target,dtype=Spectrum2XYZ.datatype_real).to(device)
Lab_result=[]
reflectances_hist=[]
for epoch in t:
    struc=lower_bound + (upper_bound - lower_bound) * torch.sigmoid(struc_logits)

    d=torch.cat([meta_data, struc.repeat(wv_array_len,1)],-1)

    result_p= simulator_p(n=n.type(torch.complex128), d=d.type(torch.float64), n0=n0, c_list=c_list)
    result_s= simulator_s(n=n.type(torch.complex128), d=d.type(torch.float64), n0=n0, c_list=c_list)
    reflectances=(result_p['R']+result_s['R'])/2
    reflectances=reflectances.reshape(theta_array_len, -1)
    XYZ=Spectrum2XYZ(reflectances)
    Lab=XYZ2Lab(XYZ)
    
    loss_lab=loss_fun(Lab,target_lab)
    
    t.set_description("loss_lab: {:.6f}".format(loss_lab.item()))
    t.refresh()
    
    Lab_result.append(Lab.detach().cpu().numpy())
    
    loss=loss_lab
    
    loss.backward()
    

    torch.nn.utils.clip_grad_norm_([struc_logits], max_norm=1)
    optimizer.step()
    optimizer.zero_grad()
    
    scheduler.step()
    
    if epoch % 50 ==0:
        reflectances_hist.append(reflectances.detach().cpu().numpy())


Lab_result=np.array(Lab_result)
final_struc=struc.detach().cpu().numpy()

#plot result
fig, axes = plt.subplots(3, 1, figsize=(15, 10))
# Plot L channel in the first subplot
axes[0].plot(Lab_result[:,0,0], label='Biel optimization')
axes[0].set_xlabel('epoch')
axes[0].set_ylabel('L')
axes[0].axhline(y=target[0,0], color='red', linestyle='--', label='target')
#axes[0].axhline(y=35.7651, color='m', linestyle='--', label='Macleod result')
axes[0].set_title('L Channel')
axes[0].legend()
# Plot a channel in the second subplot
axes[1].plot(Lab_result[:,0,1], label='Biel optimization')
axes[1].set_xlabel('epoch')
axes[1].set_ylabel('a')
axes[1].axhline(y=target[0,1], color='red', linestyle='--', label='target')
#axes[1].axhline(y=2.9157, color='m', linestyle='--', label='Macleod result')
axes[1].set_title('a Channel')
axes[1].legend()
# Plot b channel in the third subplot
axes[2].plot(Lab_result[:,0,2], label='Biel optimization')
axes[2].set_xlabel('epoch')
axes[2].set_ylabel('b')
axes[2].axhline(y=target[0,2], color='red', linestyle='--', label='target')
#axes[2].axhline(y=-46.6030, color='m', linestyle='--', label='Macleod result')
axes[2].set_title('b Channel')
axes[2].legend()
# Adjust layout to prevent overlap
plt.tight_layout()
fig.savefig(ASSETS / 'example2.png', dpi=200)
plt.close(fig)
print('Final layer thicknesses (nm):', final_struc)
