import torch
import numpy as np
from .illumination_store import illumination,Get_illumination
from .observer_store import Observer
from .chromaticity_store import chromaticity

class Spectrum2XYZ:
    def __init__(self, lightsource, observer, device, datatype, clip=True):
        '''
        if clip=True, we use ASTM E308 Practical Working Range (360nm - 780nm)
        '''
        
        if lightsource not in illumination:
            raise ValueError("Invalid light source. User input: {}. \n Valid light source: {}".format(lightsource, list(illumination.keys()))) 
        
        if observer not in Observer:
            raise ValueError("Invalid observer. User input: {}. \n Valid observer: {}".format(observer, list(Observer.keys()))) 
        
        if datatype == 'complex64':
            self.datatype_complex=torch.complex64
            self.datatype_real=torch.float32
        elif datatype == 'complex128':
            self.datatype_complex=torch.complex128
            self.datatype_real=torch.float64
        else:
            raise ValueError("datatype must be one of ['complex64', 'complex128']. User input: {}".format(datatype)) 
        
        if clip:
            index=(Observer[observer]['wavelength']>=360) & (Observer[observer]['wavelength']<=780)
            self.wavelength=torch.tensor(Observer[observer]['wavelength'][index],dtype=self.datatype_real).to(device)
            self.weight=torch.tensor(Observer[observer]['weight'][index],dtype=self.datatype_real).to(device)
            self.lightsource=torch.tensor(Get_illumination(lightsource,Observer[observer]['wavelength'][index]),dtype=self.datatype_real).to(device)
    
        else:
            self.wavelength=torch.tensor(Observer[observer]['wavelength'],dtype=self.datatype_real).to(device)
            self.weight=torch.tensor(Observer[observer]['weight'],dtype=self.datatype_real).to(device)
            self.lightsource=torch.tensor(Get_illumination(lightsource,Observer[observer]['wavelength']),dtype=self.datatype_real).to(device)
        
        self.device=device
        
        self.k=100/(self.weight[:, 1] * self.lightsource).sum()
        
        
        
    def __call__(self, reflectances):
        '''
        reflectances should be a n*m tensor, where first dimension is batch size and 
        second dimension is refrectances per wavelength
        '''
        
        XYZ = self.k * (reflectances * self.lightsource) @ self.weight
        
        return XYZ
        
        


class XYZ2Lab:
    def __init__(self, lightsource, observer, device, datatype):
        
        if datatype == 'complex64':
            self.datatype_complex=torch.complex64
            self.datatype_real=torch.float32
        elif datatype == 'complex128':
            self.datatype_complex=torch.complex128
            self.datatype_real=torch.float64
        else:
            raise ValueError("datatype must be one of ['complex64', 'complex128']. User input: {}".format(datatype)) 
        
        if observer not in chromaticity:
            raise ValueError("Invalid observer. User input: {}. \n Valid observer: {}".format(observer, list(chromaticity.keys()))) 
        else:
            temp=chromaticity[observer]
        
        if lightsource not in temp:
            raise ValueError("Invalid light source. User input: {}. \n Valid light source: {}".format(lightsource, list(temp.keys()))) 
        else:
            temp=temp[lightsource]
        
        XYZ = [temp[0]* 1/temp[1], 1, (1 - temp[0] - temp[1]) * 1/temp[1]]
        
        self.reference=np.array(XYZ)*100
        self.reference=torch.tensor(self.reference,dtype=self.datatype_real).to(device)
        
        self.device=device
        
        self.delta=torch.tensor(6/29,dtype=self.datatype_real).to(device)
        self.delta_2=self.delta**2
        self.delta_3=self.delta**3
        self.delta_4=torch.tensor(4/29,dtype=self.datatype_real).to(device)
        
        self.weight=torch.tensor(np.array([[0,116,0],[500,-500,0],[0,200,-200]]),dtype=self.datatype_real).to(self.device)
        self.bias=torch.tensor(np.array([[-16],[0],[0]]),dtype=self.datatype_real).to(self.device)
        
        
    def _function(self, t):
        
       t=torch.where(t>self.delta_3, t**(1/3), t/(3*self.delta_2)+self.delta_4)
        
       return t
    
    def __call__(self, XYZ):
        
        XYZ_n=XYZ/self.reference
        
        XYZ_n=self._function(XYZ_n)
        
        Lab=(self.weight @ XYZ_n[:,:,torch.newaxis]+self.bias).squeeze(-1)
        
        return Lab
        
        
class XYZ2sRGB:
    def __init__(self,device, datatype):
        
        if datatype == 'complex64':
            self.datatype_complex=torch.complex64
            self.datatype_real=torch.float32
        elif datatype == 'complex128':
            self.datatype_complex=torch.complex128
            self.datatype_real=torch.float64
        else:
            raise ValueError("datatype must be one of ['complex64', 'complex128']. User input: {}".format(datatype)) 
        
        self.device=device
        
        self.matrix=torch.tensor(np.array([[3.2406255, -1.537208, -0.4986286],[-0.9689307, 1.8757561, 0.0415175],[0.0557101, -0.2040211, 1.0569959]]),dtype=self.datatype_real).to(self.device)   
        
     
    def _function(self, t):
        
       t=torch.where(t>0.0031308, 1.055*t**(1/2.4)-0.055, 12.92*t)
        
       return t
        
    def __call__(self, XYZ):
        
        XYZ_n=XYZ/100

        
        rgb=(self.matrix @ XYZ_n[:,:,torch.newaxis]).squeeze(-1)
        
        rgb=self._function(rgb)
        
        return rgb#torch.clip(rgb,min=0,max=1)
        


def DE94(Lab, Lab_target):
    '''
    apple color different measurement with K1=0.048 and K2=0.014 (Textiles)
    '''
    K1=0.048
    K2=0.014
    kL=2
    kC=1
    kH=1
    
    dL=Lab[:,0]-Lab_target[:,0]
    C1=torch.linalg.vector_norm(Lab[:,1:3],dim=1)
    C2=torch.linalg.vector_norm(Lab_target[:,1:3],dim=1)
    dC=C1-C2
    dH=(Lab_target[:,1]*Lab[:,2]-Lab_target[:,2]*Lab[:,1])/np.sqrt(0.5*(C1*C2+Lab_target[:,1]*Lab[:,1]+Lab_target[:,2]*Lab[:,2]))
    
    SL=1
    SC=1+K1*C2
    SH=1+K2*C2
    
    ed94=np.sqrt((dL/(kL*SL))**2+(dC/(kC*SC))**2+(dH/(kH*SH))**2)

    return ed94






















