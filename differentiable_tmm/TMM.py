import torch
import numpy as np
from .datatype_restriction import enforce_torch_dtype
import sys
import warnings
from functools import reduce

class tmm:
    def __init__(self, pol, datatype, theta_array, wv_array):
        '''
        Transfer matrix method to calculate the transmission and reflection of light
        through a multiple thin film system
        
        pol: Polarization of incoming light. it should be one of ['s','p']
        datatype: It should be one of ['complex64', 'complex128']
        theta_array: Angle of incident light (of unit degree)
        wv_array: Wave length of incident light (in vaccum)
        
        All length unit should be nm
            
        '''
        
        if pol not in ['s','p']:
            raise ValueError("polarization must be one of ['s','p']. User input: {}".format(pol)) 
        else:
            self.pol=pol
            
        if datatype == 'complex64':
            self.datatype_complex=torch.complex64
            self.datatype_real=torch.float32
        elif datatype == 'complex128':
            self.datatype_complex=torch.complex128
            self.datatype_real=torch.float64
        else:
            raise ValueError("datatype must be one of ['complex64', 'complex128']. User input: {}".format(datatype)) 
        
        if theta_array.device == wv_array.device:
            self.device=theta_array.device
        else:
            raise ValueError("all input tensor should be of same device")
        
        self.epsilon=100*sys.float_info.epsilon
        
        theta_array=enforce_torch_dtype(theta_array, [torch.int64,torch.float32,torch.float64])
        self.theta_array=theta_array.type(self.datatype_real)
        self.theta_array=self.theta_array/180*torch.pi
        
        wv_array=enforce_torch_dtype(wv_array, [torch.int64,torch.float32,torch.float64])
        self.wv_array=wv_array.type(self.datatype_real)
        
        if not (self.theta_array.shape[0] == self.wv_array.shape[0]):
            raise ValueError("batch size doesn't match")
            
        
    
    def coherent_layer_group(self, c_list, n, d, theta, n0):
        stack_information=[] #{n0,n,d,theta_incident} 
        incoherent_stack_information=[] #{n,theta_incident} or {coherent}
        
        
        
        in_stack=False
        for i in range(len(c_list)):
            if c_list[i]=='c':
                if not in_stack:
                    
                    incoherent_stack_information.append({'coherent':len(stack_information)})
                    
                    in_stack=True
                    stack_information.append({'n0':[],'n':[],'d':[],'theta_incident':[]})
                    stack_information[-1]['theta_incident'].append(theta[:,i])
                    
                    if i==0:
                        stack_information[-1]['n0'].append(n0[:,0])
                    else:
                        stack_information[-1]['n0'].append(n[:,i-1])
                    
                stack_information[-1]['n'].append(n[:,i])
                stack_information[-1]['d'].append(d[:,i])
                stack_information[-1]['theta_incident'].append(theta[:,i+1])
                
            
            else:
                if in_stack:
                    in_stack=False
                    stack_information[-1]['n0'].append(n[:,i])
                    stack_information[-1]['theta_incident'].append(theta[:,i+1])
                    
                    
                    stack_information[-1]['n0']=torch.stack(stack_information[-1]['n0']).T
                    stack_information[-1]['n']=torch.stack(stack_information[-1]['n']).T
                    stack_information[-1]['d']=torch.stack(stack_information[-1]['d']).T
                    stack_information[-1]['theta_incident']=torch.stack(stack_information[-1]['theta_incident']).T
                else:
                    if i==0:
                        incoherent_stack_information.append({'n':torch.stack([n0[:,0],n[:,i]],-1),'theta_incident':theta[:,i:i+2]})
                    else:
                        incoherent_stack_information.append({'n':n[:,i-1:i+1],'theta_incident':theta[:,i+1:i+3]})
                
        if in_stack:
            in_stack=False
            stack_information[-1]['n0'].append(n0[:,1])
            stack_information[-1]['theta_incident'].append(theta[:,-1])
            
            stack_information[-1]['n0']=torch.stack(stack_information[-1]['n0']).T
            stack_information[-1]['n']=torch.stack(stack_information[-1]['n']).T
            stack_information[-1]['d']=torch.stack(stack_information[-1]['d']).T
            stack_information[-1]['theta_incident']=torch.stack(stack_information[-1]['theta_incident']).T
        else:
            incoherent_stack_information.append({'n':torch.stack([n[:,i],n0[:,1]],-1),'theta_incident':theta[:,i+1:i+3]})
        
        
        return incoherent_stack_information,stack_information
        

    def Snell(self, n, n0):
        """
        return angle in each layer based refractive index using Snell's law.
        """
        
        angles=torch.arcsin((n0[:,0]*torch.sin(self.theta_array))[:,torch.newaxis]/n)
    
        angles_in=self.theta_array.type(self.datatype_complex)
        
        angles_out=torch.arcsin((n0[:,0]*torch.sin(self.theta_array))/n0[:,1]+0j)
        
        ncostheta = n0[:,0]* torch.cos(angles_out)
        isforward=(torch.abs(ncostheta.imag) > self.epsilon) & (ncostheta.imag > 0)
        isforward[torch.abs(ncostheta.imag) <=self.epsilon]=(ncostheta.real[torch.abs(ncostheta.imag) <=self.epsilon] > 0)
        
        angles_out[~isforward]=torch.pi-angles_out[~isforward]

        return torch.concat([angles_in[:,torch.newaxis],angles,angles_out[:,torch.newaxis]],1)
        
    
    def T_from_t(self, t, n_i, n_f, th_i, th_f):
        """
        Calculate transmitted power T, starting with transmission amplitude t.

        n_i,n_f are refractive indices of incident and final medium.

        th_i, th_f are (complex) propegation angles through incident & final medium
        (in radians, where 0=normal). "th" stands for "theta".

        """

        if self.pol == 's':
            return torch.abs(t)**2 * (((n_f*torch.cos(th_f)).real) / (n_i*torch.cos(th_i)).real)
        elif self.pol == 'p':
            return torch.abs(t)**2 * (((n_f*torch.conj(torch.cos(th_f))).real) / (n_i*torch.conj(torch.cos(th_i))).real)
      
    
    def coherent_tmm(self, stack):
        if not torch.abs((stack['n0'][:,0]*torch.sin(stack['theta_incident'][:,0])).imag).sum()/len(stack['n0'])<self.epsilon:
            if self.datatype_complex==torch.complex64:
                warnings.warn("Simulation precision is low. Please consider switching to complex128")
            else:
                warnings.warn("Simulation precision is low")
        
        kz = 2 * torch.pi * stack['n'] * torch.cos(stack['theta_incident'][:,1:-1]) / self.wv_array[:,torch.newaxis]
        
        delta = kz * stack['d']
        
        clamped_imag = torch.clamp(delta.imag, max=35)
        delta = torch.complex(delta.real, clamped_imag)
        #torch.clamp(delta.imag, max=35, out=delta.imag)
        
        #Freshnel equation
        n_temp=torch.concat([stack['n0'][:,0:1],stack['n'],stack['n0'][:,1:2]],1)
        if self.pol=='s':
            R_list=((n_temp[:,0:-1] * torch.cos(stack['theta_incident'][:,0:-1]) - n_temp[:,1:] * torch.cos(stack['theta_incident'][:,1:])) / (n_temp[:,0:-1] * torch.cos(stack['theta_incident'][:,0:-1]) + n_temp[:,1:] * torch.cos(stack['theta_incident'][:,1:])))
            T_list=2 * n_temp[:,0:-1] * torch.cos(stack['theta_incident'][:,0:-1]) / (n_temp[:,0:-1] * torch.cos(stack['theta_incident'][:,0:-1]) + n_temp[:,1:] * torch.cos(stack['theta_incident'][:,1:]))
            
            R2_list= (n_temp[:,1:].flip(-1) * torch.cos(stack['theta_incident'][:,1:].flip(-1)) - n_temp[:,0:-1].flip(-1) * torch.cos(stack['theta_incident'][:,0:-1].flip(-1))) / (n_temp[:,1:].flip(-1) * torch.cos(stack['theta_incident'][:,1:].flip(-1)) + n_temp[:,0:-1].flip(-1) * torch.cos(stack['theta_incident'][:,0:-1].flip(-1)))
            T2_list=2 * n_temp[:,1:].flip(-1) * torch.cos(stack['theta_incident'][:,1:].flip(-1)) / (n_temp[:,1:].flip(-1) * torch.cos(stack['theta_incident'][:,1:].flip(-1)) + n_temp[:,0:-1].flip(-1) * torch.cos(stack['theta_incident'][:,0:-1].flip(-1)))
        elif self.pol=='p':
            R_list=((n_temp[:,1:] * torch.cos(stack['theta_incident'][:,0:-1]) - n_temp[:,0:-1] * torch.cos(stack['theta_incident'][:,1:])) / (n_temp[:,1:] * torch.cos(stack['theta_incident'][:,0:-1]) + n_temp[:,0:-1] * torch.cos(stack['theta_incident'][:,1:])))
            T_list=2 * n_temp[:,0:-1] * torch.cos(stack['theta_incident'][:,0:-1]) / (n_temp[:,1:] * torch.cos(stack['theta_incident'][:,0:-1]) + n_temp[:,0:-1] * torch.cos(stack['theta_incident'][:,1:]))
            
            R2_list=((n_temp[:,0:-1].flip(-1) * torch.cos(stack['theta_incident'][:,1:].flip(-1)) - n_temp[:,1:].flip(-1) * torch.cos(stack['theta_incident'][:,0:-1].flip(-1))) / (n_temp[:,0:-1].flip(-1) * torch.cos(stack['theta_incident'][:,1:].flip(-1)) + n_temp[:,1:].flip(-1) * torch.cos(stack['theta_incident'][:,0:-1].flip(-1))))
            T2_list=2 * n_temp[:,1:].flip(-1) * torch.cos(stack['theta_incident'][:,1:].flip(-1)) / (n_temp[:,0:-1].flip(-1) * torch.cos(stack['theta_incident'][:,1:].flip(-1)) + n_temp[:,1:].flip(-1) * torch.cos(stack['theta_incident'][:,0:-1].flip(-1)))
        
        
        a=torch.exp(-1j*delta)
        b=R_list[:,1:]*a
        aa=torch.stack([a,b],-1)
        a=torch.exp(1j*delta)
        b=R_list[:,1:]*a
        bb=torch.stack([b,a],-1)
        M_list=torch.stack([aa,bb],-2)*(1/T_list[:,1:,torch.newaxis,torch.newaxis])
        Mtilde = reduce(torch.matmul,torch.unbind(M_list,1))
        
        a2=torch.exp(-1j*delta.flip(-1))
        b2=R2_list[:,1:]*a2
        aa2=torch.stack([a2,b2],-1)
        a2=torch.exp(1j*delta.flip(-1))
        b2=R2_list[:,1:]*a2
        bb2=torch.stack([b2,a2],-1)
        M_list2=torch.stack([aa2,bb2],-2)*(1/T2_list[:,1:,torch.newaxis,torch.newaxis])
        Mtilde2 = reduce(torch.matmul,torch.unbind(M_list2,1))
        
        a=1/T_list[:,0]
        b=R_list[:,0]*a
        temp=torch.stack([torch.stack([a,b],-1),torch.stack([b,a],-1)],-2)
        Mtilde =temp @ Mtilde
        
        a=1/T2_list[:,0]
        b=R2_list[:,0]*a
        temp=torch.stack([torch.stack([a,b],-1),torch.stack([b,a],-1)],-2)
        Mtilde2 =temp @ Mtilde2


        R = torch.abs( Mtilde[:,1,0]/Mtilde[:,0,0])**2
        T = self.T_from_t(1/Mtilde[:,0,0], stack['n0'][:,0], stack['n0'][:,1], stack['theta_incident'][:,0], stack['theta_incident'][:,-1])
        
        R2 = torch.abs( Mtilde2[:,1,0]/Mtilde2[:,0,0])**2
        T2 = self.T_from_t(1/Mtilde2[:,0,0], stack['n0'][:,1], stack['n0'][:,0], stack['theta_incident'][:,-1], stack['theta_incident'][:,0])
        
        
        return {'R': R, 'T': T, 'R2': R2, 'T2': T2}
    
        # return {'R': R, 'T': T}
                
    def __call__(self,  n, d, c_list, n0):
        '''
        n: Reflactive index of multiple thin film layers
        d: Thickness of multiple thin film layers
        c_list:  It is "coherency list". Each entry should be 'i' for incoherent or 'c' for 'coherent'.
        n0: Reflactive index of incoming and output media. We regard it as media with incoherency and infinite thickness. 
            Although the input type is restricted to complex number, only real part will be considered
        '''
        
        #check input data
        
        if not (self.device==n.device==d.device==n0.device):
            raise ValueError("all input tensor should be of same device")
        
        # Process and validate n0
        n0=enforce_torch_dtype(n0, [torch.int64,torch.complex64,torch.complex128,torch.float32,torch.float64])
        n0=n0.type(self.datatype_complex)
        
        if (torch.abs(n0.imag)>self.epsilon).sum().item():
            warnings.warn("The incident and ouput is dissipative media. The algorithm will treat them as non-dissipative media")
        n0=n0.real
        
        if n0.shape[1] !=2:
            raise ValueError("incident and output media setting incorrect")
        
        if not (n0.shape[0] == n.shape[0] == d.shape[0] == self.theta_array.shape[0] == self.wv_array.shape[0]):
            raise ValueError("batch size doesn't match")
        
        if not isinstance(c_list, list) and all(element in ['i', 'c'] for element in c_list):
            raise ValueError("c_list must be a list with only 'i' and 'c' as its elements")

        if not (n.shape[1]==d.shape[1]==len(c_list)):
            raise ValueError("layer number doesn't match")
        
        
        # # not recommended. changing precision can affect the magnitude of the gradient and can sometimes lead to gradients becoming zero ("underflow")
        # n=enforce_torch_dtype(n, [torch.complex64,torch.complex128])
        # d=enforce_torch_dtype(d, [torch.float32,torch.float64])
        # n=n.type(self.datatype_complex)
        # d=d.type(self.datatype_real)
        
        n=enforce_torch_dtype(n, [self.datatype_complex])
        d=enforce_torch_dtype(d, [self.datatype_real])
        
        theta=self.Snell(n, n0)
        
        incoherent_stack_information,stack_information=self.coherent_layer_group(c_list,n,d,theta,n0)
        
        P=[]
        for i in range(len(c_list)):
            if c_list[i]=='i':
                P.append(torch.clip(torch.exp(-4 * torch.pi * d[:,i] * (n[:,i] *torch.cos(theta[:,i+1])).imag / self.wv_array),min=self.epsilon))
                
        
        Ltilde=None
        count=0
        for information in incoherent_stack_information:
            if 'coherent' in information:
                result=self.coherent_tmm(stack_information[information['coherent']])
                T_list=(result['T'])
                R_list=(result['R'])
                T2_list=(result['T2'])
                R2_list=(result['R2'])
            else:
                if self.pol == 's':
                    R_list=(torch.abs(((information['n'][:,0] * torch.cos(information['theta_incident'][:,0]) - information['n'][:,1] * torch.cos(information['theta_incident'][:,1])) / (information['n'][:,0] * torch.cos(information['theta_incident'][:,0]) + information['n'][:,1] * torch.cos(information['theta_incident'][:,1]))))**2)
                    t=2 * information['n'][:,0] * torch.cos(information['theta_incident'][:,0]) / (information['n'][:,0] * torch.cos(information['theta_incident'][:,0]) + information['n'][:,1] * torch.cos(information['theta_incident'][:,1]))
                    R2_list=(torch.abs(((information['n'][:,1] * torch.cos(information['theta_incident'][:,1]) - information['n'][:,0] * torch.cos(information['theta_incident'][:,0])) / (information['n'][:,1] * torch.cos(information['theta_incident'][:,1]) + information['n'][:,0] * torch.cos(information['theta_incident'][:,0]))))**2)
                    t2=2 * information['n'][:,1] * torch.cos(information['theta_incident'][:,1]) / (information['n'][:,1] * torch.cos(information['theta_incident'][:,1]) + information['n'][:,0] * torch.cos(information['theta_incident'][:,0]))
                elif self.pol  == 'p':
                    R_list=(torch.abs(((information['n'][:,1] * torch.cos(information['theta_incident'][:,0]) - information['n'][:,0] * torch.cos(information['theta_incident'][:,1])) / (information['n'][:,1] * torch.cos(information['theta_incident'][:,0]) + information['n'][:,0] * torch.cos(information['theta_incident'][:,1]))))**2)
                    t=2 * information['n'][:,0] * torch.cos(information['theta_incident'][:,0]) / (information['n'][:,1] * torch.cos(information['theta_incident'][:,0]) + information['n'][:,0] * torch.cos(information['theta_incident'][:,1]))
                    R2_list=(torch.abs(((information['n'][:,0] * torch.cos(information['theta_incident'][:,1]) - information['n'][:,1] * torch.cos(information['theta_incident'][:,0])) / (information['n'][:,0] * torch.cos(information['theta_incident'][:,1]) + information['n'][:,1] * torch.cos(information['theta_incident'][:,0]))))**2)
                    t2=2 * information['n'][:,1] * torch.cos(information['theta_incident'][:,1]) / (information['n'][:,0] * torch.cos(information['theta_incident'][:,1]) + information['n'][:,1] * torch.cos(information['theta_incident'][:,0]))
                    
                T_list=(self.T_from_t(t, information['n'][:,0], information['n'][:,1], information['theta_incident'][:,0], information['theta_incident'][:,1]))
                T2_list=(self.T_from_t(t2, information['n'][:,1], information['n'][:,0], information['theta_incident'][:,1], information['theta_incident'][:,0]))
            
            
            if Ltilde==None:
                Ltilde=torch.stack([torch.stack([1/T_list,-R2_list/T_list],-1), torch.stack([R_list/T_list,T2_list-R2_list*R_list/T_list],-1)],-2)
            else:
                Ltilde=Ltilde @ torch.stack([torch.stack([1/T_list/P[count],-R2_list/T_list/P[count]],-1), torch.stack([P[count]*R_list/T_list,P[count]*(T2_list-R2_list*R_list/T_list)],-1)],-2)
                count+=1
            
            
        T = 1 / Ltilde[:,0,0]
        R = Ltilde[:,1,0] / Ltilde[:,0,0]

        
        return  {'R':R,'T':T}
      
                

        
        
        
        

