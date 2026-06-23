import numpy as np
from scipy.interpolate import interp1d,CubicSpline
from .datatype_restriction import enforce_numpy_dtype

class interpolate_refrective():
    def __init__(self, wavelength, refrective, method='1d_interpolate'):
        '''
        unit wave length should be nm and refrective is complex array
        method can be choose between (1d_interpolate,cubicspline). Default is 1d 1d_interpolate.
        if input wavelength is outside the initiate wavelength range, the algorithm will choose the nearest one
        '''
        
        self.wavelength=enforce_numpy_dtype(wavelength, [int,np.float32,np.float64])
        self.wavelength=self.wavelength.astype(np.float64)
        self.refrective=enforce_numpy_dtype(refrective, [int,np.float32,np.float64,np.complex64,np.complex128])
        self.refrective=self.refrective.astype(np.complex128)
        
        self.wv_range=[wavelength.min(),wavelength.max()]
        
        #right now only support two types of interpolation
        if method=='1d_interpolate':
            self.fn=interp1d(wavelength, refrective, kind='linear')
        elif method=='cubicspline':
            self.fn=CubicSpline(wavelength, refrective)
        else:
            raise ValueError("Right now only support two types of interpolation: ['1d_interpolate','cubicspline']. User input: {}".format(method)) 
    
    def __call__(self, value):
        '''
        value should be in unit nm
        '''
        
        value=enforce_numpy_dtype(value, [int,np.float32,np.float64])
        value=value.astype(np.float64)
        
        value[value<self.wv_range[0]]=self.wv_range[0]
        value[value>self.wv_range[1]]=self.wv_range[1]
        
        return self.fn(value)


class constent_refrective():
    def __init__(self, refrective):
        '''
        refrective must be a real or complex number
        '''
        
        try:
            self.refrective=complex(refrective)
        except:
            raise TypeError(f"Can not transform refrective index to complex number, input type: {type(refrective)}.")
        
        
    def __call__(self, value):
        '''
        value should be in unit nm
        '''
        value=enforce_numpy_dtype(value, [int,np.float32,np.float64])
        output=np.ones(value.shape,dtype=np.complex128)
        
        return output*self.refrective


