import numpy as np
import torch 

def enforce_numpy_dtype(data_to_check,expected_dtype):
    """
    Enforce that a specific argument is a NumPy array
    with a list of given dtype.
    
    expected_dtype should be a list of given datatype
    """
    
    if not isinstance(data_to_check, np.ndarray):
        raise TypeError(f"Argument must be a NumPy array, got {type(data_to_check).__name__}.")
    if data_to_check.dtype not in expected_dtype:
        raise TypeError(f"Argument must be one of the following dtype {expected_dtype}, but got {data_to_check.dtype}.")
        
    return data_to_check

def enforce_torch_dtype(data_to_check,expected_dtype):
    """
    Enforce that a specific argument is a torch tensor
    with a list of given dtype.
    
    expected_dtype should be a list of given datatype
    """
    
    if not isinstance(data_to_check, torch.Tensor):
        raise TypeError(f"Argument must be a torch tensor, got {type(data_to_check).__name__}.")
    if data_to_check.dtype not in expected_dtype:
        raise TypeError(f"Argument must be one of the following dtype {expected_dtype}, but got {data_to_check.dtype}.")
        
    return data_to_check