# differentiable_tmm

`differentiable_tmm` is a PyTorch-based transfer-matrix method (TMM) toolkit for
multilayer thin-film optics. It computes angle- and wavelength-dependent
reflectance/transmittance and can be connected directly to PyTorch optimizers for
inverse design of thin-film structures. The repository also includes color
conversion utilities for converting spectra to CIE XYZ, CIE Lab, and sRGB.

All optical lengths in this project use **nanometers (nm)**. Incident angles are
given in **degrees** when constructing the simulator.

## Installation

Install the package and all dependencies, including the example dependencies:

```bash
pip install differentiable_tmm
```

## Main API

The recommended public API is:

```python
from differentiable_tmm import (
    TMM,
    spectrum_interpolation,
    Spectrum2XYZ,
    XYZ2Lab,
    XYZ2sRGB,
    DE94,
)
```

`TMM` is the transfer-matrix module, `spectrum_interpolation` contains the
refractive-index helper classes, and the color-conversion classes/functions are
available directly from `differentiable_tmm`.

## Refractive Index Helpers

### `spectrum_interpolation.interpolate_refrective`

Creates a callable interpolation object for wavelength-dependent refractive
indices.

```python
from differentiable_tmm import spectrum_interpolation

n_fn = spectrum_interpolation.interpolate_refrective(
    wavelength,
    refrective,
    method="1d_interpolate",
)
```

Parameters:

- `wavelength`: `numpy.ndarray`
  - Shape: `(num_wavelengths,)`
  - Dtype: integer, `np.float32`, or `np.float64`
  - Unit: nm
  - Meaning: sampled wavelengths for the material data.
- `refrective`: `numpy.ndarray`
  - Shape: `(num_wavelengths,)`
  - Dtype: integer, float, or complex
  - Meaning: refractive index values `n` or complex values `n + 1j*k`.
- `method`: `str`
  - Supported values: `"1d_interpolate"` and `"cubicspline"`
  - Default: `"1d_interpolate"`

Call input:

```python
n_values = n_fn(query_wavelengths)
```

- `query_wavelengths`: `numpy.ndarray`
  - Shape: any 1D shape, usually `(batch_size,)`
  - Unit: nm
  - Values outside the source wavelength range are clipped to the nearest
    available wavelength.

Return value:

- `n_values`: `numpy.ndarray`
  - Shape: same as `query_wavelengths`
  - Dtype: complex
  - Unit: dimensionless refractive index.

Example:

```python
import numpy as np
from differentiable_tmm import spectrum_interpolation

data = np.loadtxt("example/assets/SiO2.txt")
SiO2_n_fn = spectrum_interpolation.interpolate_refrective(
    data[:, 0] * 1e3,
    data[:, 1] + 1j * data[:, 2],
)

wavelengths = np.array([450.0, 550.0, 650.0])
n_sio2 = SiO2_n_fn(wavelengths)
```

### `spectrum_interpolation.constent_refrective`

Creates a callable object for a wavelength-independent refractive index.

```python
air_n_fn = spectrum_interpolation.constent_refrective(1.0)
glass_n_fn = spectrum_interpolation.constent_refrective(1.52)
```

Parameters:

- `refrective`: real or complex scalar
  - Examples: `1.0`, `1.52`, `1.5 + 0.01j`

Call input:

- `value`: `numpy.ndarray`
  - Shape: any 1D shape, usually `(batch_size,)`
  - Unit: nm

Return value:

- `numpy.ndarray`
  - Shape: same as `value`
  - Dtype: complex
  - Every entry equals the constant refractive index.

## Transfer-Matrix Simulation

### `TMM.tmm`

`TMM.tmm` is the main differentiable transfer-matrix simulator.

```python
simulator = TMM.tmm(
    pol="s",
    datatype="complex128",
    theta_array=theta_array,
    wv_array=wv_array,
)
```

Constructor parameters:

- `pol`: `str`
  - `"s"` for s-polarized light.
  - `"p"` for p-polarized light.
- `datatype`: `str`
  - `"complex64"` or `"complex128"`.
  - This also determines the real dtype used internally:
    - `"complex64"` -> `torch.float32`
    - `"complex128"` -> `torch.float64`
- `theta_array`: `torch.Tensor`
  - Shape: `(batch_size,)`
  - Dtype: integer, `torch.float32`, or `torch.float64`
  - Unit: degrees
  - Meaning: incident angle for each simulation point.
- `wv_array`: `torch.Tensor`
  - Shape: `(batch_size,)`
  - Dtype: integer, `torch.float32`, or `torch.float64`
  - Unit: nm
  - Meaning: vacuum wavelength for each simulation point.

`theta_array` and `wv_array` must be on the same device and have the same first
dimension.

### Calling a Simulator

```python
result = simulator(
    n=n,
    d=d,
    n0=n0,
    c_list=c_list,
)
```

Input tensors:

- `n`: `torch.Tensor`
  - Shape: `(batch_size, num_layers)`
  - Dtype: complex dtype matching the simulator, usually `torch.complex128`
  - Meaning: refractive index of every physical layer in the stack.
- `d`: `torch.Tensor`
  - Shape: `(batch_size, num_layers)`
  - Dtype: real dtype matching the simulator, usually `torch.float64`
  - Unit: nm
  - Meaning: thickness of every physical layer in the stack.
- `n0`: `torch.Tensor`
  - Shape: `(batch_size, 2)`
  - Dtype: complex or real
  - Meaning:
    - `n0[:, 0]`: incident medium refractive index
    - `n0[:, 1]`: output/substrate-side medium refractive index
  - The incident and output media are treated as incoherent semi-infinite media.
- `c_list`: `list[str]`
  - Length: `num_layers`
  - Each entry must be:
    - `"c"` for coherent layer
    - `"i"` for incoherent layer
  - The layer count must satisfy:

```python
n.shape[1] == d.shape[1] == len(c_list)
```

Return value:

```python
{
    "R": R,
    "T": T,
}
```

- `R`: `torch.Tensor`
  - Shape: `(batch_size,)`
  - Reflectance.
- `T`: `torch.Tensor`
  - Shape: `(batch_size,)`
  - Transmittance.

The operations are differentiable with respect to differentiable inputs such as
`d`, so layer thicknesses can be optimized with PyTorch.

### Minimal Reflectance/Transmittance Example

```python
import numpy as np
import torch
from differentiable_tmm import TMM, spectrum_interpolation

device = torch.device("cpu")

air_n_fn = spectrum_interpolation.constent_refrective(1.0)
glass_n_fn = spectrum_interpolation.constent_refrective(1.52)
film_n_fn = spectrum_interpolation.constent_refrective(2.0)

wavelengths_np = np.linspace(400, 700, 31)
angles_np = np.zeros_like(wavelengths_np)

n0 = torch.tensor(
    np.array([air_n_fn(wavelengths_np), air_n_fn(wavelengths_np)]).T,
    dtype=torch.complex128,
).to(device)

n = torch.tensor(
    np.array([glass_n_fn(wavelengths_np), film_n_fn(wavelengths_np)]).T,
    dtype=torch.complex128,
).to(device)

d = torch.tensor(
    np.array([[500000.0, 100.0]] * len(wavelengths_np)),
    dtype=torch.float64,
).to(device)

theta_array = torch.tensor(angles_np, dtype=torch.float64).to(device)
wv_array = torch.tensor(wavelengths_np, dtype=torch.float64).to(device)
c_list = ["i", "c"]

sim_s = TMM.tmm("s", "complex128", theta_array, wv_array)
sim_p = TMM.tmm("p", "complex128", theta_array, wv_array)

result_s = sim_s(n=n, d=d, n0=n0, c_list=c_list)
result_p = sim_p(n=n, d=d, n0=n0, c_list=c_list)

reflectance = (result_s["R"] + result_p["R"]) / 2
transmittance = (result_s["T"] + result_p["T"]) / 2
```

## Color Conversion

The color utilities are imported directly from `differentiable_tmm`.

### `Spectrum2XYZ`

Converts spectral reflectance or transmittance curves to CIE XYZ.

```python
spectrum_to_xyz = Spectrum2XYZ(
    lightsource="D65",
    observer="CIE 1931 2 Degree Standard Observer",
    device=device,
    datatype="complex128",
    clip=True,
)
```

Parameters:

- `lightsource`: `str`
  - Must be one of the keys in `differentiable_tmm.illumination_store.illumination`.
  - Common values include `"A"`, `"D50"`, `"D55"`, `"D65"`, `"D75"`, and
    fluorescent illuminants such as `"FL1"`.
- `observer`: `str`
  - Supported observers are stored in `differentiable_tmm.observer_store.Observer`.
  - Common values:
    - `"CIE 1931 2 Degree Standard Observer"`
    - `"CIE 1964 10 Degree Standard Observer"`
- `device`: `torch.device`
  - Device where internal tensors are stored.
- `datatype`: `str`
  - `"complex64"` or `"complex128"`.
- `clip`: `bool`
  - If `True`, uses the ASTM E308 practical working wavelength range
    360-780 nm.

Useful attributes:

- `spectrum_to_xyz.wavelength`
  - `torch.Tensor`
  - Shape: `(num_wavelengths,)`
  - Unit: nm
  - Use this wavelength array when building TMM spectra for color conversion.
- `spectrum_to_xyz.datatype_real`
  - Real dtype corresponding to the selected complex dtype.

Call input:

```python
XYZ = spectrum_to_xyz(reflectances)
```

- `reflectances`: `torch.Tensor`
  - Shape: `(batch_size, num_wavelengths)`
  - Dtype: real or complex tensor compatible with the selected device/dtype
  - Meaning: spectral reflectance or transmittance sampled at
    `spectrum_to_xyz.wavelength`.

Return value:

- `XYZ`: `torch.Tensor`
  - Shape: `(batch_size, 3)`
  - Columns: `X`, `Y`, `Z`

### `XYZ2Lab`

Converts CIE XYZ to CIE Lab under a specified illuminant and observer.

```python
xyz_to_lab = XYZ2Lab(
    lightsource="D65",
    observer="CIE 1931 2 Degree Standard Observer",
    device=device,
    datatype="complex128",
)

Lab = xyz_to_lab(XYZ)
```

Input:

- `XYZ`: `torch.Tensor`
  - Shape: `(batch_size, 3)`
  - Columns: `X`, `Y`, `Z`

Output:

- `Lab`: `torch.Tensor`
  - Shape: `(batch_size, 3)`
  - Columns: `L*`, `a*`, `b*`

### `XYZ2sRGB`

Converts CIE XYZ to sRGB.

```python
xyz_to_srgb = XYZ2sRGB(device, datatype="complex128")
RGB = xyz_to_srgb(XYZ)
```

Input:

- `XYZ`: `torch.Tensor`
  - Shape: `(batch_size, 3)`

Output:

- `RGB`: `torch.Tensor`
  - Shape: `(batch_size, 3)`
  - Columns: red, green, blue
  - Values are not clipped inside the function. For plotting with Matplotlib,
    use:

```python
RGB_plot = np.clip(np.real(RGB.detach().cpu().numpy()), 0, 1)
```

### `DE94`

Computes CIE94 color difference.

```python
delta_e = DE94(Lab, Lab_target)
```

Inputs:

- `Lab`: `torch.Tensor`
  - Shape: `(batch_size, 3)`
- `Lab_target`: `torch.Tensor`
  - Shape: `(batch_size, 3)`

Output:

- `delta_e`: `torch.Tensor`
  - Shape: `(batch_size,)`

## Differentiable Optimization Pattern

A common inverse-design workflow is to optimize unconstrained variables and map
them into physical layer thickness bounds with a sigmoid:

```python
lower_bound = 1.0
upper_bound = 2000.0

struc_logits = torch.zeros(num_layers, dtype=torch.float64, requires_grad=True)
optimizer = torch.optim.Adam([struc_logits], lr=0.01)

for epoch in range(epochs):
    struc = lower_bound + (upper_bound - lower_bound) * torch.sigmoid(struc_logits)
    d = torch.cat([meta_data, struc.repeat(num_samples, 1)], dim=-1)

    # Run TMM -> spectrum -> XYZ -> Lab, then compare with target Lab.
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

This keeps every optimized layer thickness inside `[lower_bound, upper_bound]`
without adding a penalty term.

## Examples

Example scripts live in the `example/` directory. Run them from the project root:

```bash
python example/example1.py
python example/example2.py
python example/example3.py
python example/example4.py
python example/example5.py
```

All example PNG outputs are saved in the `example/assets/` directory.

### Example 1: Angle and Thickness Color Map

File: `example/example1.py`

Purpose:

- Simulates a 5-layer coating stack.
- Sweeps observation angle and one film thickness.
- Converts reflectance and transmittance spectra to sRGB.
- Saves two color maps:
  - `example/assets/example1.png`
  - `example/assets/example1_1.png`

Main output:

- Reflectance color map.
- Transmittance color map.

### Example 2: Normal-Incidence Lab Optimization

File: `example/example2.py`

Purpose:

- Optimizes a 5-layer stack at normal incidence.
- Target color is specified as Lab.
- Uses sigmoid thickness bounds.
- Saves:
  - `example/assets/example2.png`

Main output:

- Plot of `L*`, `a*`, and `b*` versus optimization epoch.
- Printed final layer thicknesses in nm.

### Example 3: Multi-Angle Lab Optimization

File: `example/example3.py`

Purpose:

- Optimizes a configurable multilayer coating against Lab targets at several
  observation angles.
- Uses a configurable stack builder:

```python
num_coating_layers = 11

def build_layer_refractive_indices(wavelengths):
    layers = [glass_n_fn(wavelengths)]
    coating_n_fns = [SiO2_n_fn, Si3N4_n_fn]
    for i in range(num_coating_layers):
        layers.append(coating_n_fns[i % 2](wavelengths))
    return np.array(layers)

def build_c_list():
    return ["i"] + ["c"] * num_coating_layers
```

Saves:

- `example/assets/example3.png`
- `example/assets/example3_1.png`

Main output:

- Lab value versus observation angle.
- Color bar visualization of the optimized design.
- Printed final layer thicknesses in nm.

### Example 4: Tolerance-Aware Optimization

File: `example/example4.py`

Purpose:

- Optimizes a configurable multilayer coating while sampling thickness tolerance
  variations with Latin hypercube sampling.
- Uses the same configurable layer count pattern as Example 3.
- Saves:
  - `example/assets/example4.png`

Main output:

- Comparison between initial and optimized designs under a tolerance sweep.
- Printed initial and final layer thicknesses in nm.

### Example 5: Color Under Different Illuminants

File: `example/example5.py`

Purpose:

- Computes reflectance once for a fixed coating design.
- Converts the same spectrum under all illuminants stored in
  `differentiable_tmm.illumination_store.illumination`.
- Saves:
  - `example/assets/example5.png`

Main output:

- A color bar showing the perceived sRGB color under each illuminant.

## Building the Package

Clean old builds:

```bash
rm -rf build dist *.egg-info
```

Build source distribution and wheel:

```bash
python -m build
```

The build artifacts will be created in `dist/`.

Check the package metadata:

```bash
twine check dist/*
```

Upload to TestPyPI first:

```bash
twine upload --repository testpypi dist/*
```

Upload to PyPI:

```bash
twine upload dist/*
```

After uploading, users can install with the command shown in the installation
section.

## Notes for Maintainers

- The recommended public import is `from differentiable_tmm import ...`.
- The package still includes `core` and `utils` as top-level packages for
  compatibility with existing scripts.
- The historical class/function names use `refrective` and `constent` spelling.
  They are kept unchanged to preserve compatibility with the existing code.
- Keep `differentiable_tmm/__init__.py` synchronized with the intended public
  API when adding new user-facing functions.
- `torch` wheels are platform-specific. If users need a CUDA-specific PyTorch
  installation, they should install PyTorch following the official PyTorch
  instructions before installing this package.
