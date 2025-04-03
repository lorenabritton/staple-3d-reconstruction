# Entangled Hydrogel Granule Analysis

Software tool for 3D reconstruction and quantitative analysis of entangled U-shaped hydrogel granules from micro-CT images, enabling measurement of entanglements and void fraction essential for injectable gel therapies.

## Project Overview

Injectable hydrogels show promise for minimally invasive medical treatments, but current granular hydrogels often disintegrate post-injection. This project supports the development of novel entangled U-shaped granule hydrogels by providing tools to visualize and quantify their 3D microstructure.

The software pipeline processes micro-CT images to:
- Identify and track individual granules/staples through 3D space
- Detect and quantify entanglements between granules
- Calculate void fraction (porosity)
- Generate 3D visualizations of the microstructure

## Installation

### Dependencies

```
numpy
opencv-python
scikit-image
pandas
matplotlib
tqdm
mpl_toolkits.mplot3d
tifffile
```
Optional Dependencies:
```
scipy.spatial.KDTree
scipy.optimize.linear_sum_assignment
mpld3
```

## Usage

### Image Processing and Staple/Granule Tracking

Need a single tif image
```python
reconstruct_staples.py
```

### Entanglement Detection

```python
entanglements.py
```

### Void Fraction Calculation

```python
void_fraction.py
```

### Fake Staple Generator

```python
fake_granule_generator.py
```

## Citation

If you use this code in your research, please cite:
```
Britton, L. (2025). Visualization of Entangled Hydrogel Granules for Injectable Gel Therapies. 
Harvard University.
```
