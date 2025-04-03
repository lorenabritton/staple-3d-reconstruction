# Entangled Hydrogel Granule Analysis

Software tool for 3D reconstruction and quantitative analysis of entangled U-shaped hydrogel granules from micro-CT images, enabling measurement of entanglements and void fraction essential for injectable gel therapies.

## Project Overview

Injectable hydrogels show promise for minimally invasive medical treatments, but current granular hydrogels often disintegrate post-injection. This project supports the development of novel entangled U-shaped granule hydrogels by providing tools to visualize and quantify their 3D microstructure.

The software pipeline processes micro-CT images to:
- Identify and track individual granules/staples through 3D space
- Detect and quantify entanglements between granules
- Calculate void fraction (porosity)
- Generate 3D visualizations of the microstructure

## Repository Structure

- `/image-processing/`: Scripts for image preprocessing and feature detection
- `/tracking/`: Algorithms for positional tracking of granules across CT slices
- `/reconstruction/`: 3D reconstruction code
- `/analysis/`: Scripts for entanglement detection and void fraction calculation
- `/simulation/`: Fake staple generator for testing scalability
- `/examples/`: Sample data and results

## Installation

### Dependencies

```
# Will be added later
```

## Usage

### Image Processing

```python
# Example code for processing micro-CT images
```

### Tracking Granules/Staples

```python
# Example code for tracking centroids across slices
```

### Entanglement Detection

```python
# Example code for detecting entanglements
```

### Void Fraction Calculation

```python
# Example code for calculating void fraction
```

## Citation

If you use this code in your research, please cite:
```
Britton, L. (2025). Visualization of Entangled Hydrogel Granules for Injectable Gel Therapies. 
Harvard University.
```

## License

MIT License

## Acknowledgments

- Kathy Liu
- Prof. Joanna Aizenberg
- Aizenberg Lab - Bioinspired Engineering Research Group
- Harvard Center for Nanoscale Systems (CNS)
