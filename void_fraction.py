
"""
will prompt for:
- path to tif file
- path for output csv file
- threshold value, use default of 156
- whether to generate visualizations (default is yes)

can also run with command-line:
        python void_fraction_calculator.py --input your_file.tif --output results.csv --threshold 150 --no-vis
(Add `--no-vis` if you want to skip visualization)

"""

import numpy as np
import cv2
from skimage import io
import pandas as pd
from tqdm import tqdm
import argparse
import os
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def calculate_void_fraction(input_tif, output_csv, threshold=156, visualize=True):
    """
    Calculate void fraction for each slice and total void volume from a multi-page TIF file.
    Optionally visualize the slices with void area highlighted.
    
    Parameters:
    -----------
    input_tif : str
        Path to the multi-page TIF file containing all slices
    output_csv : str
        Path to save the output CSV file
    threshold : int, optional
        Intensity threshold for binarization (default: 156)
    visualize : bool, optional
        Whether to create visualization images (default: True)
    
    Returns:
    --------
    float: The total void fraction for the entire volume
    """
    # Read all slices from the multi-page TIF file
    print(f"Reading slices from {input_tif}...")
    try:
        # tifffile is more reliable for multi-page TIFs
        from tifffile import imread
        all_slices = imread(input_tif)
    except ImportError:
        # Fall back to skimage if tifffile is not available
        all_slices = io.imread(input_tif)
    
    # Check if we have a 3D stack
    if len(all_slices.shape) < 3:
        print("The input file does not appear to be a multi-page TIF. Only one slice found.")
        all_slices = np.expand_dims(all_slices, axis=0)  # Convert to 3D array with one slice
    
    # Get dimensions
    num_slices, height, width = all_slices.shape[:3]
    total_pixels_per_slice = height * width
    
    print(f"Processing {num_slices} slices of size {height}x{width}...")
    
    # Initialize results dictionary
    results = {
        'slice_number': [],
        'void_area_pixels': [],
        'total_area_pixels': [],
        'void_fraction': []
    }
    
    # Create output directory for visualizations
    if visualize:
        output_dir = os.path.dirname(output_csv)
        if not output_dir:
            output_dir = '.'
            
        vis_dir = os.path.join(output_dir, 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)
        print(f"Visualizations will be saved to {vis_dir}")
    
    # Process each slice
    total_void_volume = 0
    
    for i in tqdm(range(num_slices)):
        # Get the current slice
        img = all_slices[i]
        
        # Convert to grayscale if image is RGB
        if len(img.shape) > 2:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # Ensure the image is in uint8 format for thresholding
        if img.dtype != np.uint8:
            img = (img / img.max() * 255).astype(np.uint8)
        
        # Binarize the image
        _, binary = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
        
        # Calculate void area (white pixels are material, black pixels are void)
        # SWITCH TEMPORARY
        void_pixels = np.sum(binary > 0)
        material_pixels = total_pixels_per_slice - void_pixels
        
        # Calculate void fraction
        void_fraction = void_pixels / total_pixels_per_slice
        
        # Accumulate total void volume
        total_void_volume += void_pixels
        
        # Store results
        results['slice_number'].append(i+1)
        results['void_area_pixels'].append(void_pixels)
        results['total_area_pixels'].append(total_pixels_per_slice)
        results['void_fraction'].append(void_fraction)
        
        # Create visualization if requested
        if visualize and (i < 10 or i % 10 == 0 or i >= num_slices - 10):  # First 10, every 10th, and last 10
            # Create custom colormap: material=blue, void=red
            # SWITCH
            colors = [(1, 0, 0, 0.7), (0, 0, 1, 0.7)]  # Blue for material, Red for void
            cmap = LinearSegmentedColormap.from_list('material_void', colors, N=2)
            
            # Create a figure with two subplots
            fig, axs = plt.subplots(1, 3, figsize=(15, 5))
            
            # Original grayscale image
            axs[0].imshow(img, cmap='gray')
            axs[0].set_title('Original Slice')
            axs[0].axis('off')
            
            # Binary image
            axs[1].imshow(binary, cmap='gray')
            axs[1].set_title(f'After Thresholding ({threshold})')
            axs[1].axis('off')
            
            # Visualization with material and void areas colored
            # Invert binary for visualization (1=material, 0=void)
            #SWITCH
            material_void_map = np.zeros_like(binary)
            material_void_map[binary > 0] = 0  # Material = 1
            material_void_map[binary == 0] = 1  # Void = 0
            
            axs[2].imshow(material_void_map, cmap=cmap)
            axs[2].set_title(f'Void Fraction: {void_fraction:.4f} ({void_fraction*100:.1f}%)')
            axs[2].axis('off')
            
            # Add a colorbar
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=axs[2])
            cbar.set_ticks([0.25, 0.75])
            cbar.set_ticklabels(['Void', 'Material'])
            
            # Save figure
            plt.tight_layout()
            fig_path = os.path.join(vis_dir, f'slice_{i+1:04d}.png')
            plt.savefig(fig_path, dpi=150)
            plt.close(fig)
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(results)
    
    # Add total void volume as the last row
    total_row = pd.DataFrame({
        'slice_number': ['Total'],
        'void_area_pixels': [total_void_volume],
        'total_area_pixels': [total_pixels_per_slice * num_slices],
        'void_fraction': [total_void_volume / (total_pixels_per_slice * num_slices)]
    })
    
    # Append total row to DataFrame
    df = pd.concat([df, total_row], ignore_index=True)
    
    # Save to CSV
    df.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")
    
    # Create summary visualization
    if visualize:
        # Plot void fraction for each slice
        plt.figure(figsize=(12, 6))
        plt.plot(df['slice_number'][:-1], df['void_fraction'][:-1], 'b-')
        plt.axhline(y=df['void_fraction'].iloc[-1], color='r', linestyle='--', 
                   label=f'Average: {df["void_fraction"].iloc[-1]:.4f}')
        plt.xlabel('Slice Number')
        plt.ylabel('Void Fraction')
        plt.title('Void Fraction Distribution Across Slices')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Save the summary plot
        summary_path = os.path.join(output_dir, 'void_fraction_summary.png')
        plt.savefig(summary_path, dpi=150)
        plt.close()
        print(f"Summary visualization saved to {summary_path}")
    
    # Return the final void fraction for the entire volume
    return total_void_volume / (total_pixels_per_slice * num_slices)

def main():
    print("===== Micro-CT Void Fraction Calculator =====")
    
    # Get input from the user if not provided as arguments
    parser = argparse.ArgumentParser(description='Calculate void fraction from multi-page TIF file')
    parser.add_argument('--input', type=str, help='Path to multi-page TIF file')
    parser.add_argument('--output', type=str, help='Path to output CSV file')
    parser.add_argument('--threshold', type=int, default=156, help='Threshold for binarization (0-255)')
    parser.add_argument('--no-vis', action='store_true', help='Disable visualization')
    
    args = parser.parse_args()
    
    # If not provided through command line, ask for input
    input_tif = args.input
    if not input_tif:
        input_tif = input("Enter the path to the multi-page TIF file: ")
    
    output_csv = args.output
    if not output_csv:
        output_csv = input("Enter the path for the output CSV file: ")
    
    threshold = args.threshold
    threshold_input = input(f"Enter the threshold value for binarization (0-255) [default: {threshold}]: ")
    if threshold_input.strip():
        threshold = int(threshold_input)
    
    visualize = not args.no_vis
    if not args.no_vis:
        vis_input = input("Generate visualization images? (y/n) [default: y]: ").lower()
        visualize = vis_input != 'n'
    
    try:
        void_fraction = calculate_void_fraction(
            input_tif, 
            output_csv, 
            threshold,
            visualize
        )
        
        print(f"Total void fraction: {void_fraction:.4f}")
        print(f"This means approximately {void_fraction*100:.2f}% of the volume is void space.")
    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please check your input file and try again.")

if __name__ == "__main__":
    main()