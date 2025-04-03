"""
Staple Reconstructor
-------------------
This script implements a 3D reconstruction algorithm for U-shaped metal staples from micro-CT images.
It processes a stack of 2D slices and reconstructs the 3D structure of each staple,
identifying their positions, orientations, and generating statistics about their properties.
"""

import numpy as np
import cv2
from skimage import io, measure, morphology
import pandas as pd
from tqdm import tqdm
import argparse
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import warnings

# Import 3D visualization libraries with fallbacks
# This allows the script to run even if certain dependencies are missing
try:
    from mpl_toolkits.mplot3d import Axes3D
    has_axes3d = True
except ImportError:
    has_axes3d = False
    warnings.warn("mpl_toolkits.mplot3d.Axes3D not available, 3D visualizations will be limited")

try:
    from scipy.spatial import KDTree
    has_kdtree = True
except ImportError:
    has_kdtree = False
    warnings.warn("scipy.spatial.KDTree not available, using alternative method for matching")

try:
    from scipy.optimize import linear_sum_assignment
    has_hungarian = True
except ImportError:
    has_hungarian = False
    warnings.warn("scipy.optimize.linear_sum_assignment not available, using greedy matching")


class StapleReconstructor:
    """
    A class that reconstructs 3D staples from a stack of micro-CT image slices.
    
    This class implements the complete pipeline for:
    1. Loading and preprocessing micro-CT slices
    2. Detecting staple cross-sections in each slice (Staple Points of Interest, or SPOIs)
    3. Tracking these points across slices to reconstruct complete 3D staples
    4. Analyzing staple geometry, size, and shape characteristics
    5. Generating visualizations and statistical outputs
    """
    
    def __init__(self, input_tif, output_dir, threshold=156, max_distance=10, min_size=5, min_staple_points=20):
        """
        Initialize the staple reconstructor with the given parameters.
        
        Parameters:
        -----------
        input_tif : str
            Path to the multi-page TIF file containing all micro-CT slices
        output_dir : str
            Directory to save the output files (visualizations, data, statistics)
        threshold : int, optional
            Intensity threshold for Otsu binarization (0-255, default: 156)
        max_distance : float, optional
            Maximum distance between centroids in adjacent slices to be considered 
            part of the same staple (pixels, default: 10)
        min_size : int, optional
            Minimum size (in pixels) for a region to be considered a valid 
            staple cross-section (default: 5)
        min_staple_points : int, optional
            Minimum number of points required for a complete staple (default: 20)
            Staples with fewer points are considered incomplete and filtered out
        """
        self.input_tif = input_tif
        self.output_dir = output_dir
        self.threshold = threshold
        self.max_distance = max_distance
        self.min_size = min_size
        self.min_staple_points = min_staple_points
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize data structures
        self.all_staples = []  # Will store all identified staples
        self.staple_colors = {}  # For consistent coloring in visualizations
        
    def load_slices(self):
        """
        Load all micro-CT slices from the multi-page TIF file.
        
        The function attempts to use tifffile if available (better multi-page TIF support),
        otherwise falls back to skimage.io.imread.
        """
        print(f"Reading slices from {self.input_tif}...")
        try:
            # tifffile is more reliable for multi-page TIFs
            from tifffile import imread
            self.all_slices = imread(self.input_tif)
        except ImportError:
            # Fall back to skimage if tifffile is not available
            self.all_slices = io.imread(self.input_tif)
        
        # Check if we have a proper 3D stack
        if len(self.all_slices.shape) < 3:
            print("The input file does not appear to be a multi-page TIF. Only one slice found.")
            self.all_slices = np.expand_dims(self.all_slices, axis=0)  # Convert to 3D array with one slice
        
        # Get stack dimensions for reference
        self.num_slices, self.height, self.width = self.all_slices.shape[:3]
        print(f"Loaded {self.num_slices} slices of size {self.height}x{self.width}")
    
    def preprocess_slice(self, img):
        """
        Preprocess a single micro-CT slice to prepare it for feature detection.
        
        Steps:
        1. Convert to grayscale if necessary
        2. Apply circular masking to focus on the region of interest
           (removes container edges and other artifacts)
        3. Apply threshold to separate staples from background
        4. Perform morphological operations to remove small noise
        
        Parameters:
        -----------
        img : numpy.ndarray
            Input slice image
            
        Returns:
        --------
        binary : numpy.ndarray
            Binary image with staple cross-sections in white (255) on black (0) background
        """
        # Convert to grayscale if image is RGB
        if len(img.shape) > 2:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # Ensure the image is in uint8 format for thresholding
        if img.dtype != np.uint8:
            img = (img / img.max() * 255).astype(np.uint8)
        
        # Create a circular mask to exclude scale bar and focus on region of interest
        center_y, center_x = img.shape[0] // 2, img.shape[1] // 2
        radius = min(img.shape) // 2 - 20  # Slightly smaller than image dimensions
        
        # Create the mask (white circle on black background)
        mask = np.zeros_like(img)
        cv2.circle(mask, (center_x, center_y), radius, 255, -1)
        
        # Apply the mask to the image
        masked_img = cv2.bitwise_and(img, mask)
        
        # Apply Otsu thresholding to separate staples from background
        _, binary = cv2.threshold(masked_img, self.threshold, 255, cv2.THRESH_BINARY)
        
        # Clean up with morphological operations
        # Remove small noise (isolated pixels)
        binary = morphology.remove_small_objects(binary.astype(bool), min_size=self.min_size)
        # Fill small holes within staple cross-sections
        binary = morphology.remove_small_holes(binary, area_threshold=20)
        
        return binary.astype(np.uint8) * 255
    
    def detect_regions(self, binary):
        """
        Detect and label connected regions (staple cross-sections) in a binary image.
        
        Parameters:
        -----------
        binary : numpy.ndarray
            Binary image with staple cross-sections in white
            
        Returns:
        --------
        region_data : list of dict
            List of dictionaries containing properties of each detected region 
            (centroid, area, bounding box, label)
        labeled : numpy.ndarray
            Labeled image with unique integers for each region
        """
        # Label connected regions (assign unique integer to each region)
        labeled = measure.label(binary > 0)
        regions = measure.regionprops(labeled)
        
        region_data = []
        for region in regions:
            # Skip very small regions (likely noise)
            if region.area < self.min_size:
                continue
                
            # Get region properties
            centroid = region.centroid
            area = region.area
            bbox = region.bbox
            
            region_data.append({
                'centroid': (centroid[0], centroid[1]),  # (y, x) coordinates
                'area': area,
                'bbox': bbox,
                'label': region.label
            })
            
        return region_data, labeled
    
    def match_regions_between_slices(self, regions_prev, regions_curr):
        """
        Match regions (staple cross-sections) between consecutive slices.
        
        This function assigns regions in the current slice to regions in the previous slice
        based on spatial proximity, forming the basis for tracking staples through the volume.
        
        Important note: This only matches regions between consecutive slices, not within the same slice.
        A region within max_distance of another region in the next slice is considered 
        part of the same staple. Regions in the same slice always belong to different staples.
        
        Parameters:
        -----------
        regions_prev : list of dict
            List of region data from previous slice
        regions_curr : list of dict
            List of region data from current slice
            
        Returns:
        --------
        matches : list of tuple
            List of (prev_idx, curr_idx) pairs indicating matched regions
        """
        if not regions_prev or not regions_curr:
            return []  # No matches if either list is empty
        
        # Create arrays of centroids
        centroids_prev = np.array([r['centroid'] for r in regions_prev])
        centroids_curr = np.array([r['centroid'] for r in regions_curr])
        
        # Calculate distance matrix between all centroids
        cost_matrix = np.zeros((len(regions_prev), len(regions_curr)))
        for i, prev in enumerate(centroids_prev):
            for j, curr in enumerate(centroids_curr):
                cost_matrix[i, j] = np.sqrt((prev[0] - curr[0])**2 + (prev[1] - curr[1])**2)
        
        # Match regions using optimal assignment if available, or greedy matching otherwise
        matches = []
        if has_hungarian:
            # Use Hungarian algorithm to find optimal assignment (minimizes total distance)
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            
            # Create matches, but only if distance is below threshold
            for i, j in zip(row_ind, col_ind):
                if cost_matrix[i, j] <= self.max_distance:
                    matches.append((i, j))
        else:
            # Greedy matching algorithm (fallback if scipy is not available)
            # For each centroid in previous slice, find closest match in current slice
            # that is within max_distance
            for i, prev in enumerate(centroids_prev):
                best_j = -1
                best_dist = float('inf')
                
                for j, curr in enumerate(centroids_curr):
                    dist = cost_matrix[i, j]
                    if dist <= self.max_distance and dist < best_dist:
                        best_j = j
                        best_dist = dist
                
                if best_j >= 0:
                    matches.append((i, best_j))
        
        return matches
    
    def process_slices(self):
        """
        Process all slices and track staples through the volume.
        
        This is the core function that:
        1. Processes each slice to detect staple cross-sections
        2. Tracks these regions across slices to identify complete staples
        3. Ensures that regions in the same slice are never considered part of the same staple
        
        Returns:
        --------
        filtered_tracks : list of dict
            List of dictionaries containing staple track data, including only staples
            with at least 3 points
        """
        print("Processing slices and tracking staples...")
        
        # Initialize list to store region data and labeled images for each slice
        all_regions = []
        all_labeled_images = []
        
        # First pass: detect regions in each slice
        for i in tqdm(range(self.num_slices), desc="Detecting regions"):
            img = self.all_slices[i]
            binary = self.preprocess_slice(img)
            regions, labeled = self.detect_regions(binary)
            
            # Store data for later use
            all_regions.append(regions)
            all_labeled_images.append(labeled)
            
            # Create visualization of detected regions for selected slices
            # (first 10, every 20th, and last 10 slices)
            if i < 10 or i % 20 == 0 or i >= self.num_slices - 10:
                self.visualize_regions(img, binary, regions, labeled, i)
        
        # Second pass: track staples across slices
        next_staple_id = 0
        active_staples = {}  # Currently active staples (key: staple_id, value: list of region indices)
        staple_tracks = []  # Complete tracks (records of all staples)
        
        for i in tqdm(range(self.num_slices - 1), desc="Tracking staples"):
            regions_curr = all_regions[i]
            regions_next = all_regions[i + 1]
            
            # Match regions between current and next slice only
            matches = self.match_regions_between_slices(regions_curr, regions_next)
            
            # Update active staples for this iteration
            new_active_staples = {}
            
            # Process matches - connect regions between slices
            for curr_idx, next_idx in matches:
                curr_region = regions_curr[curr_idx]
                next_region = regions_next[next_idx]
                
                # Check if current region is part of an active staple
                staple_found = False
                for staple_id, region_indices in active_staples.items():
                    if curr_idx in region_indices:
                        # Add next region to this staple
                        new_active_staples.setdefault(staple_id, []).append(next_idx)
                        staple_found = True
                        break
                
                if not staple_found:
                    # Create a new staple
                    staple_id = next_staple_id
                    next_staple_id += 1
                    new_active_staples[staple_id] = [next_idx]
                    
                    # Start a new track
                    staple_tracks.append({
                        'id': staple_id,
                        'points': [(i, curr_region['centroid'][0], curr_region['centroid'][1])],
                        'sizes': [curr_region['area']]
                    })
                
                # Add next point to the track
                for track in staple_tracks:
                    if track['id'] in new_active_staples and next_idx in new_active_staples[track['id']]:
                        track['points'].append((i+1, next_region['centroid'][0], next_region['centroid'][1]))
                        track['sizes'].append(next_region['area'])
            
            # Find regions in current slice that weren't matched (end of a staple)
            matched_curr_indices = [curr_idx for curr_idx, _ in matches]
            for curr_idx, curr_region in enumerate(regions_curr):
                if curr_idx not in matched_curr_indices:
                    # This region is not matched to any region in the next slice
                    # It could be the start of a new staple if it hasn't been seen before
                    staple_found = False
                    for staple_id, region_indices in active_staples.items():
                        if curr_idx in region_indices:
                            # This was part of an active staple, but now the staple ends
                            staple_found = True
                            break
                    
                    if not staple_found and i > 0:
                        # This is an isolated region not connected to previous or next slice
                        # We could optionally create a single-point staple, but usually skip these
                        # as they are likely noise or partial staple fragments
                        pass
            
            # Update active staples for next iteration
            active_staples = new_active_staples
        
        # Filter staple tracks based on length
        filtered_tracks = []
        for track in staple_tracks:
            if len(track['points']) >= 3:  # Minimum number of slices
                filtered_tracks.append(track)
        
        self.staple_tracks = filtered_tracks
        print(f"Identified {len(filtered_tracks)} potential staples")
        
        # Assign colors to staples for visualization
        self.assign_staple_colors()
        
        return filtered_tracks
    
    def assign_staple_colors(self):
        """
        Assign unique colors to staples for consistent visualization.
        
        This ensures that the same staple has the same color in all visualizations,
        making it easier to identify specific staples across different views.
        """
        # Get a list of distinct colors from matplotlib
        colors = list(mcolors.TABLEAU_COLORS.values())
        # If we need more colors, add some from CSS colors
        colors.extend(list(mcolors.CSS4_COLORS.values())[::5])  # Take every 5th color
        
        # Assign colors to staples
        for i, track in enumerate(self.staple_tracks):
            self.staple_colors[track['id']] = colors[i % len(colors)]
    
    def visualize_regions(self, original, binary, regions, labeled, slice_idx):
        """
        Create visualization of detected regions in a slice.
        
        Parameters:
        -----------
        original : numpy.ndarray
            Original slice image
        binary : numpy.ndarray
            Binary image after preprocessing
        regions : list of dict
            List of region data
        labeled : numpy.ndarray
            Labeled image with unique integers for each region
        slice_idx : int
            Slice index
        """
        # Create output directory for slice visualizations
        vis_dir = os.path.join(self.output_dir, 'slice_visualizations')
        os.makedirs(vis_dir, exist_ok=True)
        
        # Create a figure with subplots
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        
        # Original image with threshold
        axs[0].imshow(original, cmap='gray')
        axs[0].set_title(f'Original Slice {slice_idx+1}')
        axs[0].axis('off')
        
        # Colored regions
        axs[1].imshow(labeled, cmap='nipy_spectral')
        axs[1].set_title(f'Detected Regions (n={len(regions)})')
        axs[1].axis('off')
        
        # Add centroids to the image
        for region in regions:
            y, x = region['centroid']
            axs[1].plot(x, y, 'w+', markersize=8)  # White plus sign at centroid
        
        # Save figure
        plt.tight_layout()
        fig_path = os.path.join(vis_dir, f'slice_{slice_idx+1:04d}_regions.png')
        plt.savefig(fig_path, dpi=150)
        plt.close(fig)
    
    def visualize_3d_staples(self):
        """
        Create 3D visualization of reconstructed staples.
        
        This function generates a 3D plot showing all staples, with colors 
        indicating staple identity. Complete staples (with enough points) are
        shown in bright colors, while incomplete staples are shown in gray.
        
        Returns:
        --------
        filtered_tracks : list of dict
            List of dictionaries containing staple track data, including only staples
            with at least min_staple_points
        """
        if not has_axes3d:
            print("Warning: 3D visualization not available without mpl_toolkits.mplot3d.Axes3D")
            print("Saving only 2D projections instead.")
            self.visualize_2d_projections()
            return []
            
        print("Creating 3D visualization of staples...")
        
        # Filter out staples with too few points
        filtered_tracks = [track for track in self.staple_tracks if len(track['points']) >= self.min_staple_points]
        
        # Create figure
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot each staple
        for i, track in enumerate(self.staple_tracks):
            # Extract points
            points = np.array(track['points'])
            
            # Get z, y, x coordinates (slice_idx, row, col)
            z, y, x = points[:, 0], points[:, 1], points[:, 2]
            
            # Get color for this staple 
            # Blue for staples with enough points, gray for short ones
            if len(track['points']) >= self.min_staple_points:
                color = self.staple_colors.get(track['id'], 'blue')
                linewidth = 1.5
                alpha = 0.7
            else:
                color = 'gray'
                linewidth = 0.5
                alpha = 0.3
            
            # Plot every point of the staple to show potential bends
            ax.scatter(x, y, z, color=color, s=10, marker='o', alpha=alpha)
            
            # Connect points with line segments to show path
            ax.plot(x, y, z, '-', color=color, linewidth=linewidth, alpha=alpha)
        
        # Set labels and title
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')
        ax.set_zlabel('Z (slice)')
        ax.set_title(f'3D Reconstruction of {len(filtered_tracks)} Complete Staples\n(Minimum {self.min_staple_points} points)')
        
        # Save figure
        fig_path = os.path.join(self.output_dir, '3d_staples_reconstruction.png')
        plt.savefig(fig_path, dpi=300)
        
        # Save from different angles for better 3D understanding
        for i, angle in enumerate([(30, 45), (0, 0), (0, 90), (90, 0)]):
            ax.view_init(elev=angle[0], azim=angle[1])
            fig_path = os.path.join(self.output_dir, f'3d_staples_view{i+1}.png')
            plt.savefig(fig_path, dpi=300)
        
        # Try to save interactive figure (can be rotated) if mpld3 is available
        try:
            from mpld3 import fig_to_html, save_html
            fig_path_html = os.path.join(self.output_dir, '3d_staples_reconstruction.html')
            save_html(fig, fig_path_html)
            print(f"Interactive 3D visualization saved to {fig_path_html}")
        except Exception as e:
            print(f"Could not create interactive visualization: {e}")
            print("This is not critical - static images have been saved.")
        
        plt.close(fig)
        
        # Create summary statistics of staple lengths
        lengths = [len(track['points']) for track in self.staple_tracks]
        if lengths:
            plt.figure(figsize=(10, 6))
            plt.hist(lengths, bins=20, color='steelblue', edgecolor='black')
            plt.axvline(x=self.min_staple_points, color='red', linestyle='--', 
                       label=f'Minimum threshold ({self.min_staple_points} points)')
            plt.xlabel('Number of Points per Staple')
            plt.ylabel('Frequency')
            plt.title('Distribution of Staple Lengths')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # Save histogram
            hist_path = os.path.join(self.output_dir, 'staple_length_histogram.png')
            plt.savefig(hist_path, dpi=150)
            plt.close()
            
            # Print summary statistics
            print(f"Staple length statistics:")
            print(f"  Total staples: {len(self.staple_tracks)}")
            print(f"  Complete staples (≥{self.min_staple_points} points): {len(filtered_tracks)}")
            print(f"  Minimum points: {min(lengths)}")
            print(f"  Maximum points: {max(lengths)}")
            print(f"  Average points: {sum(lengths)/len(lengths):.1f}")
        
        return filtered_tracks
        
    def visualize_2d_projections(self):
        """
        Create 2D projections of the 3D staple reconstruction.
        
        This function generates three 2D projections (top, side, and front views)
        of the 3D staple reconstruction. This serves as a fallback when 3D 
        visualization is not available.
        
        Returns:
        --------
        filtered_tracks : list of dict
            List of dictionaries containing staple track data, including only staples
            with at least min_staple_points
        """
        print("Creating 2D projections of staples...")
        
        # Filter out staples with too few points
        filtered_tracks = [track for track in self.staple_tracks if len(track['points']) >= self.min_staple_points]
        
        # Create XY projection (top view)
        plt.figure(figsize=(12, 12))
        
        # Plot each staple
        for track in self.staple_tracks:
            points = np.array(track['points'])
            
            # Extract x, y coordinates
            y, x = points[:, 1], points[:, 2]
            
            # Determine color and style based on length
            if len(points) >= self.min_staple_points:
                color = 'blue'
                alpha = 0.7
                linewidth = 1.5
            else:
                color = 'gray'
                alpha = 0.3
                linewidth = 0.5
            
            # Plot points
            plt.scatter(x, y, s=10, color=color, alpha=alpha)
            
            # Connect with line
            plt.plot(x, y, '-', color=color, linewidth=linewidth, alpha=alpha)
        
        plt.title(f'Top View (XY Projection) - {len(filtered_tracks)} complete staples')
        plt.xlabel('X (pixels)')
        plt.ylabel('Y (pixels)')
        plt.gca().invert_yaxis()  # Match image coordinates (y increases downward)
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.output_dir, 'staples_xy_projection.png'), dpi=300)
        plt.close()
        
        # Create XZ projection (side view)
        plt.figure(figsize=(12, 8))
        
        for track in self.staple_tracks:
            points = np.array(track['points'])
            
            # Extract x, z coordinates
            z, x = points[:, 0], points[:, 2]
            
            # Determine color and style based on length
            if len(points) >= self.min_staple_points:
                color = 'blue'
                alpha = 0.7
                linewidth = 1.5
            else:
                color = 'gray'
                alpha = 0.3
                linewidth = 0.5
            
            # Plot points and line
            plt.scatter(x, z, s=10, color=color, alpha=alpha)
            plt.plot(x, z, '-', color=color, linewidth=linewidth, alpha=alpha)
        
        plt.title(f'Side View (XZ Projection) - {len(filtered_tracks)} complete staples')
        plt.xlabel('X (pixels)')
        plt.ylabel('Z (slice)')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.output_dir, 'staples_xz_projection.png'), dpi=300)
        plt.close()
        
        # Create YZ projection (front view)
        plt.figure(figsize=(12, 8))
        
        for track in self.staple_tracks:
            points = np.array(track['points'])
            
            # Extract y, z coordinates
            z, y = points[:, 0], points[:, 1]
            
            # Determine color and style
            if len(points) >= self.min_staple_points:
                color = 'blue'
                alpha = 0.7
                linewidth = 1.5
            else:
                color = 'gray'
                alpha = 0.3
                linewidth = 0.5
            
            # Plot points and line
            plt.scatter(y, z, s=10, color=color, alpha=alpha)
            plt.plot(y, z, '-', color=color, linewidth=linewidth, alpha=alpha)
        
        plt.title(f'Front View (YZ Projection) - {len(filtered_tracks)} complete staples')
        plt.xlabel('Y (pixels)')
        plt.ylabel('Z (slice)')
        plt.gca().invert_xaxis()  # Match image coordinates (y increases downward)
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.output_dir, 'staples_yz_projection.png'), dpi=300)
        plt.close()
        
        return filtered_tracks
    
    def export_staple_data(self):
        """
        Export staple data to CSV file.
        
        This function calculates various properties of each staple (length, cross-sectional
        area, etc.) and exports them to a CSV file for further analysis.
        
        Returns:
        --------
        df : pandas.DataFrame
            DataFrame containing staple data
        """
        # Create list to store staple data
        staple_data = []
        
        for track in self.staple_tracks:
            # Calculate staple properties
            points = np.array(track['points'])
            
            # Skip staples with too few points
            if len(points) < 5:
                continue
                
            # Calculate length (sum of Euclidean distances between consecutive points)
            length = 0
            for i in range(len(points) - 1):
                p1 = points[i, 1:]  # Skip the slice index
                p2 = points[i + 1, 1:]  # Skip the slice index
                length += np.sqrt(np.sum((p2 - p1) ** 2))
            
            # Calculate average cross-sectional area
            avg_area = np.mean(track['sizes'])
            
            # Calculate number of slices spanned
            slices_spanned = points[-1, 0] - points[0, 0] + 1
            
            # Add to data list
            staple_data.append({
                'staple_id': track['id'],
                'start_slice': int(points[0, 0]) + 1,
                'end_slice': int(points[-1, 0]) + 1,
                'slices_spanned': int(slices_spanned),
                'num_points': len(points),
                'length_pixels': length,
                'avg_area_pixels': avg_area,
            })
        
        # Create DataFrame
        df = pd.DataFrame(staple_data)
        
        # Sort by number of points (descending)
        df = df.sort_values(by='num_points', ascending=False).reset_index(drop=True)
        
        # Save to CSV
        csv_path = os.path.join(self.output_dir, 'staple_data.csv')
        df.to_csv(csv_path, index=False)
        print(f"Staple data saved to {csv_path}")
        
        # Print summary statistics for number of points
        if len(df) > 0:
            print(f"Staple statistics:")
            print(f"  Min points per staple: {df['num_points'].min()}")
            print(f"  Max points per staple: {df['num_points'].max()}")
            print(f"  Median points per staple: {df['num_points'].median()}")
            print(f"  Mean points per staple: {df['num_points'].mean():.1f}")
            print(f"  Staples with ≥20 points: {len(df[df['num_points'] >= 20])}/{len(df)} ({len(df[df['num_points'] >= 20])/len(df)*100:.1f}%)")
        
        return df
    
    def analyze_staple_shapes(self):
        """
        Analyze staple shapes and classify them based on their geometry.
        
        This function identifies complete staples (with at least min_staple_points points)
        and analyzes their shape characteristics, including curvature, straightness,
        and displacement patterns.
        
        Returns:
        --------
        staple_shapes : list of dict
            List of dictionaries containing shape analysis for each complete staple
        """
        print("Analyzing staple shapes...")
        
        # Filter staples by length first
        long_staples = []
        for track in self.staple_tracks:
            points = np.array(track['points'])
            
            # Only consider staples with sufficient points (at least 20)
            if len(points) >= 20:
                long_staples.append(track)
        
        print(f"Found {len(long_staples)}/{len(self.staple_tracks)} staples with at least 20 points")
        
        # Calculate shape characteristics for long staples
        staple_shapes = []
        for track in long_staples:
            points = np.array(track['points'])
            
            # Calculate various shape metrics
            # 1. Average displacement in x and y directions
            x_displacements = []
            y_displacements = []
            for i in range(len(points) - 1):
                y_disp = points[i+1, 1] - points[i, 1]  # y-displacement
                x_disp = points[i+1, 2] - points[i, 2]  # x-displacement
                x_displacements.append(x_disp)
                y_displacements.append(y_disp)
            
            avg_x_disp = np.mean(np.abs(x_displacements))
            avg_y_disp = np.mean(np.abs(y_displacements))
            
            # 2. Calculate curvature (using a basic approximation)
            curvature_sum = 0
            if len(points) >= 3:
                for i in range(1, len(points) - 1):
                    # Get three consecutive points
                    p0 = points[i-1, 1:]  # (y, x)
                    p1 = points[i, 1:]
                    p2 = points[i+1, 1:]
                    
                    # Calculate vectors
                    v1 = p1 - p0
                    v2 = p2 - p1
                    
                    # Calculate angle between vectors using dot product
                    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                    cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Ensure valid input for arccos
                    angle = np.arccos(cos_angle)
                    
                    curvature_sum += angle
                
                avg_curvature = curvature_sum / (len(points) - 2)
            else:
                avg_curvature = 0
            
            # 3. Calculate overall displacement (end-to-end vs path length)
            start_point = points[0, 1:]  # (y, x)
            end_point = points[-1, 1:]
            
            # End-to-end distance (direct line from start to end)
            end_to_end = np.sqrt(np.sum((end_point - start_point) ** 2))
            
            # Path length (sum of distances between consecutive points)
            path_length = 0
            for i in range(len(points) - 1):
                p1 = points[i, 1:]
                p2 = points[i + 1, 1:]
                path_length += np.sqrt(np.sum((p2 - p1) ** 2))
            
            # Straightness (1.0 is perfectly straight, lower values indicate more curvature)
            straightness = end_to_end / path_length if path_length > 0 else 1.0
            
            # Store the data
            staple_shapes.append({
                'id': track['id'],
                'points': points,
                'num_points': len(points),
                'slices_spanned': points[-1, 0] - points[0, 0] + 1,
                'avg_x_displacement': avg_x_disp,
                'avg_y_displacement': avg_y_disp,
                'avg_curvature': avg_curvature,
                'straightness': straightness,
                'end_to_end_distance': end_to_end,
                'path_length': path_length
            })
        
        # Save detailed analysis to CSV
        if staple_shapes:
            shape_data = [{
                'staple_id': s['id'],
                'num_points': s['num_points'],
                'slices_spanned': s['slices_spanned'],
                'avg_x_displacement': s['avg_x_displacement'],
                'avg_y_displacement': s['avg_y_displacement'],
                'avg_curvature': s['avg_curvature'],
                'straightness': s['straightness'],
                'end_to_end_distance': s['end_to_end_distance'],
                'path_length': s['path_length']
            } for s in staple_shapes]
            
            df = pd.DataFrame(shape_data)
            csv_path = os.path.join(self.output_dir, 'staple_shape_analysis.csv')
            df.to_csv(csv_path, index=False)
            print(f"Staple shape analysis saved to {csv_path}")
        
        # Create visualizations of the staple shapes
        if staple_shapes:
            self.visualize_staple_shapes(staple_shapes)
        
        return staple_shapes
    
    def visualize_staple_shapes(self, staple_shapes):
        """
        Create visualization of complete staple shapes.
        
        This function creates 3D visualizations of complete staples, color-coded
        by their shape characteristics (straightness).
        
        Parameters:
        -----------
        staple_shapes : list of dict
            List of dictionaries containing shape analysis for each complete staple
        """
        # Create 3D visualization
        fig = plt.figure(figsize=(14, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        # Create a colormap for staples based on their straightness
        # More straight = blue, more curved = red
        straightness_values = [s['straightness'] for s in staple_shapes]
        cmap = plt.cm.coolwarm
        
        # Plot each staple with all points visible
        for i, staple_info in enumerate(staple_shapes):
            # Extract points
            points = staple_info['points']
            
            # Get z, y, x coordinates
            z, y, x = points[:, 0], points[:, 1], points[:, 2]
            
            # Determine color based on straightness (normalized from 0-1)
            norm_straightness = (staple_info['straightness'] - min(straightness_values)) / \
                                (max(straightness_values) - min(straightness_values)) \
                                if max(straightness_values) > min(straightness_values) else 0.5
            
            color = cmap(1 - norm_straightness)  # Invert so curved (low straightness) is red
            
            # Plot ALL points along the staple to show bends
            ax.scatter(x, y, z, color=color, s=15, marker='o', alpha=0.7)
            
            # Connect points with line segments to better show bends
            ax.plot(x, y, z, '-', color=color, linewidth=1.0, alpha=0.7)
        
        # Set labels and title
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')
        ax.set_zlabel('Z (slice)')
        ax.set_title(f'Complete Staple Reconstructions (n={len(staple_shapes)})')
        
        # Add a colorbar to show the straightness scale
        sm = plt.cm.ScalarMappable(cmap=cmap)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Curvature (Red=High, Blue=Low)')
        
        # Save figure
        fig_path = os.path.join(self.output_dir, 'complete_staples.png')
        plt.savefig(fig_path, dpi=300)
        
        # Save additional views from different angles for better 3D understanding
        for i, angle in enumerate([(30, 45), (0, 0), (0, 90), (90, 0)]):
            ax.view_init(elev=angle[0], azim=angle[1])
            fig_path = os.path.join(self.output_dir, f'complete_staples_view{i+1}.png')
            plt.savefig(fig_path, dpi=300)
        
        # Try to save an interactive HTML version if mpld3 is available
        # try:
        #     from mpld3 import fig_to_html, save_html
        #     fig_path_html = os.path.join(self.output_dir, 'complete_staples_interactive.html')
        #     save_html(fig, fig_path_html)
        #     print(f"Interactive 3D visualization saved to {fig_path_html}")
        # except ImportError:
        #     print("mpld3 not available for interactive visualization. Install with: pip install mpld3")
        
        plt.close(fig)
        
        # Create a second visualization focusing on staples with the most complex shapes
        # Sort staples by curvature (least straight first)
        sorted_by_curve = sorted(staple_shapes, key=lambda s: s['straightness'])
        most_curved = sorted_by_curve[:min(10, len(sorted_by_curve))]
        
        if most_curved:
            fig = plt.figure(figsize=(14, 12))
            ax = fig.add_subplot(111, projection='3d')
            
            # Get a set of distinct colors
            colors = plt.cm.Set2.colors
            
            # Plot each curved staple with a different color
            for i, staple_info in enumerate(most_curved):
                # Extract points
                points = staple_info['points']
                
                # Get z, y, x coordinates
                z, y, x = points[:, 0], points[:, 1], points[:, 2]
                
                # Get color
                color = colors[i % len(colors)]
                
                # Plot ALL points to show bends
                ax.scatter(x, y, z, color=color, s=20, marker='o', alpha=0.7,
                          label=f"Staple {staple_info['id']} (straightness: {staple_info['straightness']:.2f})")
                
                # Connect with line
                ax.plot(x, y, z, '-', color=color, linewidth=1.5, alpha=0.7)
            
            # Set labels and title
            ax.set_xlabel('X (pixels)')
            ax.set_ylabel('Y (pixels)')
            ax.set_zlabel('Z (slice)')
            ax.set_title(f'Most Curved Staples (Showing Potential Bends)')
            ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
            
            # Adjust the figure layout to make room for the legend
            plt.tight_layout()
            
            # Save figure
            fig_path = os.path.join(self.output_dir, 'curved_staples.png')
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            
            # Save from a different angle to highlight bends
            ax.view_init(elev=20, azim=45)
            fig_path = os.path.join(self.output_dir, 'curved_staples_angle2.png')
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            
            plt.close(fig)
    
    def export_staple_key_points(self):
        """
        Export the key points (two endpoints and one midpoint) of each identified staple.
        
        For each staple, this function identifies:
        - The middle point
        - The two endpoints
        
        This specific format is designed to support downstream analysis for entanglement detection
        and other geometric analyses of the staple interactions.
        
        Returns:
        --------
        formatted_key_points : list of dict
            List of dictionaries containing key point data (staple_id, point_type, x, y, z)
        """
        print("Exporting staple key points (endpoints + midpoint) to CSV...")
        
        # Filter staples by length first
        long_staples = []
        for track in self.staple_tracks:
            points = np.array(track['points'])
            
            # Only consider staples with sufficient points
            if len(points) >= self.min_staple_points:
                long_staples.append(track)
        
        # Create a list to store the formatted key points data
        formatted_key_points = []
        
        for track in long_staples:
            staple_id = track['id']
            points = np.array(track['points'])
            
            # Identify key points (midpoint and two endpoints)
            start_idx = 0
            end_idx = len(points) - 1
            mid_idx = len(points) // 2
            
            # Get the key points
            start_point = points[start_idx]  # (slice, y, x)
            mid_point = points[mid_idx]      # (slice, y, x)
            end_point = points[end_idx]      # (slice, y, x)
            
            # Add midpoint to the formatted list
            formatted_key_points.append({
                'staple_id': staple_id,
                'point_type': 'middle',
                'x': float(mid_point[2]),    # x coordinate
                'y': float(mid_point[1]),    # y coordinate
                'z': float(mid_point[0])     # z coordinate (slice)
            })
            
            # Add first endpoint (start point)
            formatted_key_points.append({
                'staple_id': staple_id,
                'point_type': 'end1',
                'x': float(start_point[2]),  # x coordinate
                'y': float(start_point[1]),  # y coordinate
                'z': float(start_point[0])   # z coordinate (slice)
            })
            
            # Add second endpoint (end point)
            formatted_key_points.append({
                'staple_id': staple_id,
                'point_type': 'end2',
                'x': float(end_point[2]),    # x coordinate
                'y': float(end_point[1]),    # y coordinate
                'z': float(end_point[0])     # z coordinate (slice)
            })
        
        # Save key points to CSV
        if formatted_key_points:
            df_key_points = pd.DataFrame(formatted_key_points)
            key_points_csv = os.path.join(self.output_dir, 'staple_key_points.csv')
            df_key_points.to_csv(key_points_csv, index=False)
            print(f"Staple key points (endpoints + midpoint) saved to {key_points_csv}")
            print(f"Total staples: {len(long_staples)}, Total key points: {len(formatted_key_points)}")
            
            return formatted_key_points
        
        return []

    def calculate_staple_lengths(self):
        """
        Calculate the length of each identified staple.
        
        This function computes the length of each staple by finding the two farthest points
        within the staple, providing an estimate of the actual physical length.
        
        Returns:
        --------
        lengths : list
            List of staple lengths in pixels
        """
        lengths = []
        for staple_id, staple_points in enumerate(self.all_staples):
            if len(staple_points) < self.min_staple_points:
                continue
                
            # Convert to numpy array for easier operations
            points = np.array(staple_points)
            
            # Find the two farthest points (endpoints of the staple)
            max_dist = 0
            endpoints = None
            
            for i in range(len(points)):
                for j in range(i+1, len(points)):
                    dist = np.sqrt(np.sum((points[i] - points[j])**2))
                    if dist > max_dist:
                        max_dist = dist
                        endpoints = (points[i], points[j])
            
            lengths.append(max_dist)
        
        return lengths

    def analyze_staple_statistics(self):
        """
        Calculate and display statistics about the identified staples.
        
        This function computes various statistics about staple lengths and,
        if calibration information is available, converts pixel measurements
        to real-world units (inches).
        
        Returns:
        --------
        tuple
            Tuple containing (lengths, avg_length, conversion_factor)
        """
        # Calculate lengths of all staples
        lengths = self.calculate_staple_lengths()
        
        if not lengths:
            print("No staples identified for statistical analysis.")
            return
        
        # Calculate statistics
        avg_length = np.mean(lengths)
        std_length = np.std(lengths)
        min_length = np.min(lengths)
        max_length = np.max(lengths)
        
        print(f"Staple Length Statistics:")
        print(f"Number of staples analyzed: {len(lengths)}")
        print(f"Average length: {avg_length:.2f} pixels")
        print(f"Standard deviation: {std_length:.2f} pixels")
        print(f"Minimum length: {min_length:.2f} pixels")
        print(f"Maximum length: {max_length:.2f} pixels")
        
        # Create a histogram of staple lengths
        plt.figure(figsize=(10, 6))
        plt.hist(lengths, bins=20, color='skyblue', edgecolor='black')
        plt.xlabel('Staple Length (pixels)')
        plt.ylabel('Frequency')
        plt.title('Distribution of Staple Lengths')
        plt.grid(alpha=0.3)
        
        # Save the histogram
        plt.savefig(os.path.join(self.output_dir, 'staple_length_histogram.png'))
        plt.close()
        
        # Convert to real units using standard staple length (0.55 inches)
        # This is an approximation based on typical office staple dimensions
        real_staple_length = 0.55  # in inches
        conversion_factor = real_staple_length / avg_length  # inches per pixel
        
        print(f"\nAssuming real staples are {real_staple_length} inches:")
        print(f"Conversion factor: {conversion_factor:.6f} inches/pixel")
        print(f"Average length: {avg_length * conversion_factor:.2f} inches")
        
        # Save statistics to a CSV file
        stats_df = pd.DataFrame({
            'Statistic': ['Number of Staples', 'Average Length (px)', 'Standard Deviation (px)', 
                        'Min Length (px)', 'Max Length (px)', 'Conversion Factor (in/px)',
                        'Average Length (in)'],
            'Value': [len(lengths), avg_length, std_length, min_length, max_length, 
                    conversion_factor, avg_length * conversion_factor]
        })
        
        stats_df.to_csv(os.path.join(self.output_dir, 'staple_statistics.csv'), index=False)
        
        return lengths, avg_length, conversion_factor
    
    def run(self):
        """
        Run the complete reconstruction pipeline.
        
        This is the main method that executes the entire pipeline from
        loading the slices to analyzing and visualizing the staples.
        """
        # Load slices
        self.load_slices()
        
        # Process slices and track staples
        self.process_slices()
        
        # Create 3D visualization of all staples
        self.visualize_3d_staples()
        
        # Export staple data
        self.export_staple_data()
        
        # Analyze staple shapes (focusing on staples with at least min_staple_points)
        self.analyze_staple_shapes()
        
        # Export the key points (endpoints + midpoint) to CSV
        self.export_staple_key_points()

        # Calculate staple lengths and analyze statistics
        self.calculate_staple_lengths()
        self.analyze_staple_statistics()
        
        print("Staple reconstruction complete!")
        print(f"Results saved to {self.output_dir}")


def main():
    """
    Main function to parse command-line arguments and run the staple reconstruction.
    
    This function handles all user input, sets up the reconstructor with the
    appropriate parameters, and executes the reconstruction.
    """
    print("===== 3D Staple Reconstruction from Micro-CT Slices =====")
    
    # Set up command-line argument parser
    parser = argparse.ArgumentParser(description='Reconstruct 3D staples from micro-CT slices')
    parser.add_argument('--input', type=str, help='Path to multi-page TIF file')
    parser.add_argument('--output', type=str, help='Path to output directory')
    parser.add_argument('--threshold', type=int, default=156, help='Threshold for binarization (0-255)')
    parser.add_argument('--max-distance', type=float, default=10, help='Maximum distance between centroids')
    parser.add_argument('--min-size', type=int, default=5, help='Minimum size of regions to consider')
    parser.add_argument('--min-staple-points', type=int, default=20, help='Minimum number of points for a complete staple')
    
    args = parser.parse_args()
    
    # If parameters not provided through command line, prompt the user
    input_tif = args.input
    if not input_tif:
        input_tif = input("Enter the path to the multi-page TIF file: ")
    
    output_dir = args.output
    if not output_dir:
        output_dir = input("Enter the path for the output directory: ")
    
    threshold = args.threshold
    threshold_input = input(f"Enter the threshold value for binarization (0-255) [default: {threshold}]: ")
    if threshold_input.strip():
        threshold = int(threshold_input)
    
    max_distance = args.max_distance
    max_dist_input = input(f"Enter the maximum distance between centroids [default: {max_distance}]: ")
    if max_dist_input.strip():
        max_distance = float(max_dist_input)
    
    min_size = args.min_size
    min_size_input = input(f"Enter the minimum region size in pixels [default: {min_size}]: ")
    if min_size_input.strip():
        min_size = int(min_size_input)
    
    min_staple_points = args.min_staple_points
    min_points_input = input(f"Enter the minimum number of points for a complete staple [default: {min_staple_points}]: ")
    if min_points_input.strip():
        min_staple_points = int(min_points_input)
    
    try:
        # Create and run the reconstructor
        reconstructor = StapleReconstructor(
            input_tif=input_tif,
            output_dir=output_dir,
            threshold=threshold,
            max_distance=max_distance,
            min_size=min_size,
            min_staple_points=min_staple_points
        )
        
        reconstructor.run()
        
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
        print("Please check your input file and parameters and try again.")


if __name__ == "__main__":
    main()