import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import csv

def generate_staple(center_point, orientation, size_variation=0.2, noise=0.05):
    """
    Generate a U-shaped staple with 3 representative points:
    - middle concave point
    - two end points
    
    Parameters:
    -----------
    center_point : tuple (x, y, z)
        Center position of the staple
    orientation : tuple (theta, phi)
        Orientation angles in radians (theta: azimuthal, phi: polar)
    size_variation : float
        Random variation in staple size (0.0-1.0)
    noise : float
        Random noise to add to points (0.0-1.0)
        
    Returns:
    --------
    dict: Dictionary with representative points (middle, end1, end2)
    """
    # Base dimensions for a staple (can be adjusted)
    base_length = 3.0 * (1 + random.uniform(-size_variation, size_variation))
    base_width = 6.0 * (1 + random.uniform(-size_variation, size_variation))
    
    # Create rotation matrix based on orientation
    theta, phi = orientation
    
    # Calculate rotation matrix components
    sin_theta, cos_theta = np.sin(theta), np.cos(theta)
    sin_phi, cos_phi = np.sin(phi), np.cos(phi)
    
    # Define rotation matrix
    R = np.array([
        [cos_theta * cos_phi, -sin_theta, cos_theta * sin_phi],
        [sin_theta * cos_phi, cos_theta, sin_theta * sin_phi],
        [-sin_phi, 0, cos_phi]
    ])
    
    # Create the three points (in local coordinates before rotation)
    # Middle point (bottom of the U)
    middle = np.array([0, 0, 0])
    
    # End points
    end1 = np.array([-base_width/2, 0, base_length])
    end2 = np.array([base_width/2, 0, base_length])
    
    # Apply rotation and translation
    middle_rotated = R @ middle + np.array(center_point)
    end1_rotated = R @ end1 + np.array(center_point)
    end2_rotated = R @ end2 + np.array(center_point)
    
    # Add some random noise
    middle_noisy = middle_rotated + np.random.normal(0, noise, 3)
    end1_noisy = end1_rotated + np.random.normal(0, noise, 3)
    end2_noisy = end2_rotated + np.random.normal(0, noise, 3)
    
    return {
        "middle": middle_noisy.tolist(),
        "end1": end1_noisy.tolist(),
        "end2": end2_noisy.tolist()
    }

def generate_dataset(num_staples=50, volume_size=(20, 20, 20)):
    """
    Generate a dataset of random U-shaped staples within a given volume
    
    Parameters:
    -----------
    num_staples : int
        Number of staples to generate
    volume_size : tuple (x_size, y_size, z_size)
        Size of the volume in which to place staples
        
    Returns:
    --------
    list: List of staple dictionaries
    """
    staples = []
    
    for i in range(num_staples):
        # Random position
        x = random.uniform(0, volume_size[0])
        y = random.uniform(0, volume_size[1])
        z = random.uniform(0, volume_size[2])
        
        # Random orientation
        theta = random.uniform(0, 2 * np.pi)
        phi = random.uniform(0, np.pi)
        
        staple = generate_staple(
            center_point=(x, y, z),
            orientation=(theta, phi)
        )
        
        # Add staple ID
        staple["id"] = i
        staples.append(staple)
    
    return staples

def visualize_staples(staples, volume_size=(20, 20, 20)):
    """
    Visualize the staples in 3D
    
    Parameters:
    -----------
    staples : list
        List of staple dictionaries
    volume_size : tuple
        Size of the bounding volume
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    for i, staple in enumerate(staples):
        middle = np.array(staple["middle"])
        end1 = np.array(staple["end1"])
        end2 = np.array(staple["end2"])
        
        # Generate random color for the staple
        color = np.random.rand(3)
        
        # Plot the three points
        ax.scatter(*middle, color=color, marker='o', s=20, label=f'Staple {staple["id"]}' if i == 0 else "")
        ax.scatter(*end1, color=color, marker='^', s=20)
        ax.scatter(*end2, color=color, marker='^', s=20)
        
        # Connect the points to represent the staple shape
        ax.plot([end1[0], middle[0], end2[0]], 
                [end1[1], middle[1], end2[1]], 
                [end1[2], middle[2], end2[2]], 
                color=color, alpha=0.7)
    
    # Set axis labels and limits
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim(0, volume_size[0])
    ax.set_ylim(0, volume_size[1])
    ax.set_zlim(0, volume_size[2])
    ax.set_title('Simulated U-Shaped Staples')
    
    # Improve the legend (show only one entry)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='best')
    
    plt.tight_layout()
    plt.show()

def save_staples_to_file(staples, filename="/Users/lorenabritton/Desktop/stats/fake_staples/volume_1400_1800_100/fake_staples_500.csv"):
    """
    Save the generated staples to a CSV file
    
    Parameters:
    -----------
    staples : list
        List of staple dictionaries
    filename : str
        Output filename
    """
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        # Write header
        writer.writerow(['staple_id', 'point_type', 'x', 'y', 'z'])
        
        # Write data for each staple
        for staple in staples:
            sid = staple["id"]
            # Middle point
            writer.writerow([
                sid, 'middle', 
                f"{staple['middle'][0]:.6f}", 
                f"{staple['middle'][1]:.6f}", 
                f"{staple['middle'][2]:.6f}"
            ])
            # End point 1
            writer.writerow([
                sid, 'end1', 
                f"{staple['end1'][0]:.6f}", 
                f"{staple['end1'][1]:.6f}", 
                f"{staple['end1'][2]:.6f}"
            ])
            # End point 2
            writer.writerow([
                sid, 'end2', 
                f"{staple['end2'][0]:.6f}", 
                f"{staple['end2'][1]:.6f}", 
                f"{staple['end2'][2]:.6f}"
            ])
    print(f"Saved {len(staples)} staples (total {len(staples)*3} points) to {filename}")

def simulate_cross_sections(staples, num_slices=20, volume_size=(20, 20, 20)):
    """
    Simulate cross-sections of the staples to mimic the tif stack.
    This creates a visualization of what the cross-sections might look like.
    
    Parameters:
    -----------
    staples : list
        List of staple dictionaries
    num_slices : int
        Number of slices to generate
    volume_size : tuple
        Size of the volume
    """
    # Define slice thickness
    slice_thickness = volume_size[2] / num_slices
    
    # Create a figure for the slices
    fig, axes = plt.subplots(4, 5, figsize=(15, 12))
    axes = axes.flatten()
    
    for slice_idx in range(min(num_slices, len(axes))):
        ax = axes[slice_idx]
        z_min = slice_idx * slice_thickness
        z_max = (slice_idx + 1) * slice_thickness
        
        ax.set_xlim(0, volume_size[0])
        ax.set_ylim(0, volume_size[1])
        ax.set_title(f'Slice {slice_idx}: z = {z_min:.1f}-{z_max:.1f}')
        
        # Check each staple for intersection with this slice
        for staple in staples:
            middle = np.array(staple["middle"])
            end1 = np.array(staple["end1"])
            end2 = np.array(staple["end2"])
            
            # Create a color for this staple
            color = np.random.rand(3)
            
            # Check if any point is in this slice
            for point in [middle, end1, end2]:
                if z_min <= point[2] <= z_max:
                    ax.scatter(point[0], point[1], color=color, s=30, alpha=0.7)
            
            # Check for line intersections with the slice
            for p1, p2 in [(middle, end1), (middle, end2)]:
                if (p1[2] <= z_max and p2[2] >= z_min) or (p1[2] >= z_min and p2[2] <= z_max):
                    # Calculate intersection point with the slice (simple linear interpolation)
                    if p1[2] != p2[2]:  # Avoid division by zero
                        t = (z_min + slice_thickness/2 - p1[2]) / (p2[2] - p1[2])
                        if 0 <= t <= 1:  # Intersection is within the line segment
                            x = p1[0] + t * (p2[0] - p1[0])
                            y = p1[1] + t * (p2[1] - p1[1])
                            ax.scatter(x, y, color=color, s=50, alpha=0.7)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Generate a set of random staples
    volume_size = (1400, 1800, 100)
    staples = generate_dataset(num_staples=500, volume_size=volume_size)
    
    # Visualize the staples in 3D
    visualize_staples(staples, volume_size)
    
    # Simulate cross-sections (like the tif stack)
    simulate_cross_sections(staples, num_slices=100, volume_size=volume_size)
    
    # Save the staples to a CSV file
    save_staples_to_file(staples)
    
    # Display a sample staple's coordinates
    sample_staple = staples[0]
    print(f"\nSample Staple (ID: {sample_staple['id']})")
    print(f"Middle point: {sample_staple['middle']}")
    print(f"End point 1: {sample_staple['end1']}")
    print(f"End point 2: {sample_staple['end2']}")
    
    # Print the first few rows of the CSV format as a preview
    print("\nPreview of CSV file format:")
    print("staple_id,point_type,x,y,z")
    for staple in staples[:2]:  # Show first 2 staples as examples
        sid = staple["id"]
        print(f"{sid},middle,{staple['middle'][0]:.6f},{staple['middle'][1]:.6f},{staple['middle'][2]:.6f}")
        print(f"{sid},end1,{staple['end1'][0]:.6f},{staple['end1'][1]:.6f},{staple['end1'][2]:.6f}")
        print(f"{sid},end2,{staple['end2'][0]:.6f},{staple['end2'][1]:.6f},{staple['end2'][2]:.6f}")
    
    print(f"\nFull dataset with {len(staples)} staples saved to fake_staples_500.csv")