import csv
import numpy as np
from itertools import combinations

# Read CSV file in the new format
def read_staples_from_csv(csv_file):
    """
    Read staples from CSV file with format:
    staple_id, point_type, x, y, z
    
    Parameters:
    -----------
    csv_file : str
        Path to the CSV file
        
    Returns:
    --------
    list: List of staple dictionaries with all three points
    """
    # Dictionary to store staples temporarily during parsing
    staples_dict = {}
    
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header row
        
        for row in reader:
            # Parse the row
            staple_id = int(row[0])
            point_type = row[1]
            x = float(row[2])
            y = float(row[3])
            z = float(row[4])
            
            # Create staple entry if it doesn't exist
            if staple_id not in staples_dict:
                staples_dict[staple_id] = {"id": staple_id}
            
            # Add point data based on point_type
            if point_type == 'middle':
                staples_dict[staple_id]["middle"] = [x, y, z]
            elif point_type == 'end1':
                staples_dict[staple_id]["end1"] = [x, y, z]
            elif point_type == 'end2':
                staples_dict[staple_id]["end2"] = [x, y, z]
    
    # Convert dictionary to list and filter out incomplete staples
    staples = []
    for staple_id, staple_data in staples_dict.items():
        if "middle" in staple_data and "end1" in staple_data and "end2" in staple_data:
            staples.append(staple_data)
        else:
            print(f"Warning: Staple ID {staple_id} is incomplete and will be skipped.")
    
    print(f"Successfully loaded {len(staples)} complete staples from {csv_file}")
    return staples

# Check if staples are entangled
def are_entangled(staple_a, staple_b, tolerance=0.1):
    """
    Check if two staples are entangled by checking for bounding box overlap
    
    Parameters:
    -----------
    staple_a, staple_b : dict
        Staple dictionaries containing middle, end1, and end2 points
    tolerance : float
        Tolerance for overlap detection
        
    Returns:
    --------
    bool: True if staples are potentially entangled
    """
    # Get all points for staple A
    points_a = np.array([staple_a["middle"], staple_a["end1"], staple_a["end2"]])
    
    # Get all points for staple B
    points_b = np.array([staple_b["middle"], staple_b["end1"], staple_b["end2"]])
    
    # Check for bounding box overlap in each dimension
    for dim in range(3):  # x, y, z dimensions
        min_a = np.min(points_a[:, dim])
        max_a = np.max(points_a[:, dim])
        min_b = np.min(points_b[:, dim])
        max_b = np.max(points_b[:, dim])
        
        # If bounding boxes don't overlap in any dimension, staples cannot be entangled
        if max_a < min_b - tolerance or min_a > max_b + tolerance:
            return False
    
    # If we get here, bounding boxes overlap in all dimensions
    return True

# Find all entangled pairs of staples
def find_entangled_pairs(staples, tolerance=0.1):
    """
    Find all pairs of entangled staples
    
    Parameters:
    -----------
    staples : list
        List of staple dictionaries
    tolerance : float
        Tolerance for overlap detection
        
    Returns:
    --------
    list: List of tuples containing pairs of entangled staple IDs
    """
    entangled_pairs = []
    for staple_a, staple_b in combinations(staples, 2):
        if are_entangled(staple_a, staple_b, tolerance):
            entangled_pairs.append((staple_a["id"], staple_b["id"]))
    
    return entangled_pairs

# Save entangled pairs to a CSV file
def save_entangled_pairs(entangled_pairs, output_file="/Users/lorenabritton/Desktop/stats/entangled_pairs.csv"):
    """
    Save entangled pairs to a CSV file
    
    Parameters:
    -----------
    entangled_pairs : list
        List of tuples containing pairs of entangled staple IDs
    output_file : str
        Output filename
    """
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        # Write header
        writer.writerow(['staple_id_1', 'staple_id_2'])
        
        # Write data
        for pair in entangled_pairs:
            writer.writerow([pair[0], pair[1]])
    
    print(f"Saved {len(entangled_pairs)} entangled pairs to {output_file}")

# Calculate entanglement statistics
def calculate_entanglement_stats(staples, entangled_pairs):
    """
    Calculate statistics about the entanglements
    
    Parameters:
    -----------
    staples : list
        List of staple dictionaries
    entangled_pairs : list
        List of tuples containing pairs of entangled staple IDs
        
    Returns:
    --------
    dict: Dictionary with entanglement statistics
    """
    # Count entanglements per staple
    entanglements_per_staple = {}
    for pair in entangled_pairs:
        for staple_id in pair:
            if staple_id not in entanglements_per_staple:
                entanglements_per_staple[staple_id] = 0
            entanglements_per_staple[staple_id] += 1
    
    # Calculate statistics
    total_staples = len(staples)
    entangled_staples = len(entanglements_per_staple)
    non_entangled_staples = total_staples - entangled_staples
    max_entanglements = max(entanglements_per_staple.values()) if entanglements_per_staple else 0
    avg_entanglements = sum(entanglements_per_staple.values()) / entangled_staples if entangled_staples > 0 else 0
    
    stats = {
        "total_staples": total_staples,
        "entangled_pairs": len(entangled_pairs),
        "entangled_staples": entangled_staples,
        "non_entangled_staples": non_entangled_staples,
        "percent_entangled": (entangled_staples / total_staples) * 100 if total_staples > 0 else 0,
        "max_entanglements": max_entanglements,
        "avg_entanglements": avg_entanglements,
        "entanglements_per_staple": entanglements_per_staple
    }
    
    return stats

# Main function
if __name__ == "__main__":
    # File paths
    input_file = "/Users/lorenabritton/Thesis/A_USE/3-18-25/reconstruct_results_3/staple_key_points.csv" 
    output_file = "/Users/lorenabritton/Desktop/stats/staples/reconstruct_3_entangled_pairs.csv"
    
    # Read staples from CSV
    staples = read_staples_from_csv(input_file)
    
    # Find entangled pairs
    tolerance = 0.1  # Set overlap tolerance
    entangled_pairs = find_entangled_pairs(staples, tolerance)
    
    # Print entangled pairs
    print(f"\nFound {len(entangled_pairs)} entangled pairs:")
    for i, pair in enumerate(entangled_pairs[:10]):  # Show first 10 pairs
        print(f"  {pair[0]} <-> {pair[1]}")
    if len(entangled_pairs) > 10:
        print(f"  ... and {len(entangled_pairs) - 10} more pairs")
    
    # Save entangled pairs to CSV
    save_entangled_pairs(entangled_pairs, output_file)
    
    # Calculate and print statistics
    stats = calculate_entanglement_stats(staples, entangled_pairs)
    print("\nEntanglement Statistics:")
    print(f"  Total staples: {stats['total_staples']}")
    print(f"  Entangled pairs: {stats['entangled_pairs']}")
    print(f"  Entangled staples: {stats['entangled_staples']} ({stats['percent_entangled']:.1f}%)")
    print(f"  Non-entangled staples: {stats['non_entangled_staples']}")
    print(f"  Maximum entanglements per staple: {stats['max_entanglements']}")
    print(f"  Average entanglements per entangled staple: {stats['avg_entanglements']:.2f}")
    
    # Identify most entangled staples
    if entangled_pairs:
        print("\nMost entangled staples:")
        sorted_staples = sorted(stats['entanglements_per_staple'].items(), 
                               key=lambda x: x[1], reverse=True)
        for i, (staple_id, count) in enumerate(sorted_staples[:5]):  # Show top 5
            print(f"  Staple {staple_id}: {count} entanglements")


