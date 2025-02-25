import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import tifffile
import matplotlib.pyplot as plt
import logging
from typing import List, Tuple, Optional, Union
import os
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_3d_shepp_logan_phantom(size: int = 256) -> torch.Tensor:
    """Create a 3D Shepp-Logan-like phantom.
    
    Args:
        size: Size of the cube (size x size x size)
        
    Returns:
        3D tensor of shape [size, size, size]
    """
    phantom = torch.zeros((size, size, size))
    center = size // 2
    
    # Create coordinate grids
    z, y, x = torch.meshgrid(
        torch.arange(size),
        torch.arange(size),
        torch.arange(size),
        indexing='ij'
    )
    
    # Create outer ellipsoid
    a, b, c = size//3, size//4, size//3
    mask = ((x - center)**2 / a**2 + (y - center)**2 / b**2 + (z - center)**2 / c**2) <= 1
    phantom[mask] = 1.0
    
    # Add some internal features (smaller ellipsoid)
    a, b, c = size//6, size//8, size//6
    mask = ((x - center - size//8)**2 / a**2 + (y - center)**2 / b**2 + (z - center)**2 / c**2) <= 1
    phantom[mask] = 0.5
    
    # Add another internal feature
    a, b, c = size//10, size//10, size//10
    mask = ((x - center + size//8)**2 / a**2 + (y - center + size//8)**2 / b**2 + (z - center)**2 / c**2) <= 1
    phantom[mask] = 0.8
    
    return phantom

def create_parallel_beam_projections(
    phantom: torch.Tensor, 
    angles: torch.Tensor,
    device: torch.device
) -> torch.Tensor:
    """Create parallel beam projections of a 3D phantom.
    
    For parallel beam tomography, we rotate the volume around the z-axis
    and project along the y-axis for each angle.
    
    Args:
        phantom: 3D phantom tensor [depth, height, width]
        angles: Rotation angles in radians (around z-axis)
        device: Torch device
        
    Returns:
        Projections tensor [num_angles, depth, width]
    """
    # Move phantom to device and get dimensions
    phantom = phantom.to(device)
    depth, height, width = phantom.shape
    num_angles = len(angles)
    angles = angles.to(device)
    
    # For parallel beam along y-axis, projections will be [num_angles, depth, width]
    projections = torch.zeros((num_angles, depth, width), device=device)
    
    # Process each z-slice separately to save memory
    pbar = tqdm(range(depth), desc="Processing slices", unit="slice")
    for z in pbar:
        # Get the current 2D slice
        slice_2d = phantom[z].to(device)  # [height, width]
        
        # Process each angle
        for i, angle in enumerate(angles):
            # Rotate the slice
            cos_a = torch.cos(angle)
            sin_a = torch.sin(angle)
            
            # Create rotation grid
            y_grid, x_grid = torch.meshgrid(
                torch.linspace(-1, 1, height),
                torch.linspace(-1, 1, width),
                indexing='ij'
            )
            y_grid, x_grid = y_grid.to(device), x_grid.to(device)
            
            # Rotate coordinates
            x_rot = x_grid * cos_a - y_grid * sin_a
            y_rot = x_grid * sin_a + y_grid * cos_a
            
            # Create sampling grid
            grid = torch.stack([x_rot, y_rot], dim=-1).unsqueeze(0)  # [1, height, width, 2]
            
            # Rotate slice using grid_sample
            rotated = F.grid_sample(
                slice_2d.unsqueeze(0).unsqueeze(0),  # [1, 1, height, width]
                grid,
                mode='bilinear',
                padding_mode='zeros',
                align_corners=True
            )[0, 0]  # [height, width]
            
            # Sum along y-axis (height) to get projection at this angle for this slice
            projection_line = rotated.sum(dim=0)  # [width]
            
            # Store in projections tensor
            projections[i, z] = projection_line
    
    return projections

def save_projections_as_tiff(
    projections: torch.Tensor,
    output_dir: Path,
    angles: torch.Tensor
) -> List[Path]:
    """Save projections as TIFF files.
    
    Args:
        projections: Projections tensor [num_angles, depth, width]
        output_dir: Directory to save TIFF files
        angles: Angles used for projections (in radians)
        
    Returns:
        List of paths to saved TIFF files
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    
    # Normalize for better visualization
    min_val = projections.min()
    max_val = projections.max()
    normalized = (projections - min_val) / (max_val - min_val)
    
    # Save projections as TIFF files
    pbar = tqdm(enumerate(angles), desc="Saving projections", total=len(angles), unit="proj")
    for i, angle in pbar:
        # Format angle in degrees for filename
        angle_deg = np.degrees(angle.item())
        
        # Create filename
        filename = f"projection_{i:03d}_angle{angle_deg:.1f}.tiff"
        file_path = output_dir / filename
        
        # Convert to uint16 for TIFF
        proj_np = (normalized[i].cpu().numpy() * 65535).astype(np.uint16)
        
        # Save as TIFF
        tifffile.imwrite(str(file_path), proj_np)
        paths.append(file_path)
        
        # Update progress bar postfix with current angle
        pbar.set_postfix({"angle": f"{angle_deg:.1f}°"})
    
    # Save angles as a numpy file for reference
    angles_np = angles.cpu().numpy()
    np.save(output_dir / "projection_angles.npy", angles_np)
    
    return paths

def main():
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Generate 3D phantom dataset for tomographic reconstruction')
    parser.add_argument('--size', type=int, default=256,
                        help='Size of the 3D phantom (size x size x size)')
    parser.add_argument('--angles', type=int, default=180,
                        help='Number of projection angles')
    args = parser.parse_args()
    
    # Set up device - prefer MPS for Apple Silicon
    device = (
        torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    logger.info(f"Using device: {device}")
    
    # Parameters from command line
    size = args.size
    num_angles = args.angles
    
    logger.info(f"Generating phantom with size={size}³ and {num_angles} projection angles")
    
    # Create organized output directory structure
    base_dir = Path("test_data")
    dataset_name = f"phantom_3d_{size}_{num_angles}angles"
    output_dir = base_dir / dataset_name
    projections_dir = output_dir / "projections"
    phantom_dir = output_dir / "phantom"
    
    # Create directories
    for dir_path in [output_dir, projections_dir, phantom_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Create phantom
    logger.info(f"Creating 3D phantom of size {size}×{size}×{size}...")
    phantom = create_3d_shepp_logan_phantom(size)
    
    # Generate angles (0 to 180 degrees)
    angles = torch.linspace(0, np.pi, num_angles)
    
    # Create projections
    logger.info(f"Creating {num_angles} parallel beam projections...")
    projections = create_parallel_beam_projections(phantom, angles, device)
    
    # Save projections as TIFF files
    logger.info("Saving projections and phantom slices...")
    tiff_paths = save_projections_as_tiff(projections, projections_dir, angles)
    
    # Save the phantom as a set of TIFF slices
    pbar = tqdm(range(size), desc="Saving phantom slices", unit="slice")
    for z in pbar:
        slice_path = phantom_dir / f"slice_{z:03d}.tiff"
        # Normalize and convert to uint16
        slice_np = phantom[z].cpu().numpy()
        slice_np = ((slice_np - slice_np.min()) / (slice_np.max() - slice_np.min()) * 65535).astype(np.uint16)
        tifffile.imwrite(str(slice_path), slice_np)
    
    # Visualize example projections with improved aesthetics
    plt.style.use('dark_background')
    
    # Create figure with dark background
    fig = plt.figure(figsize=(15, 10), dpi=150, facecolor='black')
    gs = plt.GridSpec(2, 3, figure=fig, wspace=0.15, hspace=0.2)
    
    # Select key angles to display
    indices = [0, num_angles//5, 2*num_angles//5, 3*num_angles//5, 4*num_angles//5, num_angles-1]
    
    for i, idx in enumerate(indices):
        if idx < len(projections):
            ax = fig.add_subplot(gs[i//3, i%3])
            
            # Get projection data and enhance contrast
            proj = projections[idx].cpu().numpy()
            p2, p98 = np.percentile(proj, (2, 98))
            proj = np.clip(proj, p2, p98)
            
            # Display with improved aesthetics
            im = ax.imshow(proj, cmap='gray')
            angle_deg = np.degrees(angles[idx].item())
            ax.set_title(f"θ = {angle_deg:.1f}°", fontsize=12, pad=10, color='white')
            ax.axis('off')
    
    # Add main title
    fig.suptitle('Example Projections at Different Angles', 
                 fontsize=14, color='white', y=0.95)
    
    # Save with proper background
    plt.savefig(output_dir / "example_projections.png", 
                dpi=150, facecolor='black', edgecolor='none', 
                bbox_inches='tight', pad_inches=0.2)
    
    logger.info(f"Done! Generated {len(tiff_paths)} projection files in {projections_dir}")
    logger.info(f"Saved phantom slices in {phantom_dir}")
    logger.info(f"Example projections visualization saved to {output_dir}/example_projections.png")

if __name__ == "__main__":
    main()