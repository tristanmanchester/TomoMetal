"""Test module for 3D tomographic reconstruction using synthetic data.

This module tests the 3D reconstruction functionality by:
1. Checking for existing test data in test_data/
2. Generating new test data if none exists (by running generate_3d_dataset.py)
3. Loading the projections and performing 3D reconstruction
4. Saving visualization results for comparison
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from tomometal.reconstruction import TomographicReconstructor
import logging
from pathlib import Path
import subprocess
import tifffile
from typing import List, Tuple
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def ensure_test_data_exists(size: int = 256, num_angles: int = 180) -> Path:
    """Ensure 3D test data exists with specified parameters, generate it if not.
    
    Args:
        size: Size of the phantom cube (size x size x size)
        num_angles: Number of projection angles
        
    Returns:
        Path to the dataset directory
    """
    base_dir = Path("test_data")
    target_dir = base_dir / f"phantom_3d_{size}_{num_angles}angles"
    
    if not target_dir.exists():
        logger.info(f"Generating new test data (size={size}³, angles={num_angles})...")
        subprocess.run(
            ["python", "tests/generate_3d_dataset.py", 
             f"--size={size}", f"--angles={num_angles}"],
            check=True
        )
        
    if not target_dir.exists():
        raise RuntimeError("Failed to generate test data")
        
    return target_dir

def load_projections(proj_dir: Path) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load projections and angles from the test data directory."""
    # Load angles
    angles = torch.tensor(np.load(proj_dir.parent / "projections" / "projection_angles.npy"))
    
    # Load all projection files
    proj_files = sorted(proj_dir.glob("projection_*.tiff"))
    projections = []
    
    for f in proj_files:
        proj = tifffile.imread(f)
        projections.append(torch.from_numpy(proj.astype(np.float32)))
    
    return torch.stack(projections), angles

def main():
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Test 3D tomographic reconstruction')
    parser.add_argument('--size', type=int, default=256,
                        help='Size of the phantom cube (size x size x size)')
    parser.add_argument('--angles', type=int, default=180,
                        help='Number of projection angles')
    args = parser.parse_args()
    
    # Initialize reconstructor
    reconstructor = TomographicReconstructor()
    logger.info(f"Using device: {reconstructor.device}")
    
    # Ensure test data exists with specified parameters
    data_dir = ensure_test_data_exists(args.size, args.angles)
    logger.info(f"Using test data from: {data_dir}")
    
    # Load projections and angles
    proj_dir = data_dir / "projections"
    projections, angles = load_projections(proj_dir)
    
    # Move data to device
    projections = projections.to(reconstructor.device)
    angles = angles.to(reconstructor.device)
    
    # Get middle slice for 2D reconstruction test
    middle_slice_idx = projections.shape[1] // 2
    middle_slice = projections[:, middle_slice_idx]
    
    logger.info("Performing 2D reconstruction of middle slice...")
    start_time = time.time()
    reconstruction_2d = reconstructor.reconstruct(middle_slice, angles)
    end_time = time.time()
    logger.info(f"Reconstruction completed in {end_time - start_time:.2f} seconds")
    
    # TODO: Implement full 3D reconstruction
    # For now, just reconstruct middle slice as proof of concept
    
    # Save results
    output_dir = data_dir / "reconstruction_results"
    output_dir.mkdir(exist_ok=True)
    
    # Save middle slice reconstruction with improved aesthetics
    plt.style.use('dark_background')
    
    # Create figure
    fig = plt.figure(figsize=(12, 10), dpi=150, facecolor='black')
    ax = plt.gca()
    
    # Plot reconstruction with enhanced contrast
    recon_data = reconstruction_2d.cpu().numpy()
    # Clip extreme values for better contrast
    p2, p98 = np.percentile(recon_data, (2, 98))
    recon_data = np.clip(recon_data, p2, p98)
    
    im = ax.imshow(recon_data, cmap='gray')
    ax.set_title('Middle Slice Reconstruction', fontsize=14, pad=15, color='white')
    ax.axis('off')
    
    # Add subtle crosshair at center
    center = recon_data.shape[0] // 2
    line_length = 20
    ax.axhline(y=center, xmin=0.48, xmax=0.52, color='white', alpha=0.3, linewidth=0.5)
    ax.axvline(x=center, ymin=0.48, ymax=0.52, color='white', alpha=0.3, linewidth=0.5)
    
    # Save with tight layout
    plt.tight_layout(pad=1.5)
    plt.savefig(output_dir / 'middle_slice_reconstruction.png', 
                facecolor='black', edgecolor='none', bbox_inches='tight')
    plt.close()
    
    logger.info(f"Results saved in {output_dir}")

if __name__ == "__main__":
    main()
