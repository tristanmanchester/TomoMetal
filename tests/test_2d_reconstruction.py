"""Test module for 2D tomographic reconstruction using synthetic data.

This module demonstrates and tests the core functionality of the tomographic reconstruction
pipeline using a 2D Shepp-Logan-like phantom. It includes:

1. Phantom Generation: Creates a 2D phantom with elliptical features
2. Forward Projection: Simulates X-ray projections at different angles to create sinograms
3. Back Projection: Reconstructs the original phantom from the sinograms

The test creates visualizations showing:
- The original phantom
- The sinogram (projections at different angles)
- The reconstructed image

The results are saved as 'docs/images/example_reconstruction.png'.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from tomometal.reconstruction import TomographicReconstructor
import logging
import torch.nn.functional as F
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_shepp_logan_phantom(size: int = 256) -> torch.Tensor:
    """Create a simple 2D Shepp-Logan-like phantom with elliptical features.
    
    Args:
        size: Size of the square image (size x size pixels)
        
    Returns:
        torch.Tensor: 2D tensor of shape [size, size] containing the phantom image
        with values between 0 and 1, where 0 represents no attenuation and 1
        represents maximum attenuation.
    """
    phantom = torch.zeros((size, size))
    center = size // 2
    
    # Create main ellipse
    y, x = torch.meshgrid(
        torch.arange(size),
        torch.arange(size),
        indexing='ij'
    )
    
    # Create outer ellipse
    a, b = size//3, size//4
    mask = ((x - center)**2 / a**2 + (y - center)**2 / b**2) <= 1
    phantom[mask] = 1.0
    
    # Add some internal features
    a, b = size//6, size//8
    mask = ((x - center - size//8)**2 / a**2 + (y - center)**2 / b**2) <= 1
    phantom[mask] = 0.5
    
    return phantom

def create_projections(phantom: torch.Tensor, angles: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Create parallel beam projections of a 2D phantom at specified angles.
    
    This function simulates X-ray projections by rotating the phantom and summing
    along the vertical axis. It uses grid sampling for accurate rotation.
    
    Args:
        phantom: 2D tensor [height, width] containing the phantom image
        angles: 1D tensor containing projection angles in radians
        device: Torch device to perform computations on
        
    Returns:
        torch.Tensor: Projection data of shape [num_angles, width] containing
        the simulated X-ray projections at each angle
    """
    phantom = phantom.to(device)
    size = phantom.shape[0]
    projections = torch.zeros((len(angles), size), device=device)  # [num_angles, width]
    
    for i, angle in enumerate(angles):
        cos_a = torch.cos(angle)
        sin_a = torch.sin(angle)
        y, x = torch.meshgrid(torch.linspace(-1, 1, size), torch.linspace(-1, 1, size), indexing='ij')
        y, x = y.to(device), x.to(device)
        x_rot = x * cos_a - y * sin_a
        y_rot = x * sin_a + y * cos_a
        grid = torch.stack([x_rot, y_rot], dim=-1).unsqueeze(0)
        rotated = F.grid_sample(
            phantom.unsqueeze(0).unsqueeze(0),
            grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True
        )[0, 0]  # [size, size]
        projection = rotated.sum(dim=0)  # Sum along y-axis: [size]
        projections[i] = projection
    
    return projections

def main():
    # Initialize reconstructor
    reconstructor = TomographicReconstructor()
    logger.info(f"Using device: {reconstructor.device}")
    
    # Create phantom
    size = 512  # Higher resolution
    phantom = create_shepp_logan_phantom(size)
    
    # Generate projection angles (in radians)
    key_angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]  # 0, 45, 90, 135 degrees
    num_angles = 360  # More angles for better quality
    angles = torch.linspace(0, np.pi, num_angles, device=reconstructor.device)
    logger.info(f"Created {num_angles} angles from 0 to π radians")
    
    # Create projections
    logger.info("Creating synthetic projections...")
    projections = create_projections(phantom, angles, reconstructor.device)
    
    # Perform reconstruction
    logger.info("Performing reconstruction...")
    reconstruction = reconstructor.reconstruct(projections, angles, use_hamming=False)
    
    # Move tensors to CPU for visualization
    phantom_cpu = phantom.cpu().numpy()
    projections_cpu = projections.cpu().numpy()
    reconstruction_cpu = reconstruction.cpu().numpy()
    
    # Set style for clean, modern look
    plt.style.use('dark_background')
    
    # Create figure with specific dimensions
    fig = plt.figure(figsize=(20, 6), dpi=150, facecolor='black')
    gs = plt.GridSpec(1, 3, figure=fig, wspace=0.15)
    
    # Original phantom
    ax1 = fig.add_subplot(gs[0])
    im1 = ax1.imshow(phantom_cpu, cmap='gray')
    ax1.set_title('Original Phantom', fontsize=14, pad=15, color='white')
    ax1.axis('off')
    
    # Sinogram with custom extent and aspect
    ax2 = fig.add_subplot(gs[1])
    im2 = ax2.imshow(projections_cpu, cmap='gray', aspect='auto', 
                    extent=[0, size, 180, 0])
    ax2.set_title('Sinogram', fontsize=14, pad=15, color='white')
    # Only show a few angle ticks
    ax2.set_yticks([0, 45, 90, 135, 180])
    ax2.set_ylabel('Angle (degrees)', fontsize=12, color='white', labelpad=10)
    ax2.set_xlabel('Detector Position', fontsize=12, color='white', labelpad=10)
    # Style the ticks
    ax2.tick_params(colors='white', which='both')
    
    # Reconstruction
    ax3 = fig.add_subplot(gs[2])
    im3 = ax3.imshow(reconstruction_cpu, cmap='gray')
    ax3.set_title('Reconstruction', fontsize=14, pad=15, color='white')
    ax3.axis('off')
    
    # Adjust layout
    plt.tight_layout(pad=2.0)
    # Create output directory
    output_dir = Path('test_data/phantom_2d_reconstruction')
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / 'reconstruction_results.png')
    plt.close()
    
    logger.info("Test complete! Check reconstruction_test.png for results.")

if __name__ == "__main__":
    main()
