import torch
import numpy as np
import matplotlib.pyplot as plt
from tomometal.reconstruction import TomographicReconstructor
import logging
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_shepp_logan_phantom(size: int = 256) -> torch.Tensor:
    """Create a simple Shepp-Logan-like phantom."""
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
    
    # Main visualization with higher DPI and better aspect ratio
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), dpi=150)
    plt.subplots_adjust(wspace=0.3)  # Add more space between subplots
    
    # Original phantom
    axes[0].imshow(phantom_cpu, cmap='gray')
    axes[0].set_title('Original Phantom', fontsize=12, pad=10)
    axes[0].axis('off')
    
    # Sinogram
    axes[1].imshow(projections_cpu, cmap='gray', aspect='auto', extent=[0, size, 360, 0])
    axes[1].set_title('Sinogram', fontsize=12, pad=10)
    axes[1].set_ylabel('Angle (degrees)', fontsize=10)
    axes[1].set_xlabel('Detector Position', fontsize=10)
    
    # Reconstruction
    axes[2].imshow(reconstruction_cpu, cmap='gray')
    axes[2].set_title('Reconstruction', fontsize=12, pad=10)
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig('docs/images/example_reconstruction.png')
    plt.close()
    
    logger.info("Test complete! Check reconstruction_test.png for results.")

if __name__ == "__main__":
    main()
