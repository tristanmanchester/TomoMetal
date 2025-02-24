import torch
import numpy as np
from pathlib import Path
from tomometal.reconstruction import TomographicReconstructor
import matplotlib.pyplot as plt

def main():
    # Initialize reconstructor
    reconstructor = TomographicReconstructor()
    
    # Load your TIFF projections
    tiff_dir = Path("path/to/your/tiffs")  # Update this path
    tiff_files = sorted(tiff_dir.glob("*.tiff"))
    
    # Load projections
    sinogram = reconstructor.load_projections(tiff_files)
    
    # Create or load your angles (example: 180 angles from 0 to 180 degrees)
    angles = torch.linspace(0, 180, len(tiff_files), device=reconstructor.device)
    
    # Perform reconstruction
    reconstruction = reconstructor.reconstruct(sinogram, angles)
    
    # Move result to CPU for visualization
    reconstruction_cpu = reconstruction.cpu().numpy()
    
    # Visualize result
    plt.figure(figsize=(10, 10))
    plt.imshow(reconstruction_cpu, cmap='gray')
    plt.colorbar()
    plt.title('Reconstructed Image')
    plt.savefig('reconstruction.png')
    plt.close()

if __name__ == "__main__":
    main()
