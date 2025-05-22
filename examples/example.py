import torch
import numpy as np
from pathlib import Path
from tomometal.reconstruction import TomographicReconstructor
import matplotlib.pyplot as plt
import argparse
import sys

def main():
    try:
        # Set up argument parser
        parser = argparse.ArgumentParser(description="Reconstruct tomographic data.")
        parser.add_argument("--input_dir", type=str, required=True,
                            help="Path to the directory containing TIFF files.")
        parser.add_argument("--output_file", type=str, default="reconstruction.png",
                            help="Path and filename for the saved reconstruction image.")
        parser.add_argument("--use_hamming", action="store_true",
                            help="Enable Hamming window during filtering.")
        parser.add_argument("--num_angles", type=int, default=180,
                            help="The number of projection angles to generate.")
        args = parser.parse_args()

        # Input directory check
        input_path = Path(args.input_dir)
        if not input_path.exists():
            print(f"Error: Input directory '{args.input_dir}' does not exist. Exiting.")
            sys.exit(1)
        if not input_path.is_dir():
            print(f"Error: Input path '{args.input_dir}' is not a directory. Exiting.")
            sys.exit(1)

        # Initialize reconstructor
        reconstructor = TomographicReconstructor()
        
        # Load your TIFF projections
        tiff_files = sorted(input_path.glob("*.tiff"))

        # TIFF file check
        if not tiff_files:
            print(f"No TIFF files found in '{args.input_dir}'. Exiting.")
            sys.exit(1)
        
        # Load projections
        sinogram = reconstructor.load_projections(tiff_files)
        
        # Create angles
        angles = torch.linspace(0, 180, args.num_angles, device=reconstructor.device)
        
        # Perform reconstruction
        reconstruction = reconstructor.reconstruct(sinogram, angles, use_hamming=args.use_hamming)
        
        # Move result to CPU for visualization
        reconstruction_cpu = reconstruction.cpu().numpy()
        
        # Visualize result
        plt.figure(figsize=(10, 10))
        plt.imshow(reconstruction_cpu, cmap='gray')
        plt.colorbar()
        plt.title('Reconstructed Image')
        plt.savefig(args.output_file)
        plt.close()
        print(f"Reconstruction saved to {args.output_file}")

    except Exception as e:
        print(f"An error occurred during the process: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
