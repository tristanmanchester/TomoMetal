import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import tifffile
from typing import List, Optional, Tuple, Union
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TomographicReconstructor:
    def __init__(self, device: Optional[str] = None):
        """Initialize the reconstructor with optional device specification."""
        # Use MPS if available, otherwise CPU
        self.device = (
            torch.device("mps") if torch.backends.mps.is_available() and device != "cpu"
            else torch.device("cpu")
        )
        logger.info(f"Using device: {self.device}")

    def load_projections(self, tiff_paths: List[Union[str, Path]]) -> torch.Tensor:
        """Load TIFF projections into a tensor."""
        projections = []
        for path in tiff_paths:
            img = tifffile.imread(str(path))
            projections.append(torch.from_numpy(img.astype(np.float32)))
        
        # Stack projections into a single tensor [num_projections, height, width]
        sinogram = torch.stack(projections, dim=0)
        return sinogram.to(self.device)

    def create_ramp_filter(self, size: int, use_hamming: bool = True) -> torch.Tensor:
        """Create a ramp filter in Fourier space.
        
        Args:
            size: Size of the filter
            use_hamming: Whether to apply a Hamming window to reduce high-frequency noise
        """
        # Create frequency axis
        freq = torch.fft.fftfreq(size, d=1.0, device=self.device)
        omega = 2 * np.pi * freq
        
        # Create ramp filter
        ramp = torch.abs(omega)
        
        if use_hamming:
            # Apply Hamming window to reduce high frequency noise
            hamming = 0.54 + 0.46 * torch.cos(np.pi * torch.arange(size, device=self.device) / (size - 1))
            ramp = ramp * hamming
        
        return ramp

    def apply_filter(self, sinogram: torch.Tensor, use_hamming: bool = True) -> torch.Tensor:
        num_angles, width = sinogram.shape
        ramp_filter = self.create_ramp_filter(width, use_hamming)
        filtered = []
        for i in range(num_angles):
            row = sinogram[i]
            padded = F.pad(row, (width//4, width//4))
            row_fft = torch.fft.fft(padded)
            filtered_fft = row_fft * self.create_ramp_filter(len(padded), use_hamming)
            filtered_row = torch.fft.ifft(filtered_fft).real
            filtered.append(filtered_row[width//4:-width//4])
        return torch.stack(filtered)

    def backproject(self, filtered_sinogram: torch.Tensor, angles: torch.Tensor, output_size: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        num_angles, width = filtered_sinogram.shape
        if output_size is None:
            output_size = (width, width)
        
        # Create output coordinate grid
        y, x = torch.meshgrid(
            torch.linspace(-1, 1, output_size[0], device=self.device),
            torch.linspace(-1, 1, output_size[1], device=self.device),
            indexing='ij'
        )
        reconstruction = torch.zeros(output_size, device=self.device)
        
        for i, angle in enumerate(angles):
            # Projection coordinate (Radon transform geometry)
            t = x * torch.cos(angle) + y * torch.sin(angle)  # t in [-1, 1] since x,y are in [-1, 1]
            # No need for extra normalization since our input coordinates are already in [-1, 1]
            grid = t.unsqueeze(-1).unsqueeze(0)  # [1, H, W, 1]
            
            # Prepare projection as [1, 1, width, 1] to treat it as a 1D signal along "width"
            proj = filtered_sinogram[i].unsqueeze(0).unsqueeze(0).unsqueeze(-1)  # [1, 1, width, 1]
            
            # Interpolate using grid_sample (treat proj as a 1D signal along the 3rd dimension)
            contribution = F.grid_sample(
                proj,
                grid.expand(-1, output_size[0], output_size[1], 2),  # Expand to [1, H, W, 2], second dim is dummy
                mode='bilinear',
                padding_mode='zeros',
                align_corners=True
            )
            reconstruction += contribution.squeeze()  # Remove extra dims: [H, W]
        
        return reconstruction / num_angles

    def reconstruct(
        self,
        sinogram: torch.Tensor,
        angles: torch.Tensor,
        output_size: Optional[Tuple[int, int]] = None,
        use_hamming: bool = True
    ) -> torch.Tensor:
        """Complete FBP reconstruction pipeline.
        
        Args:
            sinogram: Input sinogram [num_angles, height, width]
            angles: Projection angles
            output_size: Optional size of output image
            use_hamming: Whether to apply Hamming window in filter
        """
        # Move angles to device if they aren't already
        angles = angles.to(self.device)
        
        # Apply filtering
        logger.info("Applying ramp filter...")
        filtered = self.apply_filter(sinogram, use_hamming)
        
        # Perform backprojection
        logger.info("Performing backprojection...")
        reconstruction = self.backproject(filtered, angles, output_size)
        
        return reconstruction
