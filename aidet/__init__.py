"""aidet: classic computer-vision building blocks for AI-generated image detection.

Week-1 modules (numpy + Pillow only, no deep learning framework needed):

- residuals   : NPR, SRM and denoise-and-subtract noise residuals
- spectrum    : log-magnitude FFT spectra, averaging, radial profiles
- preprocess  : 224x224 crops that never resize, JPEG re-encoding, file listing
- viz         : turn spectra / residuals into PNG panels without matplotlib
"""

from . import preprocess, residuals, spectrum, viz  # noqa: F401

__all__ = ["preprocess", "residuals", "spectrum", "viz"]
