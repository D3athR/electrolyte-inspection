"""
Multi-Exposure HDR Fusion for Electrolyte Detection.

Captures 3 frames at different exposures, aligns and fuses using
Mertens algorithm, then feeds the fused image to the classifier.

This implements the patented "adaptive exposure fusion" approach.
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class HDRConfig:
    """Multi-exposure capture configuration."""
    base_exposure_us: int = 10000   # Normal exposure (10ms = 10000us)
    under_exposure_us: int = 5000   # -1 EV
    over_exposure_us: int = 20000   # +1 EV
    use_alignment: bool = True      # Align frames before fusion
    fusion_contrast: float = 1.0    # Mertens contrast weight
    fusion_saturation: float = 1.0  # Mertens saturation weight
    fusion_well_exposed: float = 1.0  # Mertens well-exposedness weight


class HDRFusion:
    """Multi-exposure capture + Mertens fusion."""

    def __init__(self, config: HDRConfig = None):
        self.config = config or HDRConfig()
        self.fuser = cv2.createMergeMertens(
            self.config.fusion_contrast,
            self.config.fusion_saturation,
            self.config.fusion_well_exposed,
        )

    def fuse(self, frames: list[np.ndarray]) -> np.ndarray:
        """
        Fuse multiple exposure frames into one HDR image.

        Args:
            frames: List of BGR images at different exposures (same resolution)

        Returns:
            Fused HDR image as uint8 BGR
        """
        if len(frames) < 2:
            return frames[0] if frames else None

        # Align frames if needed
        if self.config.use_alignment and len(frames) >= 2:
            frames = self._align_frames(frames)

        # Convert to float32 in 0-1 range
        float_frames = [f.astype(np.float32) / 255.0 for f in frames]

        # Mertens fusion
        fused = self.fuser.process(float_frames)

        # Back to uint8
        return np.clip(fused * 255, 0, 255).astype(np.uint8)

    def _align_frames(self, frames: list[np.ndarray]) -> list[np.ndarray]:
        """Align frames using ECC (Enhanced Correlation Coefficient)."""
        aligned = [frames[0]]
        ref = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY).astype(np.float32)

        warp_mode = cv2.MOTION_TRANSLATION
        warp_matrix = np.eye(2, 3, dtype=np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-6)

        for frame in frames[1:]:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
            try:
                _, warp_matrix = cv2.findTransformECC(
                    ref, gray, warp_matrix, warp_mode, criteria, None, 5)
                h, w = frame.shape[:2]
                aligned_frame = cv2.warpAffine(
                    frame, warp_matrix, (w, h),
                    flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
                aligned.append(aligned_frame)
            except cv2.error:
                # ECC failed (e.g., too much motion), use original
                aligned.append(frame)

        return aligned

    def capture_and_fuse(self, capture_fn, *args, **kwargs) -> tuple[np.ndarray, dict]:
        """
        Capture 3 frames at different exposures and fuse.

        Args:
            capture_fn: Function that takes exposure_us and returns a BGR frame
            *args, **kwargs: Passed to capture_fn after exposure

        Returns:
            (fused_image, metadata_dict)
        """
        exposures = [
            self.config.under_exposure_us,
            self.config.base_exposure_us,
            self.config.over_exposure_us,
        ]
        frames = []
        for exp in exposures:
            frame = capture_fn(exp, *args, **kwargs)
            if frame is not None:
                frames.append(frame)

        if len(frames) < 2:
            return frames[0] if frames else None, {"error": "Not enough frames"}

        fused = self.fuse(frames)
        return fused, {
            "exposures_used": exposures[:len(frames)],
            "frames_captured": len(frames),
            "method": "Mertens",
        }


def simulate_multi_exposure(image: np.ndarray, num_exposures: int = 3,
                            ev_step: float = 1.0) -> list[np.ndarray]:
    """
    Simulate multi-exposure from a single image (for testing without hardware).

    Args:
        image: Base image (BGR, uint8)
        num_exposures: Number of exposure levels
        ev_step: EV step between exposures

    Returns:
        List of simulated multi-exposure images
    """
    frames = []
    base_ev = 0
    for i in range(num_exposures):
        ev_offset = (i - num_exposures // 2) * ev_step
        scale = 2.0 ** ev_offset
        frame = np.clip(image.astype(np.float32) * scale, 0, 255).astype(np.uint8)
        frames.append(frame)

    return frames
