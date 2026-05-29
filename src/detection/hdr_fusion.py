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
    """Multi-exposure capture configuration.

    Defaults match the production electrolyte inspection camera:
    - miscible camera: 50ms (50000us) normal, 100ms over
    - hierarchical camera: 10ms (10000us) normal, 20ms over
    """
    base_exposure_us: int = 10000    # Normal (matches hierarchical 10ms)
    over_exposure_us: int = 20000    # +1 EV (double exposure time)
    use_under_exposure: bool = False # Disabled per operator preference
    use_alignment: bool = True       # ECC frame alignment
    fusion_contrast: float = 1.0
    fusion_saturation: float = 1.0
    fusion_well_exposed: float = 1.0

    @classmethod
    def for_hierarchical(cls) -> "HDRConfig":
        """Config for hierarchical (分层) camera: 10ms normal, 20ms over.
        Note: 10ms→20ms = 1 EV. Avoid 50ms (5x over = blown out)."""
        return cls(base_exposure_us=10000, over_exposure_us=20000)

    @classmethod
    def for_miscible(cls) -> "HDRConfig":
        """Config for miscible (互溶) camera: 50ms normal, 80ms over.
        Note: 50ms→80ms = ~0.7 EV. Avoid large gaps to prevent blowout."""
        return cls(base_exposure_us=50000, over_exposure_us=80000)


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

        # Auto-reject blown-out frames (mean > 250, std < 10 = all white)
        valid_frames = []
        for f in frames:
            gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if len(f.shape) == 3 else f
            if gray.mean() > 250 and gray.std() < 10:
                print(f'[HDR] Skipping blown-out frame (mean={gray.mean():.0f}, std={gray.std():.0f})')
                continue
            valid_frames.append(f)

        if len(valid_frames) < 2:
            return valid_frames[0] if valid_frames else frames[0]
        frames = valid_frames

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
        Capture 2 frames (normal + over-exposed), fuse with Mertens.

        Args:
            capture_fn: Function that takes exposure_us and returns a BGR frame
            *args, **kwargs: Passed to capture_fn after exposure

        Returns:
            (fused_image, metadata_dict)
        """
        exposures = [self.config.base_exposure_us, self.config.over_exposure_us]
        frames = []
        for exp in exposures:
            frame = capture_fn(exp, *args, **kwargs)
            if frame is not None:
                frames.append(frame)

        if len(frames) < 2:
            return frames[0] if frames else None, {"error": "Not enough frames"}

        fused = self.fuse(frames)
        return fused, {
            "exposures_used": exposures,
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
