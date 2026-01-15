import numpy as np
import cv2

FFT_SIZE = 24              # v2.4/v2.5 使用 24×24 crop
NUM_PHASE_LEVELS = 16      # 相位量化分桶
SMOOTH_KERNEL = 3          # Phase smoothing kernel size


def fft_features_amp_phase(img_tensor):
    """
    Extract amplitude and phase with:
      - fftshift
      - 24×24 center crop
      - log amplitude
      - global amplitude normalization (per image)
      - smoothed phase
      - phase quantization
    """

    # Convert torch tensor -> numpy
    img = img_tensor.squeeze().numpy().astype(np.float32)

    # ---------------- FFT ----------------
    F = np.fft.fft2(img)
    F_shift = np.fft.fftshift(F)

    h, w = F_shift.shape
    c_h, c_w = h // 2, w // 2

    # ---------------- 24×24 low-frequency crop ----------------
    F_crop = F_shift[c_h-12:c_h+12, c_w-12:c_w+12]  # (24,24)

    # ---------------- Amplitude ----------------
    amp = np.abs(F_crop)
    amp = np.log1p(amp)  # log compression
    amp = amp / (amp.max() + 1e-8)  # normalize 0~1
    amp_bin = (amp * 255).astype(np.uint8)

    # ---------------- Phase ----------------
    phase = np.angle(F_crop)
    phase = cv2.blur(phase, (SMOOTH_KERNEL, SMOOTH_KERNEL))  # smoothing

    # normalize phase from [-pi,pi] -> 0..1
    phase_norm = (phase + np.pi) / (2 * np.pi)
    phase_bin = np.floor(phase_norm * NUM_PHASE_LEVELS).astype(int)
    phase_bin = np.clip(phase_bin, 0, NUM_PHASE_LEVELS - 1)

    # Flatten consistent with training order
    return amp_bin.flatten(), phase_bin.flatten()
