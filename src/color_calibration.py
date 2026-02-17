import numpy as np

def gray_world_white_balance(rgb: np.ndarray) -> np.ndarray:
    """
    Gray-world white balance to reduce warm/cool lighting bias.
    Input/Output: RGB uint8 image.
    """
    img = rgb.astype(np.float32)

    mean_r = img[..., 0].mean() + 1e-6
    mean_g = img[..., 1].mean() + 1e-6
    mean_b = img[..., 2].mean() + 1e-6
    mean_gray = (mean_r + mean_g + mean_b) / 3.0

    img[..., 0] *= (mean_gray / mean_r)
    img[..., 1] *= (mean_gray / mean_g)
    img[..., 2] *= (mean_gray / mean_b)

    return np.clip(img, 0, 255).astype(np.uint8)
