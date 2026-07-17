"""
Preprocessing module for fibroid classification pipeline.
Handles resizing, denoising, contrast enhancement, and quality checks.
"""
import cv2
import numpy as np

TARGET_SIZE = (224, 224)


def load_image(path, color_mode='rgb'):
    """Load image from disk as numpy array."""
    img = cv2.imread(path, cv2.IMREAD_COLOR if color_mode == 'rgb' else cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    if color_mode == 'rgb':
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def denoise(img):
    """Reduce speckle noise common in ultrasound images."""
    if len(img.shape) == 3:
        return cv2.fastNlMeansDenoisingColored(img, None, h=7, hColor=7,
                                                  templateWindowSize=7, searchWindowSize=21)
    return cv2.fastNlMeansDenoising(img, None, h=7,
                                      templateWindowSize=7, searchWindowSize=21)


def enhance_contrast(img):
    """Apply CLAHE (adaptive histogram equalization) for better tissue contrast."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    if len(img.shape) == 3:
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        l = clahe.apply(l)
        lab = cv2.merge((l, a, b))
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return clahe.apply(img)


def resize_image(img, target_size=TARGET_SIZE):
    """Resize with aspect-ratio-preserving pad instead of naive stretch."""
    h, w = img.shape[:2]
    target_h, target_w = target_size
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Pad to exact target size
    pad_w = target_w - new_w
    pad_h = target_h - new_h
    top, bottom = pad_h // 2, pad_h - pad_h // 2
    left, right = pad_w // 2, pad_w - pad_w // 2

    if len(img.shape) == 3:
        padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                      cv2.BORDER_CONSTANT, value=[0, 0, 0])
    else:
        padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                      cv2.BORDER_CONSTANT, value=0)
    return padded


def normalize(img):
    """Scale pixel values to [0, 1] float32."""
    return img.astype(np.float32) / 255.0


def preprocess_pipeline(path, target_size=TARGET_SIZE, apply_denoise=True, apply_clahe=True):
    """Full preprocessing pipeline: load -> denoise -> enhance -> resize -> normalize."""
    img = load_image(path, color_mode='rgb')
    if apply_denoise:
        img = denoise(img)
    if apply_clahe:
        img = enhance_contrast(img)
    img = resize_image(img, target_size)
    img = normalize(img)
    return img


def check_blur(img_gray, threshold=829.77):
    """Laplacian variance blur detection. Lower score = blurrier."""
    score = cv2.Laplacian(img_gray, cv2.CV_64F).var()
    return {'blur_score': float(score), 'is_blurry': score < threshold}


def check_contrast(img_gray, threshold=44.82):
    """Low standard deviation in pixel intensity indicates flat/low-contrast image."""
    std = float(img_gray.std())
    return {'contrast_std': std, 'is_low_contrast': std < threshold}


def check_brightness(img_gray, low=20, high=235):
    """Flag images that are too dark or too washed out."""
    mean_val = float(img_gray.mean())
    return {'mean_brightness': mean_val, 'is_bad_exposure': mean_val < low or mean_val > high}


def quality_report(path):
    """Run all quality checks on a raw (unprocessed) image and return a summary dict."""
    img_gray = load_image(path, color_mode='gray')
    report = {}
    report.update(check_blur(img_gray))
    report.update(check_contrast(img_gray))
    report.update(check_brightness(img_gray))
    report['passed'] = not (report['is_blurry'] or report['is_low_contrast'] or report['is_bad_exposure'])
    return report
