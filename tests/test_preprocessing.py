"""
test_preprocessing.py — Autograding para Laboratorio 6
Universidad Rafael Landívar · Inteligencia Artificial 2026

NO MODIFICAR ESTE ARCHIVO.
"""

import os
import sys
import pytest
import numpy as np
import cv2

# ── Path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from preprocessing import (
    to_grayscale,
    resize_image,
    apply_blur,
    adjust_brightness_contrast,
    apply_threshold,
    detect_edges,
    full_pipeline,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────
IMAGES_DIR   = os.path.join(os.path.dirname(__file__), "..", "images")
EXPECTED_DIR = os.path.join(IMAGES_DIR, "expected")
SAMPLE_PATH  = os.path.join(IMAGES_DIR, "sample.jpg")

SSIM_THRESHOLD = 0.92   # Similitud estructural mínima aceptable
MAE_THRESHOLD  = 10.0   # Error absoluto medio máximo (en intensidad de píxel)


@pytest.fixture(scope="module")
def sample_image():
    img = cv2.imread(SAMPLE_PATH)
    assert img is not None, f"No se pudo cargar {SAMPLE_PATH}"
    return img


@pytest.fixture(scope="module")
def gray_image(sample_image):
    return cv2.cvtColor(sample_image, cv2.COLOR_BGR2GRAY)


def load_expected(filename):
    path = os.path.join(EXPECTED_DIR, filename)
    img  = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    assert img is not None, f"Imagen de referencia no encontrada: {path}"
    return img


def mae(a, b):
    """Mean Absolute Error entre dos arrays del mismo shape."""
    return float(np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32))))


# ══════════════════════════════════════════════════════════════════════════════
# 1. to_grayscale  (10 pts — 2 tests × 5 pts)
# ══════════════════════════════════════════════════════════════════════════════
class TestToGrayscale:

    def test_output_shape_and_dtype(self, sample_image):
        """La salida debe ser 2-D y dtype uint8."""
        result = to_grayscale(sample_image)
        assert result is not None,          "to_grayscale retornó None"
        assert result.dtype == np.uint8,    "dtype debe ser uint8"
        assert len(result.shape) == 2,      "la imagen debe tener 1 canal (shape 2-D)"
        h, w = sample_image.shape[:2]
        assert result.shape == (h, w),      f"shape esperado ({h},{w}), obtenido {result.shape}"

    def test_pixel_values(self, sample_image):
        """Los valores de píxel deben coincidir con la referencia (MAE ≤ {MAE_THRESHOLD})."""
        result   = to_grayscale(sample_image)
        expected = load_expected("grayscale.png")
        assert result.shape == expected.shape, "shape no coincide con referencia"
        assert mae(result, expected) <= MAE_THRESHOLD, (
            f"MAE={mae(result, expected):.2f} supera el umbral {MAE_THRESHOLD}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# 2. resize_image  (10 pts — 2 tests × 5 pts)
# ══════════════════════════════════════════════════════════════════════════════
class TestResizeImage:

    def test_resize_224x224(self, sample_image):
        """Debe producir shape (224, 224, 3) y dtype uint8."""
        result = resize_image(sample_image, 224, 224)
        assert result is not None,        "resize_image retornó None"
        assert result.dtype == np.uint8,  "dtype debe ser uint8"
        assert result.shape == (224, 224, 3), (
            f"shape esperado (224,224,3), obtenido {result.shape}"
        )

    def test_resize_100x80(self, sample_image):
        """Debe producir shape (80, 100, 3) — altura=80, ancho=100."""
        result = resize_image(sample_image, 100, 80)
        assert result is not None,        "resize_image retornó None"
        assert result.dtype == np.uint8,  "dtype debe ser uint8"
        assert result.shape == (80, 100, 3), (
            f"shape esperado (80,100,3), obtenido {result.shape}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# 3. apply_blur  (15 pts — 2 tests × 7-8 pts)
# ══════════════════════════════════════════════════════════════════════════════
class TestApplyBlur:

    def test_shape_dtype_preserved(self, sample_image):
        """Shape y dtype deben conservarse después del blur."""
        result = apply_blur(sample_image, kernel_size=5)
        assert result is not None,                      "apply_blur retornó None"
        assert result.dtype == np.uint8,                "dtype debe ser uint8"
        assert result.shape == sample_image.shape,      "shape no debe cambiar"

    def test_pixel_values_kernel5(self, sample_image):
        """Los valores deben coincidir con la referencia para kernel=5."""
        result   = apply_blur(sample_image, kernel_size=5)
        expected = load_expected("blurred_k5.png")
        assert mae(result, expected) <= MAE_THRESHOLD, (
            f"MAE={mae(result, expected):.2f} supera el umbral {MAE_THRESHOLD}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# 4. apply_threshold  (15 pts — 2 tests × 7-8 pts)
# ══════════════════════════════════════════════════════════════════════════════
class TestApplyThreshold:

    def test_output_binary(self, gray_image):
        """La salida solo debe contener valores 0 y 255."""
        result = apply_threshold(gray_image, thresh_value=127)
        assert result is not None,         "apply_threshold retornó None"
        assert result.dtype == np.uint8,   "dtype debe ser uint8"
        assert len(result.shape) == 2,     "la salida debe ser 2-D"
        unique = set(np.unique(result).tolist())
        assert unique.issubset({0, 255}),  f"solo deben existir valores 0 y 255, encontrado: {unique}"

    def test_pixel_values_127(self, gray_image):
        """Los valores deben coincidir con la referencia para thresh=127."""
        result   = apply_threshold(gray_image, thresh_value=127)
        expected = load_expected("threshold_127.png")
        assert mae(result, expected) <= MAE_THRESHOLD, (
            f"MAE={mae(result, expected):.2f} supera el umbral {MAE_THRESHOLD}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# 5. adjust_brightness_contrast  (15 pts — 2 tests × 7-8 pts)
# ══════════════════════════════════════════════════════════════════════════════
class TestAdjustBrightnessContrast:

    def test_shape_dtype_preserved(self, sample_image):
        """Shape y dtype deben conservarse."""
        result = adjust_brightness_contrast(sample_image, alpha=1.5, beta=30)
        assert result is not None,                 "adjust_brightness_contrast retornó None"
        assert result.dtype == np.uint8,           "dtype debe ser uint8"
        assert result.shape == sample_image.shape, "shape no debe cambiar"

    def test_pixel_values_alpha1_5_beta30(self, sample_image):
        """Los valores deben coincidir con la referencia (alpha=1.5, beta=30)."""
        result   = adjust_brightness_contrast(sample_image, alpha=1.5, beta=30)
        expected = load_expected("brightness_contrast_1_5_30.png")
        assert mae(result, expected) <= MAE_THRESHOLD, (
            f"MAE={mae(result, expected):.2f} supera el umbral {MAE_THRESHOLD}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# 6. detect_edges  (15 pts — 3 tests × 5 pts)
# ══════════════════════════════════════════════════════════════════════════════
class TestDetectEdges:

    def test_output_shape_from_color(self, sample_image):
        """Desde imagen BGR la salida debe ser 2-D (H x W)."""
        result = detect_edges(sample_image, low=50, high=150)
        assert result is not None,         "detect_edges retornó None"
        assert result.dtype == np.uint8,   "dtype debe ser uint8"
        assert len(result.shape) == 2,     "la salida debe ser 2-D (mapa de bordes)"
        h, w = sample_image.shape[:2]
        assert result.shape == (h, w),     f"shape esperado ({h},{w}), obtenido {result.shape}"

    def test_output_shape_from_gray(self, gray_image):
        """Desde imagen en grises la salida debe seguir siendo 2-D."""
        result = detect_edges(gray_image, low=30, high=100)
        assert result is not None,         "detect_edges retornó None"
        assert len(result.shape) == 2,     "la salida debe ser 2-D"

    def test_pixel_values_50_150(self, sample_image):
        """Los valores deben coincidir con la referencia (low=50, high=150)."""
        result   = detect_edges(sample_image, low=50, high=150)
        expected = load_expected("edges_50_150.png")
        assert mae(result, expected) <= MAE_THRESHOLD, (
            f"MAE={mae(result, expected):.2f} supera el umbral {MAE_THRESHOLD}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# 7. full_pipeline  (20 pts — 3 tests)
# ══════════════════════════════════════════════════════════════════════════════
class TestFullPipeline:

    def test_output_shape(self, sample_image):
        """La salida debe ser (224, 224) — 2-D, dtype uint8."""
        result = full_pipeline(sample_image, 224, 224)
        assert result is not None,         "full_pipeline retornó None"
        assert result.dtype == np.uint8,   "dtype debe ser uint8"
        assert len(result.shape) == 2,     "la salida debe ser 2-D (mapa de bordes)"
        assert result.shape == (224, 224), f"shape esperado (224,224), obtenido {result.shape}"

    def test_output_custom_size(self, sample_image):
        """El pipeline debe respetar dimensiones personalizadas (150x100)."""
        result = full_pipeline(sample_image, 150, 100)
        assert result is not None,           "full_pipeline retornó None"
        assert result.shape == (100, 150),   f"shape esperado (100,150), obtenido {result.shape}"

    def test_pixel_values_224(self, sample_image):
        """Los valores deben coincidir con la referencia del pipeline 224x224."""
        result   = full_pipeline(sample_image, 224, 224)
        expected = load_expected("pipeline_224_224.png")
        assert mae(result, expected) <= MAE_THRESHOLD, (
            f"MAE={mae(result, expected):.2f} supera el umbral {MAE_THRESHOLD}"
        )
