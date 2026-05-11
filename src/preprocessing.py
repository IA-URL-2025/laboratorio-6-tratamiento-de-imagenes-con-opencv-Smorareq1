import cv2
import numpy as np


def to_grayscale(image):
    """
    Convierte la imagen a escala de grises.

    Parámetros:
        image (np.ndarray): Imagen BGR de entrada (H x W x 3).

    Retorna:
        np.ndarray: Imagen en escala de grises (H x W), dtype uint8.

    Pista: cv2.cvtColor con cv2.COLOR_BGR2GRAY
    """
    # TODO: Implementar
    pass


def resize_image(image, width, height):
    """
    Redimensiona la imagen a las dimensiones indicadas.

    Parámetros:
        image (np.ndarray): Imagen de entrada.
        width  (int): Ancho deseado en píxeles.
        height (int): Alto deseado en píxeles.

    Retorna:
        np.ndarray: Imagen redimensionada (height x width x canales), dtype uint8.

    Pista: cv2.resize con interpolación cv2.INTER_AREA.
           Recuerda que cv2.resize recibe (width, height) como dsize.
    """
    # TODO: Implementar
    pass


def apply_blur(image, kernel_size=5):
    """
    Aplica un filtro de suavizado Gaussiano a la imagen.

    Parámetros:
        image       (np.ndarray): Imagen de entrada.
        kernel_size (int): Tamaño del kernel (debe ser impar: 3, 5, 7...). Default=5.

    Retorna:
        np.ndarray: Imagen suavizada, mismas dimensiones y dtype que la entrada.

    Pista: cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    """
    # TODO: Implementar
    pass


def adjust_brightness_contrast(image, alpha=1.0, beta=0):
    """
    Ajusta el brillo y el contraste de la imagen.

    Parámetros:
        image (np.ndarray): Imagen de entrada.
        alpha (float): Factor de contraste (1.0 = sin cambio). Default=1.0.
        beta  (int):   Ajuste de brillo  (0   = sin cambio). Default=0.

    Retorna:
        np.ndarray: Imagen ajustada, mismas dimensiones, dtype uint8.

    Pista: cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    """
    # TODO: Implementar
    pass


def apply_threshold(image, thresh_value=127):
    """
    Aplica umbralización binaria a una imagen en escala de grises.

    Parámetros:
        image        (np.ndarray): Imagen en escala de grises (1 canal), dtype uint8.
        thresh_value (int): Valor umbral [0-255]. Default=127.

    Retorna:
        np.ndarray: Imagen binarizada (H x W), dtype uint8 con valores 0 o 255.

    Nota: Si la imagen no es de un solo canal, lanza ValueError.
    Pista: cv2.threshold con cv2.THRESH_BINARY, valor máximo=255.
           cv2.threshold retorna (retval, imagen_umbralizada).
    """
    if len(image.shape) != 2:
        raise ValueError("apply_threshold requiere una imagen en escala de grises (1 canal).")
    # TODO: Implementar
    pass


def detect_edges(image, low=50, high=150):
    """
    Detecta bordes con el algoritmo de Canny.

    Parámetros:
        image (np.ndarray): Imagen de entrada (BGR o escala de grises).
        low   (int): Umbral inferior de Canny. Default=50.
        high  (int): Umbral superior de Canny. Default=150.

    Retorna:
        np.ndarray: Mapa de bordes (H x W), dtype uint8.

    Pista: Si la imagen tiene más de un canal, conviértela primero a escala de grises.
           Luego aplica cv2.Canny(gray, low, high).
    """
    # TODO: Implementar
    pass


def full_pipeline(image, target_width=224, target_height=224):
    """
    Ejecuta el pipeline completo de preprocesamiento.

    Orden obligatorio:
        1. Redimensionar a (target_width x target_height)
        2. Convertir a escala de grises
        3. Aplicar blur Gaussiano (kernel_size=3)
        4. Detectar bordes con Canny (low=50, high=150)

    Parámetros:
        image         (np.ndarray): Imagen BGR de entrada.
        target_width  (int): Ancho de salida. Default=224.
        target_height (int): Alto de salida.  Default=224.

    Retorna:
        np.ndarray: Imagen final procesada (target_height x target_width), dtype uint8.
    """
    # TODO: Implementar usando las funciones anteriores en el orden indicado
    pass
