[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/JPMKmwcO)
# Laboratorio 6 — Preprocesamiento de Imágenes con OpenCV

**Universidad Rafael Landívar · Inteligencia Artificial · Primer Semestre 2026**

---

## Objetivo

Implementar técnicas fundamentales de preprocesamiento de imágenes utilizando Python y OpenCV,
comprendiendo cómo cada operación transforma la representación digital de una imagen.

---

## Instrucciones

### 1. Clonar el repositorio

```bash
git clone <URL-de-tu-repo>
cd image-preprocessing-lab
```

### 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 3. Implementar las funciones

**El único archivo que debes editar es `src/preprocessing.py`.**

Implementa cada una de las 7 funciones siguiendo las instrucciones del enunciado:

| Función | Descripción | Puntos |
|---|---|---|
| `to_grayscale(image)` | Convierte a escala de grises | 10 |
| `resize_image(image, width, height)` | Redimensiona la imagen | 10 |
| `apply_blur(image, kernel_size=5)` | Aplica suavizado Gaussiano | 15 |
| `apply_threshold(image, thresh_value=127)` | Umbralización binaria | 15 |
| `adjust_brightness_contrast(image, alpha, beta)` | Ajusta brillo/contraste | 15 |
| `detect_edges(image, low=50, high=150)` | Detecta bordes (Canny) | 15 |
| `full_pipeline(image, w=224, h=224)` | Pipeline completo | 20 |
| **Total** | | **100** |

### 4. Probar localmente

```bash
# Todos los tests
pytest tests/

# Un test específico
pytest tests/test_preprocessing.py::TestToGrayscale -v

# Con más detalle en los errores
pytest tests/ --tb=long
```

### 5. Entregar

Haz `push` a tu repositorio de GitHub Classroom. Los tests correrán automáticamente y verás
tu puntaje en la pestaña **Actions** de tu repositorio.

---

## Estructura del repositorio

```
image-preprocessing-lab/
├── images/
│   ├── sample.jpg              # Imagen de entrada — NO modificar
│   └── expected/               # Imágenes de referencia — NO modificar
├── src/
│   └── preprocessing.py        # ← ÚNICO archivo que debes editar
├── tests/
│   └── test_preprocessing.py   # Tests automáticos — NO modificar
├── requirements.txt
├── pytest.ini
└── README.md
```

---

## Notas importantes

- **NO modifiques** ningún archivo fuera de `src/preprocessing.py`.
- Todas las funciones deben retornar arrays de NumPy con `dtype=np.uint8`.
- La función `apply_threshold` solo acepta imágenes de un solo canal (escala de grises).
- La función `detect_edges` debe manejar imágenes BGR **y** en escala de grises.
- El `full_pipeline` debe encadenar las funciones en el **orden exacto** indicado en el enunciado.

---

## Recursos útiles

- [Documentación oficial OpenCV Python](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html)
- [Tutorial: Image Thresholding](https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html)
- [Tutorial: Canny Edge Detection](https://docs.opencv.org/4.x/da/d22/tutorial_py_canny.html)
- [Tutorial: Smoothing Images](https://docs.opencv.org/4.x/d4/d13/tutorial_py_filtering.html)
