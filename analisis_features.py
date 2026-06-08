"""
analisis_features.py
--------------------
Análisis cuantitativo espacial de lesiones cutáneas dermoscópicas (MEL vs NV).

El objetivo principal es caracterizar la lesión como una ESTRUCTURA ESPACIAL,
no simplemente medir descriptores globales. Para ello se divide la lesión en
zonas concéntricas, se analiza la transición lesión/piel sana, y se cuantifica
la heterogeneidad interna. La comparación MEL vs NV al final es una validación
exploratoria de que los descriptores capturan diferencias clínicamente relevantes.

Features extraídas:
  A — Asimetría       : índice de asimetría por eje, momentos de Hu, circularidad
  B — Borde           : compacidad, varianza radial, dimensión fractal del contorno
  C — Color global    : media/std/skewness por canal en RGB, HSV, L*a*b*
                        + colores dominantes (K-Means k=6) + entropía histograma
  D — Textura global  : GLCM (contraste, entropía, homogeneidad, correlación, energía)
                        + LBP (entropía del histograma)
  E — Análisis zonal  : estadísticas L*a*b* por zona (interna / anular / piel sana)
                        + ratios de contraste entre zonas
  F — Gradiente borde : magnitud del gradiente en la franja lesión/piel
                        (abrupto vs difuso) + índice de abrupticidad
  G — Heterogeneidad  : variación entre cuadrantes + varianza local de intensidad
                        (textura homogénea vs fragmentada)

Uso
---
  python analisis_features.py --n 10 --save_csv --save_figs

Salidas
-------
  features.csv              → tabla con todas las features por imagen
  analisis_figs/            → figura resumen por imagen (incluye mapa zonal)
  boxplots.png              → comparativa descriptiva MEL vs NV por feature
"""

import argparse
import os
import warnings
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import skew
from scipy.ndimage import binary_fill_holes
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from sklearn.cluster import KMeans

warnings.filterwarnings("ignore")

# ── Rutas ────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
CLASES     = {"mel": os.path.join(BASE_DIR, "mel"),
              "nv":  os.path.join(BASE_DIR, "nv")}
MASK_DIRS  = {"mel": os.path.join(BASE_DIR, "segmentacion_mascaras", "mel"),
              "nv":  os.path.join(BASE_DIR, "segmentacion_mascaras", "nv")}
OUT_CSV    = os.path.join(BASE_DIR, "features.csv")
OUT_FIGS   = os.path.join(BASE_DIR, "analisis_figs")
OUT_BOX    = os.path.join(BASE_DIR, "boxplots.png")
# ─────────────────────────────────────────────────────────────────────────────


# ══════════════════════════════════════════════════════════════════════════════
# PREPROCESAMIENTO INLINE (corrección de iluminación y color)
# ══════════════════════════════════════════════════════════════════════════════
def corregir_iluminacion(img: np.ndarray, clip_limit: float = 2.0,
                         tile_grid: tuple = (8, 8)) -> np.ndarray:
    """CLAHE sobre canal L de L*a*b* para corregir iluminación no uniforme."""
    lab   = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def normalizar_color(img: np.ndarray,
                     percentil_bajo: float = 2.0,
                     percentil_alto: float = 98.0) -> np.ndarray:
    """Estiramiento de histograma por canal (percentiles 2–98)."""
    img_norm = np.zeros_like(img, dtype=np.float32)
    for c in range(3):
        canal  = img[:, :, c].astype(np.float32)
        p_low  = np.percentile(canal, percentil_bajo)
        p_high = np.percentile(canal, percentil_alto)
        if p_high > p_low:
            img_norm[:, :, c] = np.clip(
                (canal - p_low) / (p_high - p_low) * 255.0, 0, 255
            )
        else:
            img_norm[:, :, c] = canal
    return img_norm.astype(np.uint8)


# ══════════════════════════════════════════════════════════════════════════════
# CARGA (con normalización integrada)
# ══════════════════════════════════════════════════════════════════════════════
def cargar(img_path, mask_path):
    img_raw = cv2.cvtColor(np.array(Image.open(img_path).convert("RGB")), cv2.COLOR_RGB2BGR)
    img     = normalizar_color(corregir_iluminacion(img_raw))   # imagen normalizada
    mask    = (cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)
    return img, mask


# ══════════════════════════════════════════════════════════════════════════════
# A — ASIMETRÍA
# ══════════════════════════════════════════════════════════════════════════════
def feat_asimetria(mask: np.ndarray) -> dict:
    """
    Índice de asimetría (0=perfectamente simétrico).
    Se bisecta la lesión en horizontal y vertical; se mide diferencia de área.
    También calcula circularidad y los 4 primeros momentos de Hu.
    """
    feats = {}

    # Asimetría por eje
    h, w     = mask.shape
    top      = mask[:h//2, :]
    bottom   = mask[h//2:, :]
    left     = mask[:, :w//2]
    right    = mask[:, w//2:]

    area     = int(mask.sum()) + 1e-6
    asim_v   = abs(int(top.sum()) - int(bottom.sum())) / area    # asimetría vertical
    asim_h   = abs(int(left.sum()) - int(right.sum()))  / area   # asimetría horizontal
    feats["asim_vertical"]   = round(asim_v, 4)
    feats["asim_horizontal"] = round(asim_h, 4)
    feats["asim_total"]      = round((asim_v + asim_h) / 2, 4)

    # Circularidad (1 = círculo perfecto, >1 más irregular)
    contours, _ = cv2.findContours((mask * 255).astype(np.uint8),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cnt        = max(contours, key=cv2.contourArea)
        perimetro  = cv2.arcLength(cnt, True)
        area_cnt   = cv2.contourArea(cnt) + 1e-6
        feats["circularidad"] = round(perimetro**2 / (4 * np.pi * area_cnt), 4)
    else:
        feats["circularidad"] = np.nan

    # Momentos de Hu (log escala para manejabilidad)
    momentos   = cv2.moments((mask * 255).astype(np.uint8))
    hu         = cv2.HuMoments(momentos).flatten()
    for j, h_val in enumerate(hu[:4], 1):
        feats[f"hu_{j}"] = round(-np.sign(h_val) * np.log10(abs(h_val) + 1e-10), 4)

    return feats


# ══════════════════════════════════════════════════════════════════════════════
# B — BORDE
# ══════════════════════════════════════════════════════════════════════════════
def feat_borde(mask: np.ndarray) -> dict:
    feats = {}

    contours, _ = cv2.findContours((mask * 255).astype(np.uint8),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return {k: np.nan for k in ["compacidad", "var_radial", "dim_fractal"]}

    cnt       = max(contours, key=cv2.contourArea)
    area      = cv2.contourArea(cnt) + 1e-6
    perimetro = cv2.arcLength(cnt, True) + 1e-6

    # Compacidad (1 = círculo, mayor = más irregular)
    feats["compacidad"] = round(perimetro**2 / (4 * np.pi * area), 4)

    # Varianza de distancia radial al centroide
    M      = cv2.moments(cnt)
    cx     = M["m10"] / (M["m00"] + 1e-6)
    cy     = M["m01"] / (M["m00"] + 1e-6)
    pts    = cnt[:, 0, :]
    dists  = np.sqrt((pts[:, 0] - cx)**2 + (pts[:, 1] - cy)**2)
    feats["var_radial"] = round(float(dists.std() / (dists.mean() + 1e-6)), 4)

    # Dimensión fractal (método box-counting simplificado)
    # Trabajamos sobre la imagen del contorno
    h_m, w_m = mask.shape
    contour_img = np.zeros((h_m, w_m), dtype=np.uint8)
    cv2.drawContours(contour_img, [cnt], -1, 1, 1)

    sizes  = [2, 4, 8, 16, 32]
    counts = []
    for s in sizes:
        reduced = contour_img[:h_m//s*s, :w_m//s*s].reshape(h_m//s, s, w_m//s, s)
        counts.append((reduced.max(axis=(1, 3)) > 0).sum())

    if len(set(counts)) > 1:
        coeffs = np.polyfit(np.log(sizes), np.log(np.array(counts) + 1e-6), 1)
        feats["dim_fractal"] = round(abs(coeffs[0]), 4)
    else:
        feats["dim_fractal"] = np.nan

    return feats


# ══════════════════════════════════════════════════════════════════════════════
# C — COLOR
# ══════════════════════════════════════════════════════════════════════════════
def feat_color(img: np.ndarray, mask: np.ndarray) -> dict:
    feats = {}
    m     = mask.astype(bool)

    # Espacios de color
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    espacios = {"rgb": img_rgb, "hsv": img_hsv, "lab": img_lab}
    nombres  = {"rgb": ["r","g","b"], "hsv": ["h","s","v"], "lab": ["l","a","b_"]}

    for esp, im in espacios.items():
        for ch_idx, ch_name in enumerate(nombres[esp]):
            canal = im[:,:,ch_idx][m].astype(float)
            feats[f"{esp}_{ch_name}_mean"] = round(float(canal.mean()), 3)
            feats[f"{esp}_{ch_name}_std"]  = round(float(canal.std()),  3)
            feats[f"{esp}_{ch_name}_skew"] = round(float(skew(canal)),  3)

    # Entropía del histograma de luminosidad (canal L de Lab)
    canal_L = img_lab[:,:,0][m]
    hist, _ = np.histogram(canal_L, bins=32, range=(0, 255), density=True)
    hist    = hist[hist > 0]
    feats["color_entropia_L"] = round(float(-np.sum(hist * np.log2(hist))), 4)

    # Número de colores dominantes con K-Means (k=6)
    pixels = img_rgb[m].astype(np.float32)
    if len(pixels) >= 6:
        # Submuestreo para acelerar K-Means
        idx = np.random.choice(len(pixels), min(2000, len(pixels)), replace=False)
        km = KMeans(n_clusters=6, n_init=3, max_iter=50, random_state=42)
        km.fit(pixels[idx])
        # Proporción mínima para contar como "dominante" (>5% de píxeles)
        labels, counts_km = np.unique(km.labels_, return_counts=True)
        proporciones = counts_km / counts_km.sum()
        feats["colores_dominantes"] = int((proporciones > 0.05).sum())
    else:
        feats["colores_dominantes"] = np.nan

    return feats


# ══════════════════════════════════════════════════════════════════════════════
# D — TEXTURA (GLCM + LBP)
# ══════════════════════════════════════════════════════════════════════════════
def feat_textura(img: np.ndarray, mask: np.ndarray) -> dict:
    feats = {}

    # Canal L de L*a*b* (más estable que gris para dermoscopia)
    lab   = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    gray  = lab[:,:,0]

    # Enmascarar región de interés (fondo a 0)
    gray_masked = (gray * mask).astype(np.uint8)

    # ── GLCM ─────────────────────────────────────────────────────────────────
    # Cuantizamos a 64 niveles para acelerar sin perder información relevante
    gray_q = (gray_masked // 4).astype(np.uint8)
    glcm = graycomatrix(gray_q, distances=[1, 2],
                        angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                        levels=64, symmetric=True, normed=True)

    for prop in ["contrast", "homogeneity", "energy", "correlation"]:
        val = graycoprops(glcm, prop).mean()
        feats[f"glcm_{prop}"] = round(float(val), 5)

    # Entropía GLCM manual (graycoprops no la incluye)
    glcm_norm = glcm_q.mean(axis=(2, 3)) if False else glcm.mean(axis=(2, 3))  # promedio sobre distancias y ángulos
    glcm_norm = glcm_norm / (glcm_norm.sum() + 1e-10)
    p         = glcm_norm[glcm_norm > 0]
    feats["glcm_entropy"] = round(float(-np.sum(p * np.log2(p))), 4)

    # ── LBP ──────────────────────────────────────────────────────────────────
    # LBP uniforme: radio=2, n_points=16
    lbp  = local_binary_pattern(gray, P=16, R=2, method="uniform")
    # Solo dentro de la máscara
    lbp_vals = lbp[mask.astype(bool)]
    hist_lbp, _ = np.histogram(lbp_vals, bins=18, range=(0, 18), density=True)
    hist_lbp    = hist_lbp[hist_lbp > 0]
    feats["lbp_entropy"] = round(float(-np.sum(hist_lbp * np.log2(hist_lbp))), 4)

    # Energía LBP (suma de cuadrados del histograma normalizado)
    hist_lbp2, _ = np.histogram(lbp_vals, bins=18, range=(0, 18), density=True)
    feats["lbp_energy"] = round(float(np.sum(hist_lbp2**2)), 5)

    return feats


# ══════════════════════════════════════════════════════════════════════════════
# E — ANÁLISIS ZONAL (zonas internas vs periféricas)
# ══════════════════════════════════════════════════════════════════════════════
def feat_zonas(img: np.ndarray, mask: np.ndarray) -> dict:
    """
    Divide la lesión en zonas concéntricas y compara sus características.

    Zonas definidas:
      - zona_interna : núcleo erosionado hasta ~50% del área original.
      - zona_anular  : corona entre el núcleo y el borde de la lesión.
      - franja_piel  : banda de piel sana adyacente al borde exterior.

    Para cada zona se computan media y std de L, a*, b* (espacio L*a*b*).
    Los ratios interna/anular y lesión/piel cuantifican el contraste espacial
    del pigmento: en el melanoma se espera mayor heterogeneidad zonal.
    """
    feats = {}
    lab   = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    area_total = float(mask.sum()) + 1e-6

    # Erosión iterativa hasta retener ~50 % del área
    zona_interna = mask.astype(np.uint8).copy()
    for _ in range(40):
        eroded = cv2.erode(zona_interna, kernel, iterations=1)
        if eroded.sum() < 0.45 * area_total or eroded.sum() == 0:
            break
        zona_interna = eroded
    zona_interna = (zona_interna > 0).astype(np.uint8)
    zona_anular  = np.clip(mask.astype(np.uint8) - zona_interna, 0, 1)

    # Franja de piel sana: banda dilatada fuera de la lesión
    mask_dil   = cv2.dilate((mask * 255).astype(np.uint8), kernel, iterations=12)
    franja_piel = np.clip((mask_dil > 0).astype(np.uint8) - mask.astype(np.uint8), 0, 1)

    zonas = {"int": zona_interna, "anular": zona_anular, "piel": franja_piel}
    for zname, zmask in zonas.items():
        m = zmask.astype(bool)
        if m.sum() < 10:
            for ch in ["L", "a", "b"]:
                feats[f"zona_{zname}_{ch}_mean"] = np.nan
                feats[f"zona_{zname}_{ch}_std"]  = np.nan
            continue
        for chi, ch in enumerate(["L", "a", "b"]):
            vals = lab[:, :, chi][m].astype(float)
            feats[f"zona_{zname}_{ch}_mean"] = round(float(vals.mean()), 3)
            feats[f"zona_{zname}_{ch}_std"]  = round(float(vals.std()),  3)

    # Ratios de contraste zonal (indicadores de distribución del pigmento)
    for ch in ["L", "a", "b"]:
        vi = feats.get(f"zona_int_{ch}_mean",    np.nan)
        va = feats.get(f"zona_anular_{ch}_mean", np.nan)
        vp = feats.get(f"zona_piel_{ch}_mean",   np.nan)
        if not any(np.isnan([vi, va])) and abs(va) > 1e-3:
            feats[f"ratio_int_anular_{ch}"] = round(vi / (va + 1e-6), 4)
        else:
            feats[f"ratio_int_anular_{ch}"] = np.nan
        if not any(np.isnan([va, vp])) and abs(vp) > 1e-3:
            feats[f"ratio_lesion_piel_{ch}"] = round(va / (vp + 1e-6), 4)
        else:
            feats[f"ratio_lesion_piel_{ch}"] = np.nan

    return feats


# ══════════════════════════════════════════════════════════════════════════════
# F — GRADIENTE DE BORDE (abrupto vs difuso)
# ══════════════════════════════════════════════════════════════════════════════
def feat_gradiente_borde(img: np.ndarray, mask: np.ndarray) -> dict:
    """
    Cuantifica la abrupticidad de la transición lesión / piel sana.

    Se extrae una franja de borde (dilatación - erosión de la máscara) y se
    mide la magnitud del gradiente de Sobel sobre el canal L de L*a*b* en esa
    franja. Un gradiente alto y concentrado (bajo std) indica borde nítido
    (abrupto); uno bajo y disperso indica transición gradual (difusa).

    El índice_abrupticidad = media/std del gradiente resume este comportamiento
    en un único escalar interpretable clínicamente.
    """
    feats  = {}
    lab    = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    gray_L = lab[:, :, 0].astype(np.float32)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask_u8 = (mask * 255).astype(np.uint8)

    dilated = cv2.dilate(mask_u8, kernel, iterations=4)
    eroded  = cv2.erode(mask_u8,  kernel, iterations=4)
    franja  = ((dilated > 0) & ~(eroded > 0)).astype(np.uint8)

    sobelx   = cv2.Sobel(gray_L, cv2.CV_64F, 1, 0, ksize=3)
    sobely   = cv2.Sobel(gray_L, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(sobelx**2 + sobely**2)

    grad_borde = grad_mag[franja.astype(bool)]
    if len(grad_borde) > 0:
        feats["grad_borde_mean"]   = round(float(grad_borde.mean()), 4)
        feats["grad_borde_std"]    = round(float(grad_borde.std()),  4)
        feats["grad_borde_max"]    = round(float(grad_borde.max()),  4)
        feats["indice_abrupticidad"] = round(
            float(grad_borde.mean() / (grad_borde.std() + 1e-6)), 4
        )
    else:
        for k in ["grad_borde_mean", "grad_borde_std", "grad_borde_max", "indice_abrupticidad"]:
            feats[k] = np.nan

    return feats


# ══════════════════════════════════════════════════════════════════════════════
# G — HETEROGENEIDAD INTERNA
# ══════════════════════════════════════════════════════════════════════════════
def feat_heterogeneidad(img: np.ndarray, mask: np.ndarray) -> dict:
    """
    Cuantifica si la textura interna de la lesión es homogénea o fragmentada.

    Dos métricas complementarias:
    1. CV entre cuadrantes: coeficiente de variación de la intensidad media L
       en los 4 cuadrantes de la lesión. Alto → pigmentación no uniforme entre
       regiones (lesión fragmentada).
    2. Varianza local media: varianza de intensidad calculada en ventanas
       deslizantes sobre toda la lesión. Alto → micro-heterogeneidad de textura.
    """
    feats  = {}
    lab    = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    gray_L = lab[:, :, 0]

    h, w = mask.shape
    cuadrantes = {
        "sup_izq": (mask[:h//2, :w//2],    gray_L[:h//2, :w//2]),
        "sup_der": (mask[:h//2, w//2:],    gray_L[:h//2, w//2:]),
        "inf_izq": (mask[h//2:, :w//2],    gray_L[h//2:, :w//2]),
        "inf_der": (mask[h//2:, w//2:],    gray_L[h//2:, w//2:]),
    }

    medias_cuad = []
    for m_cuad, im_cuad in cuadrantes.values():
        m = m_cuad.astype(bool)
        if m.sum() > 10:
            medias_cuad.append(float(im_cuad[m].mean()))

    if len(medias_cuad) >= 2:
        medias = np.array(medias_cuad)
        feats["heterog_cv_cuadrantes"]    = round(
            float(medias.std() / (medias.mean() + 1e-6)), 4
        )
        feats["heterog_rango_cuadrantes"] = round(
            float((medias.max() - medias.min()) / (medias.mean() + 1e-6)), 4
        )
    else:
        feats["heterog_cv_cuadrantes"]    = np.nan
        feats["heterog_rango_cuadrantes"] = np.nan

    # Varianza local (ventana 15×15) dentro de la máscara
    gray_f = gray_L.astype(np.float32)
    mu     = cv2.GaussianBlur(gray_f,    (15, 15), 0)
    mu2    = cv2.GaussianBlur(gray_f**2, (15, 15), 0)
    var_local = np.maximum(mu2 - mu**2, 0)
    var_lesion = var_local[mask.astype(bool)]

    if len(var_lesion) > 0:
        feats["heterog_var_local_mean"] = round(float(var_lesion.mean()), 4)
        feats["heterog_var_local_std"]  = round(float(var_lesion.std()),  4)
    else:
        feats["heterog_var_local_mean"] = np.nan
        feats["heterog_var_local_std"]  = np.nan

    return feats


# ══════════════════════════════════════════════════════════════════════════════
# FIGURA POR IMAGEN
# ══════════════════════════════════════════════════════════════════════════════
def _construir_mapa_zonas(mask: np.ndarray) -> np.ndarray:
    """
    Devuelve imagen RGB con zonas coloreadas:
      verde  → zona interna
      amarillo → zona anular
      azul   → franja de piel sana
      gris   → fondo
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    area_total = float(mask.sum()) + 1e-6

    zona_interna = mask.astype(np.uint8).copy()
    for _ in range(40):
        eroded = cv2.erode(zona_interna, kernel, iterations=1)
        if eroded.sum() < 0.45 * area_total or eroded.sum() == 0:
            break
        zona_interna = eroded
    zona_interna = (zona_interna > 0).astype(np.uint8)
    zona_anular  = np.clip(mask.astype(np.uint8) - zona_interna, 0, 1)
    mask_dil     = cv2.dilate((mask * 255).astype(np.uint8), kernel, iterations=12)
    franja_piel  = np.clip((mask_dil > 0).astype(np.uint8) - mask.astype(np.uint8), 0, 1)

    h, w = mask.shape
    mapa = np.full((h, w, 3), 50, dtype=np.uint8)          # fondo gris oscuro
    mapa[franja_piel.astype(bool)]  = [100, 149, 237]      # azul piel sana
    mapa[zona_anular.astype(bool)]  = [255, 215, 0]        # amarillo zona anular
    mapa[zona_interna.astype(bool)] = [60,  179, 113]      # verde zona interna
    return mapa


def guardar_fig_imagen(img, mask, nombre, clase, feats, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    img_rgb  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_lab  = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    gray_L   = img_lab[:,:,0]

    # LBP visualización
    lbp_img  = local_binary_pattern(gray_L, P=16, R=2, method="uniform")

    # Mapa de zonas concéntricas
    mapa_zonas = _construir_mapa_zonas(mask)

    # Overlay con contorno
    overlay = img_rgb.copy()
    overlay[mask == 1] = (overlay[mask == 1] * 0.45 + np.array([255,80,80]) * 0.55).astype(np.uint8)
    contours, _ = cv2.findContours((mask*255).astype(np.uint8),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (255, 230, 0), 2)

    fig = plt.figure(figsize=(20, 9))
    fig.suptitle(f"{clase.upper()} — {nombre} · Análisis cuantitativo espacial",
                 fontsize=13, fontweight="bold")

    gs = fig.add_gridspec(2, 5, hspace=0.45, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, 0]); ax1.imshow(img_rgb);            ax1.set_title("Imagen normalizada");    ax1.axis("off")
    ax2 = fig.add_subplot(gs[0, 1]); ax2.imshow(overlay);            ax2.set_title("Segmentación");          ax2.axis("off")
    ax3 = fig.add_subplot(gs[0, 2]); ax3.imshow(mapa_zonas);         ax3.set_title("Zonas (int/anular/piel)"); ax3.axis("off")
    ax4 = fig.add_subplot(gs[0, 3]); ax4.imshow(lbp_img, cmap="nipy_spectral"); ax4.set_title("LBP (textura)"); ax4.axis("off")
    ax5 = fig.add_subplot(gs[0, 4]); ax5.imshow(gray_L * mask, cmap="gray"); ax5.set_title("Canal L (luminosidad)"); ax5.axis("off")

    # Histograma Lab por zona (histograma de L)
    ax6 = fig.add_subplot(gs[1, 0:2])
    for ch, color, label in [(0,"gray","L"),(1,"green","a*"),(2,"blue","b*")]:
        vals = img_lab[:,:,ch][mask.astype(bool)].astype(float)
        ax6.hist(vals, bins=50, alpha=0.5, color=color, label=label, density=True)
    ax6.set_title("Histograma L*a*b* (lesión completa)")
    ax6.legend(fontsize=8); ax6.set_xlabel("Intensidad"); ax6.set_ylabel("Densidad")

    # Tabla de features espaciales clave
    ax7 = fig.add_subplot(gs[1, 2:5])
    ax7.axis("off")
    keys_show = [
        "asim_total", "indice_abrupticidad", "heterog_cv_cuadrantes",
        "heterog_var_local_mean", "ratio_int_anular_L", "ratio_lesion_piel_L",
        "zona_int_a_mean", "zona_anular_a_mean", "glcm_entropy", "lbp_entropy",
    ]
    labels_show = [
        "Asimetría total", "Índice abrupticidad borde", "CV entre cuadrantes",
        "Var. local media", "Ratio interno/anular (L)", "Ratio lesión/piel (L)",
        "Zona interna a* (media)", "Zona anular a* (media)", "GLCM Entropía", "LBP Entropía",
    ]
    table_data = [[l, f"{feats.get(k, 'N/A')}"] for l, k in zip(labels_show, keys_show)]
    tbl = ax7.table(cellText=table_data, colLabels=["Descriptor espacial", "Valor"],
                    loc="center", cellLoc="left")
    tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1.15, 1.4)
    ax7.set_title("Descriptores cuantitativos clave", pad=10)

    plt.savefig(os.path.join(out_dir, f"{clase}_{nombre}.png"),
                dpi=110, bbox_inches="tight")
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# BOXPLOTS COMPARATIVOS
# ══════════════════════════════════════════════════════════════════════════════
def guardar_boxplots(df: pd.DataFrame, out_path: str):
    features_plot = [
        # — Asimetría y forma —
        ("asim_total",               "Asimetría total"),
        ("circularidad",             "Circularidad"),
        # — Borde —
        ("indice_abrupticidad",      "Índice de abrupticidad"),
        ("grad_borde_mean",          "Gradiente borde (media)"),
        # — Heterogeneidad interna —
        ("heterog_cv_cuadrantes",    "CV entre cuadrantes"),
        ("heterog_var_local_mean",   "Varianza local media"),
        # — Análisis zonal (L) —
        ("ratio_int_anular_L",       "Ratio interno/anular (L)"),
        ("ratio_lesion_piel_L",      "Ratio lesión/piel (L)"),
        # — Color por zona (a*) —
        ("zona_int_a_mean",          "Zona interna — a* (media)"),
        ("zona_anular_a_mean",       "Zona anular — a* (media)"),
        # — Textura global —
        ("glcm_entropy",             "GLCM Entropía"),
        ("lbp_entropy",              "LBP Entropía"),
        # — Color global —
        ("lab_a_mean",               "Lab a* global (media)"),
        ("colores_dominantes",       "Colores dominantes"),
    ]

    n_cols = 4
    n_rows = int(np.ceil(len(features_plot) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, n_rows * 3.5))
    fig.suptitle(
        "Análisis cuantitativo espacial — MEL vs NV (validación exploratoria)",
        fontsize=14, fontweight="bold"
    )

    colors = {"mel": "#E05C5C", "nv": "#5C8AE0"}

    for idx, (feat, titulo) in enumerate(features_plot):
        ax = axes[idx // n_cols][idx % n_cols]
        data_mel = df[df["clase"] == "mel"][feat].dropna().values
        data_nv  = df[df["clase"] == "nv"][feat].dropna().values

        bp = ax.boxplot([data_mel, data_nv],
                        labels=["MEL", "NV"],
                        patch_artist=True,
                        medianprops=dict(color="white", linewidth=2),
                        whiskerprops=dict(linewidth=1.5),
                        capprops=dict(linewidth=1.5))

        bp["boxes"][0].set_facecolor(colors["mel"])
        bp["boxes"][1].set_facecolor(colors["nv"])

        # Puntos individuales
        for i, (datos, x_pos) in enumerate([(data_mel, 1), (data_nv, 2)], 0):
            ax.scatter(np.random.normal(x_pos, 0.06, len(datos)), datos,
                       alpha=0.7, s=25, color=list(colors.values())[i], zorder=5)

        ax.set_title(titulo, fontsize=10, fontweight="bold")
        ax.set_ylabel("Valor", fontsize=8)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    # Ocultar ejes sobrantes
    for idx in range(len(features_plot), n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Boxplots guardados en: {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",         type=int,  default=10)
    parser.add_argument("--save_csv",  action="store_true")
    parser.add_argument("--save_figs", action="store_true")
    args = parser.parse_args()

    print(f"\n=== Análisis ABCD HAM10000 — Grupo 8 ===")
    print(f"  N por clase: {args.n}\n")

    all_rows = []

    for clase, img_dir in CLASES.items():
        mask_dir = MASK_DIRS[clase]
        mascaras = sorted([f for f in os.listdir(mask_dir)
                           if f.endswith(".png")])[:args.n]

        print(f"[{clase.upper()}] procesando {len(mascaras)} imágenes...")

        for mask_fname in mascaras:
            # Nombre base: ISIC_XXXXXXX
            base     = mask_fname.replace("_mask.png", "")
            img_path  = os.path.join(img_dir, base + ".jpg")
            mask_path = os.path.join(mask_dir, mask_fname)

            if not os.path.exists(img_path):
                print(f"  [SKIP] imagen no encontrada: {img_path}")
                continue

            img, mask = cargar(img_path, mask_path)

            fa = feat_asimetria(mask)
            fb = feat_borde(mask)
            fc = feat_color(img, mask)
            fd = feat_textura(img, mask)
            fe = feat_zonas(img, mask)
            ff = feat_gradiente_borde(img, mask)
            fg = feat_heterogeneidad(img, mask)

            row = {"clase": clase, "imagen": base,
                   **fa, **fb, **fc, **fd, **fe, **ff, **fg}
            all_rows.append(row)

            if args.save_figs:
                guardar_fig_imagen(img, mask, base, clase, row, OUT_FIGS)

            print(f"  ✓ {base}  asim={fa['asim_total']}  "
                  f"contrast={fd['glcm_contrast']}  lbp_H={fd['lbp_entropy']}")

    df = pd.DataFrame(all_rows)

    # Resumen por clase — descriptores cuantitativos espaciales
    print(f"\n{'='*75}")
    print("  RESUMEN — Media por clase (descriptores espaciales)")
    print(f"{'='*75}")
    cols_show = [
        "asim_total", "indice_abrupticidad", "heterog_cv_cuadrantes",
        "heterog_var_local_mean", "ratio_int_anular_L", "ratio_lesion_piel_L",
        "zona_int_a_mean", "zona_anular_a_mean",
        "glcm_entropy", "lbp_entropy",
    ]
    cols_show = [c for c in cols_show if c in df.columns]
    resumen = df.groupby("clase")[cols_show].mean().round(4)
    print(resumen.to_string())

    if args.save_csv:
        df.to_csv(OUT_CSV, index=False)
        print(f"\n  CSV guardado: {OUT_CSV}")

    guardar_boxplots(df, OUT_BOX)

    print("\n=== Fin del análisis ===\n")
    return df


if __name__ == "__main__":
    main()
