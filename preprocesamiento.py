"""
preprocesamiento.py
-------------------
Pipeline de preprocesamiento para el dataset HAM10000 (clases: mel / nv).

El preprocesamiento es el paso fundacional del análisis cuantitativo: una
segmentación precisa y una imagen normalizada en iluminación y color son
condición necesaria para que los descriptores espaciales extraídos después
(heterogeneidad zonal, gradiente de borde, distribución del pigmento) sean
comparables entre imágenes adquiridas en condiciones distintas.

Pasos por imagen:
  1. Carga con PIL
  2. Remoción de pelo (blackhat + inpainting)
  3. Corrección de iluminación (CLAHE sobre canal L de L*a*b*)
  4. Normalización de color (estiramiento de histograma por canal)
  5. Segmentación por Otsu (canal verde + GaussianBlur)
  6. Retención del único objeto más grande (componente conectada mayor)
  7. Post-proceso: relleno de huecos + apertura morfológica
  8. Guardado de máscaras y figura comparativa

Uso
---
  python preprocesamiento.py --n 50 --save_masks --save_figs
  python preprocesamiento.py --n 10 --save_masks --save_figs   # prueba
"""

import argparse
import os
import numpy as np
from PIL import Image
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import binary_fill_holes

# ── Rutas base ───────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
CLASES    = {"mel": os.path.join(BASE_DIR, "mel"),
             "nv":  os.path.join(BASE_DIR, "nv")}
OUT_MASKS = os.path.join(BASE_DIR, "segmentacion_mascaras")
OUT_FIGS  = os.path.join(BASE_DIR, "segmentacion_figs")
# ─────────────────────────────────────────────────────────────────────────────


# ══════════════════════════════════════════════════════════════════════════════
# 1. CARGA
# ══════════════════════════════════════════════════════════════════════════════
def cargar_imagen(path: str) -> np.ndarray:
    """Devuelve ndarray BGR uint8."""
    pil = Image.open(path).convert("RGB")
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


# ══════════════════════════════════════════════════════════════════════════════
# 2. REMOCIÓN DE PELO
# ══════════════════════════════════════════════════════════════════════════════
def remover_pelo(img: np.ndarray, kernel_size: int = 17) -> np.ndarray:
    """Blackhat para detectar pelos + inpainting para rellenarlos."""
    gray     = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel   = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, mask_pelo = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    return cv2.inpaint(img, mask_pelo, inpaintRadius=3, flags=cv2.INPAINT_TELEA)


# ══════════════════════════════════════════════════════════════════════════════
# 2b. CORRECCIÓN DE ILUMINACIÓN (CLAHE)
# ══════════════════════════════════════════════════════════════════════════════
def corregir_iluminacion(img: np.ndarray, clip_limit: float = 2.0,
                         tile_grid: tuple = (8, 8)) -> np.ndarray:
    """
    Equalización adaptativa de histograma (CLAHE) sobre el canal L de L*a*b*.
    Corrige gradientes de iluminación no uniformes (reflejos del dermoscopio,
    sombras de borde) sin modificar la información cromática (canales a*, b*).
    Un campo de iluminación uniforme es prerequisito para comparar intensidades
    entre zonas internas y periféricas de la lesión.
    """
    lab   = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


# ══════════════════════════════════════════════════════════════════════════════
# 2c. NORMALIZACIÓN DE COLOR (estiramiento de histograma por canal)
# ══════════════════════════════════════════════════════════════════════════════
def normalizar_color(img: np.ndarray,
                     percentil_bajo: float = 2.0,
                     percentil_alto: float = 98.0) -> np.ndarray:
    """
    Estira el histograma de cada canal BGR entre sus percentiles 2–98.
    Estandariza el rango dinámico de color entre imágenes con distintas
    condiciones de adquisición (balance de blancos del dermoscopio, exposición).
    Usar percentiles en lugar de min/max hace al método robusto ante píxeles
    atípicos de alto brillo (reflejos especulares residuales).
    """
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
# 3. OTSU + COMPONENTE MÁS GRANDE
# ══════════════════════════════════════════════════════════════════════════════
def componente_mas_grande(mask_bin: np.ndarray) -> np.ndarray:
    """
    Dado una máscara binaria uint8 {0,1}, devuelve una nueva máscara
    que conserva SOLO la componente conectada de mayor área.
    """
    mask_u8 = (mask_bin * 255).astype(np.uint8)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask_u8, connectivity=8
    )
    if n_labels <= 1:               # solo fondo → máscara vacía
        return np.zeros_like(mask_bin)

    # stats[0] es el fondo; buscamos el mayor entre las lesiones (label >= 1)
    areas      = stats[1:, cv2.CC_STAT_AREA]   # excluye el fondo
    mayor_idx  = int(np.argmax(areas)) + 1     # +1 porque excluimos fondo
    return (labels == mayor_idx).astype(np.uint8)


def segmentar_otsu(img: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Segmenta con Otsu sobre el canal verde + GaussianBlur.
    Aplica post-proceso morfológico y retiene solo el objeto más grande.

    Devuelve (mascara_binaria uint8 {0,1}, umbral_otsu).
    """
    green = img[:, :, 1]
    blur  = cv2.GaussianBlur(green, (5, 5), 0)

    umbral, mask = cv2.threshold(
        blur, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Limpieza morfológica
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=2)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)

    # Relleno de huecos internos
    mask_bin = binary_fill_holes(mask > 0).astype(np.uint8)

    # ── Quedarnos con UN solo objeto (el más grande) ──────────────────────────
    mask_bin = componente_mas_grande(mask_bin)

    return mask_bin, umbral


# ══════════════════════════════════════════════════════════════════════════════
# 4. FIGURA COMPARATIVA
# ══════════════════════════════════════════════════════════════════════════════
def guardar_figura(img_orig: np.ndarray,
                   img_limpia: np.ndarray,
                   img_norm: np.ndarray,
                   mask: np.ndarray,
                   nombre: str,
                   clase: str,
                   umbral: float,
                   out_dir: str) -> str:
    """Guarda figura 1×4: original | sin pelo | normalizada | overlay Otsu."""
    os.makedirs(out_dir, exist_ok=True)

    orig_rgb = cv2.cvtColor(img_orig,   cv2.COLOR_BGR2RGB)
    limp_rgb = cv2.cvtColor(img_limpia, cv2.COLOR_BGR2RGB)
    norm_rgb = cv2.cvtColor(img_norm,   cv2.COLOR_BGR2RGB)

    # Overlay semitransparente sobre imagen normalizada
    overlay = norm_rgb.copy()
    overlay[mask == 1] = (
        overlay[mask == 1] * 0.45 + np.array([255, 80, 80]) * 0.55
    ).astype(np.uint8)
    contours, _ = cv2.findContours(
        (mask * 255).astype(np.uint8),
        cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(overlay, contours, -1, (255, 230, 0), 2)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle(f"{clase.upper()} — {nombre}  (Otsu t={umbral:.0f})",
                 fontsize=13, fontweight="bold")

    axes[0].imshow(orig_rgb);           axes[0].set_title("1. Original")
    axes[1].imshow(limp_rgb);           axes[1].set_title("2. Sin pelo")
    axes[2].imshow(norm_rgb);           axes[2].set_title("3. Corrección ilum. + color")
    axes[3].imshow(overlay);            axes[3].set_title("4. Segmentación Otsu")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    fig_path = os.path.join(out_dir, f"{clase}_{nombre}.png")
    plt.savefig(fig_path, dpi=120, bbox_inches="tight")
    plt.close()
    return fig_path


# ══════════════════════════════════════════════════════════════════════════════
# 5. PROCESAMIENTO POR LOTE
# ══════════════════════════════════════════════════════════════════════════════
def procesar_clase(clase: str, carpeta: str, n: int,
                   save_masks: bool, save_figs: bool):
    imagenes = sorted([f for f in os.listdir(carpeta)
                       if f.lower().endswith((".jpg", ".jpeg", ".png"))])[:n]

    print(f"\n{'='*60}")
    print(f"  Clase: {clase.upper()}  |  {len(imagenes)} imágenes")
    print(f"{'='*60}")

    for i, fname in enumerate(imagenes, 1):
        path   = os.path.join(carpeta, fname)
        nombre = os.path.splitext(fname)[0]

        img_orig   = cargar_imagen(path)
        img_limpia = remover_pelo(img_orig)
        img_norm   = normalizar_color(corregir_iluminacion(img_limpia))
        mask, umbral = segmentar_otsu(img_norm)

        if save_masks:
            out_m = os.path.join(OUT_MASKS, clase)
            os.makedirs(out_m, exist_ok=True)
            cv2.imwrite(
                os.path.join(out_m, f"{nombre}_mask.png"),
                mask * 255
            )

        if save_figs:
            guardar_figura(img_orig, img_limpia, img_norm, mask,
                           nombre, clase, umbral, OUT_FIGS)

        print(f"  [{i:>3}/{len(imagenes)}] {fname}"
              f"  | umbral Otsu: {umbral:.0f}"
              f"  | píxeles lesión: {mask.sum()}")

    print(f"\n  ✓ {clase} listo.\n")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Preprocesamiento HAM10000 mel/nv — Otsu")
    parser.add_argument("--n",          type=int,  default=20,
                        help="Imágenes por clase (default: 20)")
    parser.add_argument("--save_masks", action="store_true",
                        help="Guardar máscaras binarias en segmentacion_mascaras/")
    parser.add_argument("--save_figs",  action="store_true",
                        help="Guardar figuras comparativas en segmentacion_figs/")
    args = parser.parse_args()

    print("\n=== Preprocesamiento HAM10000 — Grupo 8 ===")
    print(f"  Método: Otsu (1 objeto)  |  N por clase: {args.n}")
    print(f"  Guardar máscaras: {args.save_masks}  |  Guardar figuras: {args.save_figs}")

    for clase, carpeta in CLASES.items():
        if not os.path.isdir(carpeta):
            print(f"[ADVERTENCIA] Carpeta no encontrada: {carpeta}")
            continue
        procesar_clase(clase, carpeta, args.n, args.save_masks, args.save_figs)

    print("=== Fin del preprocesamiento ===\n")


if __name__ == "__main__":
    main()
