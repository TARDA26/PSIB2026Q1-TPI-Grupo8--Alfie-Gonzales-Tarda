#!/usr/bin/env python3
"""
Explorador Dermoscópico NV / MEL — Grupo 8
PSIB 2025-Q2 · Alfie, Gonzales, Tardá

Uso: python3 visualizador.py
Requiere: pip install pillow
"""

import tkinter as tk
import csv, random
from pathlib import Path
from PIL import Image, ImageTk, ImageDraw, ImageFont

# ══════════════════════════════════════════════════════════════════════
#  PATHS
# ══════════════════════════════════════════════════════════════════════
BASE      = Path(__file__).parent / "dataverse_files"
NV_DIR    = BASE / "nv"
MEL_DIR   = BASE / "mel"
ZONES_NV  = BASE / "zone_overlays" / "nv"
ZONES_MEL = BASE / "zone_overlays" / "mel"
CSV_PATH  = BASE / "features.csv"

# ══════════════════════════════════════════════════════════════════════
#  PALETA
# ══════════════════════════════════════════════════════════════════════
C = dict(
    bg       = "#0d1117",
    surface  = "#161b27",
    surface2 = "#1e2535",
    border   = "#2a3040",
    text     = "#e0e6f0",
    muted    = "#7a8499",
    mel      = "#f87171",
    nv       = "#34d399",
    blue     = "#60a5fa",
    amber    = "#fbbf24",
    zone_int = "#3b82f6",
    zone_an  = "#fbbf24",
    zone_sk  = "#34d399",
)

def hex2rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

# ══════════════════════════════════════════════════════════════════════
#  DATOS
# ══════════════════════════════════════════════════════════════════════
features: dict = {}
with open(CSV_PATH) as f:
    for row in csv.DictReader(f):
        img_id = row["imagen"]
        rec = {"clase": row["clase"]}
        for k, v in row.items():
            if k in ("clase", "imagen"):
                continue
            try:    rec[k] = float(v)
            except: rec[k] = v
        features[img_id] = rec

# Medias por clase (del informe)
MEANS = {
    "asim_total":               {"mel": 0.2526, "nv": 0.3352},
    "asim_vertical":            {"mel": 0.2109, "nv": 0.3015},
    "asim_horizontal":          {"mel": 0.2943, "nv": 0.3688},
    "circularidad":             {"mel": 5.2508, "nv": 2.8995},
    "compacidad":               {"mel": 5.2508, "nv": 2.8995},
    "var_radial":               {"mel": 0.2807, "nv": 0.2065},
    "dim_fractal":              {"mel": 1.0311, "nv": 0.9913},
    "hu_1":                     {"mel": 2.9899, "nv": 3.0909},
    "hu_2":                     {"mel": 6.8516, "nv": 7.2055},
    "hu_3":                     {"mel": 9.5837, "nv": 9.7980},
    "hu_4":                     {"mel": 9.7527, "nv": 9.8967},
    "glcm_contrast":            {"mel": 16.354, "nv": 11.925},
    "glcm_homogeneity":         {"mel": 0.8319, "nv": 0.8659},
    "glcm_energy":              {"mel": 0.7781, "nv": 0.8257},
    "glcm_correlation":         {"mel": 0.9293, "nv": 0.9294},
    "glcm_entropy":             {"mel": 2.8792, "nv": 2.3097},
    "lbp_entropy":              {"mel": 3.6259, "nv": 3.5539},
    "lbp_energy":               {"mel": 0.1262, "nv": 0.1375},
    "zona_int_L_mean":          {"mel": 73.495, "nv": 58.965},
    "zona_int_L_std":           {"mel": 49.931, "nv": 44.652},
    "zona_int_a_mean":          {"mel": 135.45, "nv": 133.86},
    "zona_int_a_std":           {"mel": 6.7268, "nv": 6.2683},
    "zona_int_b_mean":          {"mel": 132.41, "nv": 134.08},
    "zona_int_b_std":           {"mel": 7.7858, "nv": 7.0215},
    "zona_anular_L_mean":       {"mel": 98.020, "nv": 94.521},
    "zona_anular_L_std":        {"mel": 45.954, "nv": 45.577},
    "zona_anular_a_mean":       {"mel": 135.20, "nv": 133.84},
    "zona_anular_b_mean":       {"mel": 135.59, "nv": 137.14},
    "zona_piel_L_mean":         {"mel": 190.46, "nv": 187.31},
    "ratio_int_anular_L":       {"mel": 0.7268, "nv": 0.6130},
    "ratio_lesion_piel_L":      {"mel": 0.5168, "nv": 0.5082},
    "ratio_int_anular_a":       {"mel": 1.0022, "nv": 1.0008},
    "ratio_int_anular_b":       {"mel": 0.9775, "nv": 0.9785},
    "ratio_lesion_piel_a":      {"mel": 1.0304, "nv": 1.0267},
    "ratio_lesion_piel_b":      {"mel": 1.0206, "nv": 1.0185},
    "indice_abrupticidad":      {"mel": 1.3998, "nv": 1.4197},
    "grad_borde_mean":          {"mel": 98.697, "nv": 97.627},
    "grad_borde_std":           {"mel": 73.365, "nv": 71.833},
    "grad_borde_max":           {"mel": 758.70, "nv": 681.62},
    "heterog_cv_cuadrantes":    {"mel": 0.1336, "nv": 0.1562},
    "heterog_rango_cuadrantes": {"mel": 0.3327, "nv": 0.3831},
    "heterog_var_local_mean":   {"mel": 736.40, "nv": 670.41},
    "heterog_var_local_std":    {"mel": 642.49, "nv": 594.70},
    "color_entropia_L":         {"mel": 0.9186, "nv": 0.9010},
    "colores_dominantes":       {"mel": 5.7600, "nv": 5.7800},
    "rgb_r_mean":               {"mel": 95.257, "nv": 86.163},
    "rgb_g_mean":               {"mel": 76.655, "nv": 69.463},
    "rgb_b_mean":               {"mel": 71.749, "nv": 60.979},
    "lab_l_mean":               {"mel": 86.460, "nv": 77.828},
    "lab_a_mean":               {"mel": 135.32, "nv": 133.83},
    "lab_b__mean":              {"mel": 134.10, "nv": 135.70},
    "hsv_h_mean":               {"mel": 55.552, "nv": 39.007},
    "hsv_s_mean":               {"mel": 96.647, "nv": 94.198},
    "hsv_v_mean":               {"mel": 99.505, "nv": 88.856},
}

SIG = {
    "circularidad": "p<0.001", "compacidad": "p<0.001",
    "var_radial": "p<0.001",   "dim_fractal": "p<0.001",
    "asim_total": "p=0.002",   "asim_vertical": "p<0.001", "asim_horizontal": "p=0.005",
    "glcm_contrast": "p=0.002","glcm_homogeneity": "p=0.014", "glcm_energy": "p=0.03",
    "glcm_entropy": "p=0.008", "lbp_entropy": "p=0.037",  "lbp_energy": "p=0.04",
    "zona_int_L_mean": "p<0.001", "zona_int_L_std": "p<0.001",
    "zona_int_a_mean": "p=0.005",
    "zona_anular_L_mean": "p=0.02",
    "ratio_int_anular_L": "p=0.004",
    "indice_abrupticidad": "p=0.014",
    "heterog_cv_cuadrantes": "p=0.037", "heterog_rango_cuadrantes": "p=0.047",
}

NAMES = {
    "asim_total": "Asimetría total",
    "asim_vertical": "Asimetría vertical",
    "asim_horizontal": "Asimetría horiz.",
    "circularidad": "Circularidad",
    "compacidad": "Compacidad",
    "var_radial": "Varianza radial",
    "dim_fractal": "Dim. fractal",
    "hu_1": "Hu 1", "hu_2": "Hu 2", "hu_3": "Hu 3", "hu_4": "Hu 4",
    "glcm_contrast": "Contraste GLCM",
    "glcm_homogeneity": "Homogeneidad GLCM",
    "glcm_energy": "Energía GLCM",
    "glcm_correlation": "Correlación GLCM",
    "glcm_entropy": "Entropía GLCM",
    "lbp_entropy": "Entropía LBP",
    "lbp_energy": "Energía LBP",
    "zona_int_L_mean": "L* μ interior",
    "zona_int_L_std": "L* σ interior",
    "zona_int_a_mean": "a* μ interior",
    "zona_int_a_std": "a* σ interior",
    "zona_int_b_mean": "b* μ interior",
    "zona_int_b_std": "b* σ interior",
    "zona_anular_L_mean": "L* μ anular",
    "zona_anular_L_std": "L* σ anular",
    "zona_anular_a_mean": "a* μ anular",
    "zona_anular_b_mean": "b* μ anular",
    "zona_piel_L_mean": "L* μ piel",
    "ratio_int_anular_L": "Ratio int/anular L*",
    "ratio_lesion_piel_L": "Ratio lesión/piel L*",
    "ratio_int_anular_a": "Ratio int/anular a*",
    "ratio_int_anular_b": "Ratio int/anular b*",
    "ratio_lesion_piel_a": "Ratio lesión/piel a*",
    "ratio_lesion_piel_b": "Ratio lesión/piel b*",
    "indice_abrupticidad": "Índice abrupticidad",
    "grad_borde_mean": "Gradiente borde μ",
    "grad_borde_std": "Gradiente borde σ",
    "grad_borde_max": "Gradiente borde máx",
    "heterog_cv_cuadrantes": "CV cuadrantes",
    "heterog_rango_cuadrantes": "Rango cuadrantes",
    "heterog_var_local_mean": "Varianza local μ",
    "heterog_var_local_std": "Varianza local σ",
    "color_entropia_L": "Entropía color L*",
    "colores_dominantes": "Colores dominantes",
    "rgb_r_mean": "R μ", "rgb_g_mean": "G μ", "rgb_b_mean": "B μ",
    "rgb_r_std": "R σ",  "rgb_g_std": "G σ",  "rgb_b_std": "B σ",
    "lab_l_mean": "L* μ", "lab_a_mean": "a* μ", "lab_b__mean": "b* μ",
    "lab_l_std": "L* σ",  "lab_a_std": "a* σ",  "lab_b__std": "b* σ",
    "hsv_h_mean": "H μ",  "hsv_s_mean": "S μ",  "hsv_v_mean": "V μ",
    "hsv_h_std": "H σ",   "hsv_s_std": "S σ",   "hsv_v_std": "V σ",
}

# ── Descriptores clave (gráficos) ────────────────────────────────────
# Los más discriminativos del trabajo, con sus p-valores más bajos
KEY_STATS = [
    # (key, descripción larga del significado)
    ("circularidad",          "Contorno más irregular → MEL (p<0.001)"),
    ("var_radial",            "Más variación del borde → MEL (p<0.001)"),
    ("dim_fractal",           "Borde más complejo → MEL (p<0.001)"),
    ("asim_total",            "Forma más asimétrica → NV (p=0.002)"),
    ("zona_int_L_mean",       "Centro más brillante → MEL (p<0.001)"),
    ("zona_int_L_std",        "Centro más heterogéneo → MEL (p<0.001)"),
    ("ratio_int_anular_L",    "Centro/periferia más alto → MEL (p=0.004)"),
    ("glcm_contrast",         "Textura más contrastada → MEL (p=0.002)"),
    ("glcm_entropy",          "Textura más desordenada → MEL (p=0.008)"),
    ("glcm_homogeneity",      "Textura menos uniforme → MEL (p=0.014)"),
    ("indice_abrupticidad",   "Borde más difuso → MEL (p=0.014)"),
    ("heterog_cv_cuadrantes", "Heterog. difusa (bajo CV) → MEL (p=0.037)"),
]

# ── Grupos de datos adicionales ───────────────────────────────────────
DATA_GROUPS = [
    ("A — Asimetría",       ["asim_vertical","asim_horizontal","hu_1","hu_2","hu_3","hu_4"]),
    ("B — Borde",           ["compacidad","grad_borde_mean","grad_borde_std","grad_borde_max"]),
    ("C — Color",           ["lab_l_mean","lab_a_mean","lab_b__mean",
                              "rgb_r_mean","rgb_g_mean","rgb_b_mean",
                              "hsv_h_mean","hsv_s_mean","hsv_v_mean",
                              "color_entropia_L","colores_dominantes"]),
    ("D — Textura",         ["lbp_entropy","lbp_energy","glcm_correlation","glcm_energy"]),
    ("E — Zonas",           ["zona_int_a_mean","zona_int_b_mean","zona_int_a_std","zona_int_b_std",
                              "zona_anular_L_std","zona_anular_a_mean","zona_anular_b_mean",
                              "zona_piel_L_mean",
                              "ratio_lesion_piel_L","ratio_int_anular_a","ratio_int_anular_b",
                              "ratio_lesion_piel_a","ratio_lesion_piel_b"]),
    ("G — Heterogeneidad",  ["heterog_rango_cuadrantes","heterog_var_local_mean","heterog_var_local_std"]),
]

# ══════════════════════════════════════════════════════════════════════
#  APLICACIÓN
# ══════════════════════════════════════════════════════════════════════
class App(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("Explorador Dermoscópico NV / MEL — Grupo 8")
        self.geometry("1540x880")
        self.minsize(1100, 700)
        self.configure(bg=C["bg"])

        self.current_id     = None
        self.current_filter = "all"
        self.zones          = {"interior": False, "anular": False, "piel": False}
        self._photo         = None
        self._base_pil      = None
        self._zone_pil      = {}
        self.listbox_ids    = []
        self._bar_jobs      = []   # canvas widgets that need redraw

        self._build_ui()
        self.after(120, self.load_random)

    # ─────────────────────────────────────────────────────────
    #  BUILD UI
    # ─────────────────────────────────────────────────────────
    def _build_ui(self):
        self._build_header()
        body = tk.Frame(self, bg=C["bg"])
        body.pack(fill="both", expand=True)
        self._build_sidebar(body)
        self._build_stats(body)
        self._build_center(body)

    # ── Header ──────────────────────────────────────────────
    def _build_header(self):
        hdr = tk.Frame(self, bg=C["surface"], height=48)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)

        tk.Label(hdr, text="Explorador Dermoscópico",
                 bg=C["surface"], fg=C["text"],
                 font=("Helvetica", 14, "bold")).pack(side="left", padx=14)
        tk.Label(hdr, text="NV vs MEL  ·  HAM10000  ·  Grupo 8",
                 bg=C["surface"], fg=C["muted"],
                 font=("Helvetica", 10)).pack(side="left")

        btn_f = tk.Frame(hdr, bg=C["surface"])
        btn_f.pack(side="right", padx=12)

        self._filter_btns = {}
        for fid, label, col in [("all","Todos",C["blue"]),
                                  ("nv","● NV",  C["nv"]),
                                  ("mel","● MEL", C["mel"])]:
            b = tk.Button(btn_f, text=label, cursor="hand2",
                          bg=C["surface2"], fg=C["muted"],
                          activebackground=C["border"], relief="flat",
                          font=("Helvetica", 10), padx=13, pady=5,
                          command=lambda f=fid: self.set_filter(f))
            b.pack(side="left", padx=3, pady=10)
            self._filter_btns[fid] = (b, col)

        tk.Button(btn_f, text="⚄  Aleatorio", cursor="hand2",
                  bg=C["surface2"], fg=C["blue"],
                  activebackground=C["border"], relief="flat",
                  font=("Helvetica", 10), padx=13, pady=5,
                  command=self.load_random
                  ).pack(side="left", padx=3, pady=10)

        self._refresh_filter_btns()

    def _refresh_filter_btns(self):
        for fid, (btn, col) in self._filter_btns.items():
            if fid == self.current_filter:
                btn.configure(fg=col, relief="solid", bd=1)
            else:
                btn.configure(fg=C["muted"], relief="flat", bd=0)

    # ── Sidebar ─────────────────────────────────────────────
    def _build_sidebar(self, parent):
        sb = tk.Frame(parent, bg=C["surface"], width=198)
        sb.pack(side="left", fill="y")
        sb.pack_propagate(False)

        top = tk.Frame(sb, bg=C["surface"])
        top.pack(fill="x", padx=8, pady=6)
        tk.Label(top, text="IMÁGENES", bg=C["surface"], fg=C["muted"],
                 font=("Helvetica", 8, "bold")).pack(anchor="w")

        self.search_var = tk.StringVar()
        self.search_var.trace_add("write", lambda *_: self._rebuild_list())
        tk.Entry(top, textvariable=self.search_var,
                 bg=C["surface2"], fg=C["text"],
                 insertbackground=C["text"], relief="flat",
                 font=("Helvetica", 10)).pack(fill="x", pady=4)

        tk.Frame(sb, bg=C["border"], height=1).pack(fill="x")

        lf = tk.Frame(sb, bg=C["surface"])
        lf.pack(fill="both", expand=True)

        scr = tk.Scrollbar(lf, bg=C["surface2"], troughcolor=C["surface"], width=8)
        scr.pack(side="right", fill="y")

        self.listbox = tk.Listbox(
            lf, bg=C["surface"], fg=C["muted"],
            selectbackground=C["surface2"], selectforeground=C["text"],
            activestyle="none", relief="flat", bd=0, highlightthickness=0,
            font=("Courier", 10), yscrollcommand=scr.set)
        self.listbox.pack(side="left", fill="both", expand=True)
        scr.config(command=self.listbox.yview)
        self.listbox.bind("<<ListboxSelect>>", self._on_list_select)

    def _rebuild_list(self):
        search = self.search_var.get().strip().lower()
        self.listbox.delete(0, "end")
        self.listbox_ids = []
        for img_id, feat in features.items():
            cls = feat["clase"]
            if self.current_filter != "all" and cls != self.current_filter:
                continue
            if search and search not in img_id.lower():
                continue
            prefix = "● " if cls == "mel" else "○ "
            self.listbox.insert("end", f"{prefix}{img_id.replace('ISIC_','')}")
            self.listbox_ids.append(img_id)
            idx = len(self.listbox_ids) - 1
            col = C["mel"] if cls == "mel" else C["nv"]
            self.listbox.itemconfig(idx,
                fg=col if img_id == self.current_id else C["muted"])

        if self.current_id in self.listbox_ids:
            idx = self.listbox_ids.index(self.current_id)
            self.listbox.selection_clear(0, "end")
            self.listbox.selection_set(idx)
            self.listbox.see(idx)
            cls = features[self.current_id]["clase"]
            self.listbox.itemconfig(idx, fg=C["mel"] if cls=="mel" else C["nv"])

    def _on_list_select(self, _e):
        sel = self.listbox.curselection()
        if sel and sel[0] < len(self.listbox_ids):
            self.load_image(self.listbox_ids[sel[0]])

    # ── Center ──────────────────────────────────────────────
    def _build_center(self, parent):
        center = tk.Frame(parent, bg=C["bg"])
        center.pack(side="left", fill="both", expand=True)

        self.img_canvas = tk.Canvas(center, bg="#08090f",
                                    highlightthickness=0, cursor="crosshair")
        self.img_canvas.pack(fill="both", expand=True)
        self.img_canvas.bind("<Configure>", lambda _: self._render_image())

        # Zone bar
        zbar = tk.Frame(center, bg=C["surface"], height=46)
        zbar.pack(fill="x")
        zbar.pack_propagate(False)

        tk.Label(zbar, text="ZONAS:", bg=C["surface"], fg=C["muted"],
                 font=("Helvetica", 9, "bold")).pack(side="left", padx=12, pady=13)

        self._zone_btns = {}
        for zid, zlabel, zcol in [
            ("interior", "● Interior (50%)", C["zone_int"]),
            ("anular",   "● Corona anular",  C["zone_an"]),
            ("piel",     "● Franja de piel", C["zone_sk"]),
        ]:
            btn = tk.Button(zbar, text=zlabel, cursor="hand2",
                            bg=C["surface2"], fg=C["muted"],
                            activebackground=C["border"], relief="flat",
                            font=("Helvetica", 10), padx=11, pady=5,
                            command=lambda z=zid: self.toggle_zone(z))
            btn.pack(side="left", padx=4, pady=9)
            self._zone_btns[zid] = (btn, zcol)

    # ── Stats panel ─────────────────────────────────────────
    def _build_stats(self, parent):
        panel = tk.Frame(parent, bg=C["surface"], width=350)
        panel.pack(side="right", fill="y")
        panel.pack_propagate(False)

        # Scrollable area
        outer = tk.Frame(panel, bg=C["surface"])
        outer.pack(fill="both", expand=True)

        vscr = tk.Scrollbar(outer, bg=C["surface2"], troughcolor=C["surface"], width=8)
        vscr.pack(side="right", fill="y")

        self.stats_cv = tk.Canvas(outer, bg=C["surface"],
                                   highlightthickness=0,
                                   yscrollcommand=vscr.set)
        self.stats_cv.pack(side="left", fill="both", expand=True)
        vscr.config(command=self.stats_cv.yview)

        self.stats_frame = tk.Frame(self.stats_cv, bg=C["surface"])
        self._sw = self.stats_cv.create_window((0, 0), window=self.stats_frame, anchor="nw")

        self.stats_frame.bind("<Configure>",
            lambda _: self.stats_cv.config(scrollregion=self.stats_cv.bbox("all")))
        self.stats_cv.bind("<Configure>",
            lambda e: self.stats_cv.itemconfig(self._sw, width=e.width))

        for ev in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            self.stats_cv.bind(ev, self._on_scroll)

    def _on_scroll(self, event):
        if event.num == 4:   self.stats_cv.yview_scroll(-1, "units")
        elif event.num == 5: self.stats_cv.yview_scroll( 1, "units")
        else: self.stats_cv.yview_scroll(int(-1*(event.delta/120)), "units")

    # ─────────────────────────────────────────────────────────
    #  LÓGICA
    # ─────────────────────────────────────────────────────────
    def set_filter(self, f):
        self.current_filter = f
        self._refresh_filter_btns()
        self._rebuild_list()

    def load_random(self):
        pool = [i for i,v in features.items()
                if self.current_filter == "all" or v["clase"] == self.current_filter]
        self.load_image(random.choice(pool))

    def load_image(self, img_id: str):
        self.current_id = img_id
        cls = features[img_id]["clase"]

        img_path = (NV_DIR if cls == "nv" else MEL_DIR) / f"{img_id}.jpg"
        try:
            self._base_pil = Image.open(img_path).convert("RGBA")
        except Exception as e:
            print(f"Error abriendo imagen: {e}"); return

        zone_dir = ZONES_NV if cls == "nv" else ZONES_MEL
        self._zone_pil = {}
        for z in ("interior", "anular", "piel"):
            p = zone_dir / f"{img_id}_{z}.png"
            if p.exists():
                self._zone_pil[z] = Image.open(p).convert("RGBA")

        self._render_image()
        self._rebuild_list()
        self._render_stats()

    def _render_image(self):
        if self._base_pil is None:
            return

        img = self._base_pil.copy()
        for z in ("interior", "anular", "piel"):
            if self.zones[z] and z in self._zone_pil:
                img = Image.alpha_composite(img, self._zone_pil[z])

        rgb = img.convert("RGB")

        cw = self.img_canvas.winfo_width()  or 740
        ch = self.img_canvas.winfo_height() or 540
        iw, ih = rgb.size
        scale = min(cw / iw, ch / ih, 2.0)
        nw, nh = int(iw * scale), int(ih * scale)
        rgb = rgb.resize((nw, nh), Image.LANCZOS)

        self._draw_badge(rgb, features[self.current_id]["clase"], self.current_id)

        self._photo = ImageTk.PhotoImage(rgb)
        self.img_canvas.delete("all")
        self.img_canvas.create_image(cw//2, ch//2, image=self._photo, anchor="center")

    def _draw_badge(self, img: Image.Image, cls: str, img_id: str):
        draw = ImageDraw.Draw(img)
        col  = hex2rgb(C["mel"] if cls == "mel" else C["nv"])
        text = cls.upper()

        # Large badge — top-left
        pad_x, pad_y = 14, 8
        box_w, box_h = 90, 42
        draw.rectangle([8, 8, 8 + box_w, 8 + box_h],
                        fill=(13, 17, 25), outline=col, width=2)
        # Big class text
        try:
            fnt = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 22)
        except:
            fnt = ImageFont.load_default()
        draw.text((8 + pad_x, 8 + pad_y), text, fill=col, font=fnt)

        # ID — bottom left, small
        draw.rectangle([6, img.height - 22, 6 + len(img_id)*7 + 8, img.height - 6],
                        fill=(13, 17, 23, 180))
        draw.text((10, img.height - 20), img_id, fill=(90, 100, 120))

    def toggle_zone(self, z: str):
        self.zones[z] = not self.zones[z]
        btn, col = self._zone_btns[z]
        if self.zones[z]:
            btn.configure(fg=col, bg="#1a2540", relief="solid", bd=1)
        else:
            btn.configure(fg=C["muted"], bg=C["surface2"], relief="flat", bd=0)
        self._render_image()

    # ─────────────────────────────────────────────────────────
    #  RENDER STATS
    # ─────────────────────────────────────────────────────────
    def _render_stats(self):
        for w in self.stats_frame.winfo_children():
            w.destroy()
        self._bar_jobs = []

        feat = features[self.current_id]
        cls  = feat["clase"]

        # ── Section title ─────────────────────────
        cls_col = C["mel"] if cls == "mel" else C["nv"]
        hdr = tk.Frame(self.stats_frame, bg=C["surface2"])
        hdr.pack(fill="x", pady=(0, 2))
        tk.Label(hdr, text="ESTADÍSTICAS DE LA IMAGEN",
                 bg=C["surface2"], fg=C["muted"],
                 font=("Helvetica", 8, "bold"), padx=10, pady=6
                 ).pack(side="left")
        tk.Label(hdr, text=f"Clase real: {cls.upper()}",
                 bg=C["surface2"], fg=cls_col,
                 font=("Helvetica", 9, "bold"), padx=10
                 ).pack(side="right")

        # ── KEY STATS (gráficos) ──────────────────
        sec = tk.Frame(self.stats_frame, bg=C["surface"])
        sec.pack(fill="x")
        tk.Label(sec, text="DESCRIPTORES CLAVE",
                 bg=C["border"], fg=C["blue"],
                 font=("Helvetica", 8, "bold"), padx=10, pady=4
                 ).pack(fill="x")

        for key, description in KEY_STATS:
            v = feat.get(key)
            if v is None: continue
            self._render_key_stat(sec, key, v, description)

        tk.Frame(self.stats_frame, bg=C["border"], height=2).pack(fill="x", pady=4)

        # ── DATA GROUPS (texto) ───────────────────
        tk.Label(self.stats_frame, text="DATOS COMPLETOS",
                 bg=C["border"], fg=C["muted"],
                 font=("Helvetica", 8, "bold"), padx=10, pady=4
                 ).pack(fill="x")

        for group_name, keys in DATA_GROUPS:
            self._render_data_group(group_name, keys, feat)

        tk.Frame(self.stats_frame, bg=C["bg"], height=12).pack(fill="x")

    # ── Key stat widget ──────────────────────────
    def _render_key_stat(self, parent, key: str, value: float, description: str):
        m = MEANS.get(key)
        sig = SIG.get(key, "")
        name = NAMES.get(key, key)

        card = tk.Frame(parent, bg=C["surface2"], padx=10, pady=6)
        card.pack(fill="x", padx=6, pady=3)

        # Top row: name + value
        top = tk.Frame(card, bg=C["surface2"])
        top.pack(fill="x")
        tk.Label(top, text=name, bg=C["surface2"], fg=C["text"],
                 font=("Helvetica", 10, "bold"), anchor="w").pack(side="left")
        if sig:
            tk.Label(top, text=sig, bg=C["surface2"], fg=C["amber"],
                     font=("Helvetica", 8), padx=4).pack(side="left")

        val_s = f"{value:.4f}" if abs(value) < 100 else f"{value:.2f}"
        tk.Label(top, text=val_s, bg=C["surface2"], fg=C["text"],
                 font=("Courier", 11, "bold")).pack(side="right")

        # Description (interpretation)
        tk.Label(card, text=description, bg=C["surface2"], fg=C["muted"],
                 font=("Helvetica", 8), anchor="w").pack(fill="x", pady=(2, 4))

        # Bar canvas
        if m:
            bar = tk.Canvas(card, height=20, bg=C["surface"],
                             highlightthickness=0, bd=0)
            bar.pack(fill="x", pady=(2, 0))

            def draw_bar(event=None, _bar=bar, _m=m, _v=value, _key=key):
                _bar.delete("all")
                w = _bar.winfo_width()
                if w < 20: return

                lo  = min(_m["nv"], _m["mel"]) * 0.6
                hi  = max(_m["nv"], _m["mel"]) * 1.4
                rng = hi - lo or 1

                nv_x  = int((_m["nv"] - lo) / rng * w)
                mel_x = int((_m["mel"] - lo) / rng * w)
                val_x = max(2, min(w-2, int((_v - lo) / rng * w)))

                # Track background
                _bar.create_rectangle(0, 7, w, 13,
                                       fill=C["border"], outline="")

                # Colored fill between NV and MEL
                x0, x1 = min(nv_x, mel_x), max(nv_x, mel_x)
                _bar.create_rectangle(x0, 7, x1, 13,
                                       fill=C["surface2"], outline="")

                # NV marker (green triangle above)
                _bar.create_polygon(nv_x-5, 20, nv_x+5, 20, nv_x, 13,
                                     fill=C["nv"], outline="")
                _bar.create_text(nv_x, 8, text="NV", fill=C["nv"],
                                  font=("Helvetica", 7, "bold"), anchor="s")

                # MEL marker (red triangle above)
                _bar.create_polygon(mel_x-5, 20, mel_x+5, 20, mel_x, 13,
                                     fill=C["mel"], outline="")
                _bar.create_text(mel_x, 8, text="MEL", fill=C["mel"],
                                  font=("Helvetica", 7, "bold"), anchor="s")

                # Current value — white vertical bar (taller, prominent)
                _bar.create_rectangle(val_x-3, 2, val_x+3, 18,
                                       fill="white", outline="")

                # Mean value labels
                _bar.create_text(nv_x, 20, text=f"{_m['nv']:.3g}",
                                  fill=C["nv"], font=("Helvetica", 7), anchor="n")
                _bar.create_text(mel_x, 20, text=f"{_m['mel']:.3g}",
                                  fill=C["mel"], font=("Helvetica", 7), anchor="n")

            bar.bind("<Configure>", draw_bar)
            # Force draw after layout
            bar.after(50, lambda b=bar: draw_bar(None))

    # ── Data group (texto compacto) ──────────────
    def _render_data_group(self, group_name: str, keys: list, feat: dict):
        # Header
        gh = tk.Frame(self.stats_frame, bg=C["surface2"])
        gh.pack(fill="x", pady=(4, 0))
        tk.Label(gh, text=group_name, bg=C["surface2"], fg=C["muted"],
                 font=("Helvetica", 8, "bold"), padx=10, pady=3
                 ).pack(side="left")

        # Rows
        for key in keys:
            v = feat.get(key)
            if v is None: continue
            row = tk.Frame(self.stats_frame, bg=C["surface"])
            row.pack(fill="x", padx=6)
            name = NAMES.get(key, key)
            sig  = SIG.get(key, "")
            val_s = f"{v:.4f}" if isinstance(v, float) and abs(v) < 100 else (
                    f"{v:.1f}" if isinstance(v, float) else str(v))

            label_text = name + (f"  {sig}" if sig else "")
            tk.Label(row, text=label_text, bg=C["surface"], fg=C["muted"],
                     font=("Helvetica", 9), width=24, anchor="w",
                     padx=8).pack(side="left")
            tk.Label(row, text=val_s, bg=C["surface"], fg=C["text"],
                     font=("Courier", 9, "bold"), anchor="e",
                     padx=8).pack(side="right")

        tk.Frame(self.stats_frame, bg=C["border"], height=1).pack(fill="x", pady=2)


# ══════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    app = App()
    app.mainloop()
