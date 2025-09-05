import time
import cv2
import numpy as np
import pyautogui
import threading
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from dataclasses import dataclass
from typing import List, Optional
import json
import os
from PIL import Image, ImageTk
import subprocess, sys
import shutil
# ---------- Config ----------
FACTION: Optional[str] = None

THRESHOLD = 0.9
SCALES = np.linspace(0.6, 1.4, 11)
IOU_NMS = 0.35
SAVE_DEBUG = "last_detection.png"
ROI: Optional[tuple] = None  
CONFIG_FILE = "config.json"
COORDS_FILE = "berimond_ui_coords.json"
CONFIG_FILEB = "config_berimond.json"
BONUS_ACTIVO = False
POPUP_GRACE = 1.0
ATTACK_ACTIVE = False
# ---------- Data ----------
@dataclass
class Detection:
    x: int
    y: int
    w: int
    h: int
    score: float
    label: str

# ---------- Estado Global ----------
RUNNING = False
RUNNING_BERIMOND = False
WATCHER_ACTIVE = True
N_COMANDANTES = 1
TIMER = 10
POPUP_ACTIVE = False
COOLDOWN_LOCKS = {}
camps_detected: List[Detection] = []
# templates
templates = {}
offer_templates = {}
comandante_delays = {}
horse_choice = "monedas"  # por defecto
# GUI references
root = None
combo_camps = None
spin_comandantes = None
spin_timer = None
combo_comandante = None
entry_delay = None
text_log = None
label_status = None

# ---------- Config ----------
TROOP_IMAGES = {
    "Valquiria de asalto": "assets/recruit/valkyrie_ranger.png",
    "Valquiria ballestera": "assets/recruit/valkyrie_sniper.png",
    "Doncella con escudo": "assets/recruit/shield_maiden.png",
    "Protectora del norte": "assets/recruit/protector_of_the_north.png",
    "Arquero de arco largo reliquia": "assets/recruit/relic_longbowman.png",
    "Soldado con martillo reliquia": "assets/recruit/relic_hammerman.png",
    "Arquero reliquia": "assets/recruit/relic_shortbowman.png",
    "Hachero reliquia": "assets/recruit/relic_axeman.png",
    "Alabardero veterano": "assets/recruit/veteran_halberdier.png",
    "Espadachin con mandoble veterano": "assets/recruit/veteran_two_handed_swordsman.png",
    "Arquero de arco largo veterano": "assets/recruit/veteran_longbowman.png",
    "Ballestero fuerte veterano": "assets/recruit/veteran_heavy_crossbowman.png",
    "Espadachín veterano": "assets/recruit/veteran_swordsman.png",
    "Soldado Arquero": "assets/recruit/archer.png",
    "Ballestero veterano": "assets/recruit/veteran_crossbowman.png",
    "Arquero veterano": "assets/recruit/veteran_bowman.png",
    "Guerrero con maza veterano": "assets/recruit/veteran_maceman.png",
    "Piquero veterano": "assets/recruit/veteran_spearman.png",
    "Auxiliar A": "assets/recruit/auxiliar_melee.png",
    "Auxiliar B": "assets/recruit/auxiliar_ranged.png",
}
TROOP_IMAGES_ORIGINAL = {
    "Valquiria de asalto": "assets/recruit/originals/valkyrie_rangerO.png",
    "Valquiria ballestera": "assets/recruit/originals/valkyrie_sniperO.png",
    "Doncella con escudo": "assets/recruit/originals/shield_maidenO.png",
    "Protectora del norte": "assets/recruit/originals/protector_of_the_northO.png",
    "Arquero de arco largo reliquia": "assets/recruit/originals/relic_longbowmanO.png",
    "Soldado con martillo reliquia": "assets/recruit/originals/relic_hammermanO.png",
    "Arquero reliquia": "assets/recruit/originals/relic_shortbowmanO.png",
    "Hachero reliquia": "assets/recruit/originals/relic_axemanO.png",
    "Alabardero veterano": "assets/recruit/originals/veteran_halberdierO.png",
    "Espadachin con mandoble veterano": "assets/recruit/originals/veteran_two_handed_swordsmanO.png",
    "Arquero de arco largo veterano": "assets/recruit/originals/veteran_longbowmanO.png",
    "Ballestero fuerte veterano": "assets/recruit/originals/veteran_heavy_crossbowmanO.png",
    "Espadachín veterano": "assets/recruit/originals/veteran_swordsmanO.png",
    "Soldado Arquero": "assets/recruit/originals/archerO.png",
    "Ballestero veterano": "assets/recruit/originals/veteran_crossbowmanO.png",
    "Arquero veterano": "assets/recruit/originals/veteran_bowmanO.png",
    "Guerrero con maza veterano": "assets/recruit/originals/veteran_macemanO.png",
    "Piquero veterano": "assets/recruit/originals/veteran_spearmanO.png",
    "Auxiliar A": "assets/recruit/originals/auxiliar_meleeO.png",
    "Auxiliar B": "assets/recruit/originals/auxiliar_rangedO.png"
}
TROOP_TYPES = {
    "Valquiria de asalto": "distancia",
    "Valquiria ballestera": "distancia",
    "Doncella con escudo": "cuerpo",
    "Protectora del norte": "cuerpo",
    "Arquero de arco largo reliquia": "distancia",
    "Soldado con martillo reliquia": "cuerpo",
    "Arquero reliquia": "distancia",
    "Hachero reliquia": "cuerpo",
    "Alabardero veterano": "cuerpo",
    "Espadachin con mandoble veterano": "cuerpo",
    "Arquero de arco largo veterano": "distancia",
    "Ballestero fuerte veterano": "distancia",
    "Espadachín veterano": "cuerpo",
    "Soldado Arquero": "distancia",
    "Ballestero veterano": "distancia",
    "Arquero veterano": "distancia",
    "Guerrero con maza veterano": "cuerpo",
    "Piquero veterano": "cuerpo",
    "Auxiliar A": "cuerpo",
    "Auxiliar B": "distancia"
}

FILE_TO_TROOP = {
    "valkyrie_ranger": "Valquiria de asalto",
    "valkyrie_sniper": "Valquiria ballestera",
    "shield_maiden": "Doncella con escudo",
    "protector_of_the_north": "Protectora del norte",
    "relic_longbowman": "Arquero de arco largo reliquia",
    "relic_hammerman": "Soldado con martillo reliquia",
    "relic_shortbowman": "Arquero reliquia",
    "relic_axeman": "Hachero reliquia",
    "veteran_halberdier": "Alabardero veterano",
    "veteran_two_handed_swordsman": "Espadachin con mandoble veterano",
    "veteran_longbowman": "Arquero de arco largo veterano",
    "veteran_heavy_crossbowman": "Ballestero fuerte veterano",
    "veteran_swordsman": "Espadachín veterano",
    "archer": "Soldado Arquero",
    "veteran_crossbowman": "Ballestero veterano",
    "veteran_bowman": "Arquero veterano",
    "veteran_maceman": "Guerrero con maza veterano",
    "veteran_spearman": "Piquero veterano",
    "auxiliar_melee": "Auxiliar A",
    "auxiliar_ranged": "Auxiliar B"
}

TEMPLATE_PATHS = {
    "attack_button": "assets/attack/attack_button.png",
    "attack_icon": "assets/attack/attack_icon.png",
    "horse_gold_coins": "assets/attack/horse_gold_coins.png",
    "horse_premium": "assets/attack/horse_premium.png",
    "confirm_attack2": "assets/attack/confirm_attack2.png",
    "confirm_attack": "assets/attack/confirm_attack.png",
    "template_button": "assets/attack/template_button.png",
    "reduce_icon": "assets/cooldown/reduce_icon.png",
    "clock_1h": "assets/cooldown/clock_1h.png",
    "arrow_left" :"assets/cooldown/arrow_left.png",
    "clock_30": "assets/cooldown/clock_30m.png",
    "clock_1h_bonus": "assets/cooldown/clock_1h_bonus.png",
    "clock_30m_bonus": "assets/cooldown/clock_30m_bonus.png"
}

OUTPUT_JSON = "berimond_ui_coords.json"
CAPTURE_DIR = "capturas"
CAPTURE_DIR_R = "capturas_reclutamiento"

os.makedirs(CAPTURE_DIR, exist_ok=True)

@dataclass
class Coord:
    x: int
    y: int
    w: int
    h: int

with open(COORDS_FILE, "r") as f:
    COORDS = json.load(f)
    
# ---------- Cargar coordenadas previas ----------
if os.path.exists(OUTPUT_JSON):
    try:
        with open(OUTPUT_JSON, "r") as f:
            coords = json.load(f)
        print(f"🔄 Coordenadas cargadas desde {OUTPUT_JSON}: {coords}")
    except Exception as e:
        print(f"⚠️ Error cargando {OUTPUT_JSON}: {e}")
        coords = {}
else:
    coords = {}

# ---------- Utilities ----------
def safe_imread(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        messagebox.showerror("Error", f"No se pudo leer la imagen: {path}")
    return img

def detect_on_screen(template_path, confidence=THRESHOLD):
    screenshot = pyautogui.screenshot()
    screen_rgb = np.array(screenshot)
    screen_gray = cv2.cvtColor(screen_rgb, cv2.COLOR_RGB2GRAY)
    template = safe_imread(template_path)
    if template is None:
        return None
    res = cv2.matchTemplate(screen_gray, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    if max_val >= confidence:
        h, w = template.shape
        x, y = max_loc

        cv2.rectangle(screen_rgb, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imwrite(
            os.path.join(CAPTURE_DIR, f"{os.path.basename(template_path).split('.')[0]}.png"),
            cv2.cvtColor(screen_rgb, cv2.COLOR_RGB2BGR),
        )
        return Coord(x=x + w//2, y=y + h//2, w=w, h=h)
    return None

# ---------- GUI ----------
def capture_coord(name):
    messagebox.showinfo("Instrucciones", f"Muestra la ventana del juego y asegúrate de que {name} sea visible. Se intentará detectar ahora...")
    attempts = 0
    while attempts < 100:
        result = detect_on_screen(TEMPLATE_PATHS[name])
        if result:
            coords[name] = {"x": result.x, "y": result.y, "w": result.w, "h": result.h}
            messagebox.showinfo("Detectado", f"{name} detectado en ({result.x}, {result.y}) y captura guardada")
            btns[name].config(text=f"{name}: OK")
            print(f"✅ {name} -> ({result.x}, {result.y})")
            return
        else:
            attempts += 1
            time.sleep(0.2)
    messagebox.showwarning("No detectado", f"No se pudo detectar {name} después de varios intentos")

def save_coords():
    with open(OUTPUT_JSON, "w") as f:
        json.dump(coords, f, indent=4)
    messagebox.showinfo("Guardado", f"Coordenadas guardadas en {OUTPUT_JSON}")
    print(f"💾 Coordenadas guardadas: {coords}")

def add_template():
    filepaths = filedialog.askopenfilenames(
        title="Seleccionar imágenes",
        filetypes=[("Imágenes", "*.png *.jpg *.jpeg")]
    )
    if not filepaths:
        return

    added_count = 0

    for filepath in filepaths:
        name = os.path.splitext(os.path.basename(filepath))[0]

        if name not in TEMPLATE_PATHS:
            messagebox.showwarning(
                "Nombre no permitido",
                f"❌ El template '{name}' no existe en el programa, no se añadirá."
            )
            continue

        dest_path = os.path.join("assets/attack", os.path.basename(filepath))

        # Copiar archivo a assets/attack (sobrescribe si ya existe)
        try:
            shutil.copy(filepath, dest_path)
        except Exception as e:
            messagebox.showerror(
                "Error",
                f"No se pudo copiar {filepath} -> {dest_path}\n{e}"
            )
            continue

        # Actualizar ruta en TEMPLATE_PATHS
        TEMPLATE_PATHS[name] = dest_path

        # Actualizar botón existente (si es que ya existe)
        if name in btns:
            btns[name].config(text=f"{name}: No detectado")

        added_count += 1

    messagebox.showinfo(
        "Actualización completada",
        f"✅ Se actualizaron {added_count} imagenes existentes."
    )

    # Reubicar botones fijos
    add_btn.grid(row=len(btns), column=0, pady=10)
    save_btn.grid(row=len(btns)+1, column=0, pady=10)
    
def add_troop_images():
    """Permite al usuario añadir imágenes recortadas para TROOP_IMAGES usando nombres de archivo en inglés"""
    filepaths = filedialog.askopenfilenames(
        title="Seleccionar imágenes recortadas",
        filetypes=[("Imágenes", "*.png *.jpg *.jpeg")]
    )
    if not filepaths:
        return

    added_count = 0

    for filepath in filepaths:
        filename = os.path.basename(filepath)
        name_no_ext = os.path.splitext(filename)[0]

        # Traducir nombre de archivo a nombre de tropa en español
        if name_no_ext in FILE_TO_TROOP:
            troop_name = FILE_TO_TROOP[name_no_ext]
        else:
            messagebox.showwarning(
                "Nombre no permitido",
                f"❌ La tropa '{name_no_ext}' no está registrada en TROOP_IMAGES. No se añadirá."
            )
            continue

        # Destino dentro de la carpeta recruit
        dest_path = os.path.join("assets/recruit", filename)

        try:
            shutil.copy(filepath, dest_path)
        except Exception as e:
            messagebox.showerror(
                "Error al copiar",
                f"No se pudo copiar {filepath} -> {dest_path}\n{e}"
            )
            continue

        # Actualiza la ruta de la tropa recortada en TROOP_IMAGES
        TROOP_IMAGES[troop_name] = dest_path
        added_count += 1

    messagebox.showinfo(
        "Actualización completada",
        f"✅ Se actualizaron {added_count} imágenes recortadas para las tropas."
    )

# ---------- Main ----------
def coord_config_window():
    global coords, TEMPLATE_PATHS, btns, frame, add_btn, save_btn

    coord_root = tk.Toplevel()
    coord_root.title("Captura Coordenadas Berimond")

    frame = ttk.Frame(coord_root, padding=10)
    frame.grid(row=0, column=0)

    btns = {}
    for i, name in enumerate(TEMPLATE_PATHS.keys()):
        estado = "OK" if name in coords else "No detectado"
        btn = ttk.Button(frame, text=f"{name}: {estado}", command=lambda n=name: capture_coord(n), width=25)
        btn.grid(row=i, column=0, pady=5)
        btns[name] = btn

    add_btn = ttk.Button(frame, text="➕ Añadir nueva imagen", command=add_template)
    add_btn.grid(row=len(TEMPLATE_PATHS), column=0, pady=10)

    save_btn = ttk.Button(frame, text="Guardar coordenadas", command=save_coords)
    save_btn.grid(row=len(TEMPLATE_PATHS)+1, column=0, pady=10)

# ---------- Utilities ----------
def safe_imread(path: str):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"⚠️ No se pudo leer la imagen: {path}")
    return img

# ---------- Config dinámico de templates ----------
TEMPLATE_PATHS_BY_FACTION = {
    "nomadas": {
        "camp_normal": "assets/nomadas/camp_normal.png",
        "camp_fire": "assets/nomadas/camp_fire.png",
    },
    "samurais": {
        "samurai": "assets/samurais/samurai.png",
        "samurai defeated": "assets/samurais/samurai_defeated.png",
    },
    "berimond": {
        "destroyed_watchtower": "assets/berimond/watchtower_in_fire.png",
    },
    "islas": {
        "camp_normal": "assets/islas/camp_normal.png",
        "camp_fire": "assets/islas/camp_fire.png",
    },
    "fortalezas": {
        "fortaleza_dragon_normal": "assets/fortalezas/fortaleza_dragon_normal.png",
        "fortaleza_dragon_fire": "assets/fortalezas/fortaleza_dragon_fire.png",
        "fortaleza_desierto_normal": "assets/fortalezas/fortaleza_desierto_normal.png",
        "fortaleza_desierto_fire": "assets/fortalezas/fortaleza_desierto_fire.png",
        "fortaleza_barbaros_normal": "assets/fortalezas/fortaleza_barbaros_normal.png",
        "fortaleza_barbaros_fire": "assets/fortalezas/fortaleza_barbaros_fire.png",
    }
}

# ---------- Cargar templates dinámicamente ----------
def load_templates():
    global templates, offer_templates

    templates = {}
    offer_templates = {}

    # ---------- CAMPAMENTOS POR FACCION ----------
    if FACTION == "nomadas":
        camp_paths = {
            "normal": ["assets/eventos/nomadas/camp_normal.png"],
            "fire":   ["assets/eventos/nomadas/camp_fire.png"],
        }

    elif FACTION == "samurais":
        camp_paths = {
            "normal": ["assets/eventos/samurais/camp_normal.png"],
            "fire":   ["assets/eventos/samurais/camp_fire.png"],
        }

    elif FACTION == "berimond":
        camp_paths = {
            "normal": ["assets/berimond/camp_normal.png"],
            "fire":   ["assets/berimond/camp_fire.png"],
        }

    elif FACTION == "islas":
        camp_paths = {
            "normal": ["assets/islas/camp_normal.png"],
            "fire":   ["assets/islas/camp_fire.png"],
        }

    elif FACTION == "fortalezas":
        camp_paths = {
            "fortaleza_dragon_normal": ["assets/fortalezas/fortaleza_dragon_normal.png"],
            "fortaleza_dragon_fire":   ["assets/fortalezas/fortaleza_dragon_fire.png"],

            "fortaleza_desierto_normal": ["assets/fortalezas/fortaleza_desierto_normal.png"],
            "fortaleza_desierto_fire":   ["assets/fortalezas/fortaleza_desierto_fire.png"],

            "fortaleza_barbaros_normal": ["assets/fortalezas/fortaleza_barbaros_normal.png"],
            "fortaleza_barbaros_fire":   ["assets/fortalezas/fortaleza_barbaros_fire.png"],
        }

    else:
        camp_paths = {}

    # Cargar camp templates
    for label, paths in camp_paths.items():
        arr = []
        for p in paths:
            img = safe_imread(p)
            if img is not None:
                arr.append(img)
        templates[label] = arr

    # ---------- OFFERS / POPUPS (COMUNES A TODAS LAS FACCIONES) ----------
    offer_paths = {
        "reward": "assets/popups/reward.png",
        "offer": "assets/popups/offer.png",
        "offer2": "assets/popups/offer2.png",
        "offer3": "assets/popups/offer3.png",
        "offer4": "assets/popups/offer4.png",
    }

    for key, path in offer_paths.items():
        offer_templates[key] = safe_imread(path)


# ---------- Ventana de configuración de imágenes ----------
def open_image_config_window():
    global TEMPLATE_PATHS_BY_FACTION

    window = tk.Toplevel()
    window.title("Subir imágenes para la facción")

    frame = ttk.Frame(window, padding=10)
    frame.grid(row=0, column=0)

    row = 0
    for key, path in TEMPLATE_PATHS_BY_FACTION[FACTION].items():
        ttk.Label(frame, text=key).grid(row=row, column=0, sticky="w", padx=5, pady=5)

        def upload_file(k=key, p=path):
            file_path = filedialog.askopenfilename(
                title=f"Seleccionar imagen para {k}",
                filetypes=[("PNG/JPG", "*.png *.jpg *.jpeg")]
            )
            if file_path:
                try:
                    # Sobrescribir la imagen en la ruta fija
                    shutil.copyfile(file_path, p)
                    log(f"📌 Imagen '{k}' guardada en {p}")
                    load_templates()  # recargar templates en memoria
                except Exception as e:
                    log(f"⚠️ No se pudo guardar '{k}': {e}")

        ttk.Button(frame, text="Subir imagen", command=upload_file).grid(row=row, column=1, padx=5, pady=5)
        row += 1

    ttk.Button(frame, text="Cerrar", command=window.destroy).grid(row=row, column=0, columnspan=2, pady=10)

def actualizar_combo_comandante(*args):
    try:
        n = int(spin_comandantes.get())
    except Exception:
        n = 1
    combo_comandante["values"] = [f"Comandante {i+1}" for i in range(n)]
    if n > 0:
        combo_comandante.current(0)
        actualizar_entry_delay()


def actualizar_entry_delay(*args):
    idx = combo_comandante.current()
    if idx < 0:
        entry_delay.delete(0, tk.END)
        return
    delay = comandante_delays.get(idx, 0.0)
    entry_delay.delete(0, tk.END)
    entry_delay.insert(0, str(delay))
    
def save_config():
    try:
        config = {
            "last_camp_index": combo_camps.current(),
            "N_COMANDANTES": int(spin_comandantes.get()),
            "TIMER": int(spin_timer.get()),
            "comandante_delays": {str(k): v for k, v in comandante_delays.items()}
        }
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4)
        print(f"💾 Configuración guardada: {config}")
        load_config()
    except Exception as e:
        print(f"⚠️ Error guardando configuración: {e}")


def load_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                config = json.load(f)
        except Exception as e:
            log(f"⚠️ No se pudo cargar {CONFIG_FILE}: {e}")
            return

        # Restaura valores de coms y delays
        spin_comandantes.delete(0, "end")
        spin_comandantes.insert(0, config.get("N_COMANDANTES", 1))
        spin_timer.delete(0, "end")
        spin_timer.insert(0, config.get("TIMER", 10))

        delays = config.get("comandante_delays", {})
        for k, v in delays.items():
            try:
                comandante_delays[int(k)] = float(v)
            except Exception:
                pass

        actualizar_combo_comandante()
        actualizar_entry_delay()

        last_idx = config.get("last_camp_index", -1)
        if 0 <= last_idx < len(combo_camps["values"]):
            combo_camps.current(last_idx)

        log("💾 Configuración cargada")

# ---------- Vision ----------
def grab_screen(roi=None):
    screenshot = pyautogui.screenshot()
    img = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    if roi:
        x, y, w, h = roi
        img = img[y:y+h, x:x+w]
    return img

def match_multi_scale(img_gray, templ, label, scales, threshold):
    detections = []
    if templ is None:
        return detections
    h_t, w_t = templ.shape
    for s in scales:
        new_w = int(w_t * s)
        new_h = int(h_t * s)
        if new_w < 8 or new_h < 8:
            continue
        templ_rs = cv2.resize(templ, (new_w, new_h), interpolation=cv2.INTER_AREA)
        try:
            res = cv2.matchTemplate(img_gray, templ_rs, cv2.TM_CCOEFF_NORMED)
        except Exception as e:
            print("Error en matchTemplate:", e)
            continue
        loc = np.where(res >= threshold)
        for (y, x) in zip(*loc):
            score = float(res[y, x])
            detections.append(Detection(x, y, new_w, new_h, score, label))
    return detections

def iou(a: Detection, b: Detection) -> float:
    ax1, ay1, ax2, ay2 = a.x, a.y, a.x + a.w, a.y + a.h
    bx1, by1, bx2, by2 = b.x, b.y, b.x + b.w, b.y + b.h
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_a = a.w * a.h
    area_b = b.w * b.h
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0

def nms(dets: List[Detection], thr: float) -> List[Detection]:
    dets = sorted(dets, key=lambda d: d.score, reverse=True)
    keep = []
    while dets:
        best = dets.pop(0)
        keep.append(best)
        dets = [d for d in dets if iou(best, d) < thr]
    return keep

def resolve_conflicts(dets: List[Detection]) -> List[Detection]:
    dets = sorted(dets, key=lambda d: d.score, reverse=True)
    resolved = []
    for d in dets:
        if all(iou(d, r) < 0.5 for r in resolved):
            resolved.append(d)
    return resolved

def draw_and_save(img, dets, path):
    out = img.copy()
    for d in dets:
        color = (0, 255, 0) if d.label == "normal" else (0, 0, 255)
        cv2.rectangle(out, (d.x, d.y), (d.x + d.w, d.y + d.h), color, 2)
        cv2.putText(out, f"{d.label} {d.score:.2f}", (d.x, d.y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    cv2.imwrite(path, out)

def detect_camps():
    screen = grab_screen(ROI)
    gray = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
    dets_all = []
    for label, tmpls in templates.items():
        for templ in tmpls:
            dets_all.extend(match_multi_scale(gray, templ, label, SCALES, THRESHOLD))

    if ROI:
        ox, oy = ROI[0], ROI[1]
        for d in dets_all:
            d.x += ox
            d.y += oy

    dets_normal = nms([d for d in dets_all if d.label == "normal"], IOU_NMS)
    dets_fire   = nms([d for d in dets_all if d.label == "fire"], IOU_NMS)
    dets = resolve_conflicts(dets_normal + dets_fire)
    draw_and_save(screen if not ROI else grab_screen(ROI), dets, SAVE_DEBUG)
    return dets

def detect_popup(template_path: str, confidence=0.9, timeout=1.0) -> bool:

    start_time = time.time()
    while time.time() - start_time < timeout:
        screenshot = pyautogui.screenshot()
        screen_gray = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2GRAY)
        template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
        if template is None:
            return False

        res = cv2.matchTemplate(screen_gray, template, cv2.TM_CCOEFF_NORMED)
        loc = np.where(res >= confidence)
        if loc[0].size > 0:
            return True
        time.sleep(0.05)
    return False

# ---------- Automatización ----------
def calcular_max_ciclos(n_30, n_1h, n_comandantes, bonus_activo):
    if n_comandantes == 0:
        return 0

    if bonus_activo:
        ciclos_con_30 = n_30 // (2 * n_comandantes)
        ciclos_con_1h = n_1h // (1 * n_comandantes)
        return ciclos_con_30 + ciclos_con_1h
    else:
        return min(n_30, n_1h) // n_comandantes


def log(msg):
    global text_log
    try:
        if text_log:
            text_log.insert(tk.END, msg + "\n")
            text_log.see(tk.END)
        print(msg)
    except Exception:
        print(msg)

def click_image(image_path, confidence=0.85, timeout=4, offset_x=0):
    start = time.time()
    while time.time() - start < timeout:
        try:
            loc = pyautogui.locateCenterOnScreen(image_path, confidence=confidence)
        except Exception:
            loc = None
        if loc:
            pyautogui.moveTo(loc.x + offset_x, loc.y, duration=0.2)
            pyautogui.click()
            return True
        time.sleep(0.1)
    return False

def wait_and_click(image_path, confidence=0.85, timeout=3, offset_x=0, offset_y=0):
    start = time.time()
    templ = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if templ is None:
        log(f"⚠️ Template no encontrado: {image_path}")
        return False

    while time.time() - start < timeout:
        # ⏸️ Esperar si hay popup activo
        while POPUP_ACTIVE:
            log("⚠️ Popup activo, esperando a que se cierre...")
            time.sleep(0.3)

        screen = pyautogui.screenshot()
        screen_gray = cv2.cvtColor(np.array(screen), cv2.COLOR_RGB2GRAY)
        res = cv2.matchTemplate(screen_gray, templ, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)

        if max_val >= confidence:
            h, w = templ.shape
            cx, cy = max_loc[0] + w//2 + offset_x, max_loc[1] + h//2 + offset_y
            pyautogui.moveTo(cx, cy, duration=0.08)
            pyautogui.click()
            return True

        time.sleep(0.08)
    return False


def safe_attack_click():
    ok = wait_and_click("assets/attack_icon.png", confidence=0.8, timeout=0.25)
    if not ok:
        log("⚠️ safe_attack_click: no encontró attack_icon.png")
    return ok

# -------------------- Locks de cooldown --------------------
COOLDOWN_LOCKS = {}

def is_cooldown_locked(camp_id):
    return COOLDOWN_LOCKS.get(camp_id, False)

def start_cooldown(camp_id):
    COOLDOWN_LOCKS[camp_id] = True

def end_cooldown(camp_id):
    COOLDOWN_LOCKS[camp_id] = False

def click_coord_center(name, offset_x=0, offset_y=0):
    if name not in COORDS:
        log(f"⚠️ Coordenadas no encontradas para {name}")
        return False

    coord = COORDS[name]

    # Cálculo del centro real con float para evitar errores por división entera
    cx = coord['x'] + coord['w'] / 2 + offset_x
    cy = coord['y'] + coord['h'] / 2 + offset_y

    pyautogui.moveTo(cx, cy, duration=0.1)
    pyautogui.click()
    log(f"🖱 Click en {name} -> {{'x': {cx}, 'y': {cy}, 'w': {coord['w']}, 'h': {coord['h']}}}")
    return True


# -------------------- Cooldown camp --------------------
def cool_down_camp(camp: Detection, comandante: int):
    camp_id = id(camp)
    if is_cooldown_locked(camp_id):
        log(f"⏳ Campamento ({camp.x},{camp.y}) ya en cooldown, ignorando")
        return False

    start_cooldown(camp_id)
    log(f"🔒 Lock activado para campamento ({camp.x},{camp.y})")

    try:
        # Esperar que no haya popups activos al iniciar
        while POPUP_ACTIVE:
            log("⏳ Esperando que el popup se cierre antes de enfriar el campamento")
            time.sleep(0.5)

        log(f"👉 Enfriando campamento en llamas ({camp.x},{camp.y})")

        # Click en el campamento
        cx, cy = camp.x + camp.w // 2, camp.y + camp.h // 2
        pyautogui.moveTo(cx, cy, duration=0.2)
        pyautogui.click()
        time.sleep(0.6)

        # Click en attack_icon si existe
        if "attack_icon" in COORDS:
            click_coord_center("attack_icon")
            time.sleep(0.4)

        # Leer relojes disponibles
        try:
            relojes_30 = int(spin_30min.get())
        except:
            relojes_30 = 0
        try:
            relojes_1h = int(spin_1h.get())
        except:
            relojes_1h = 0
        log(f"📦 Relojes disponibles: {relojes_30} de 30m, {relojes_1h} de 1h")
        time.sleep(0.2)

        # Configurar pasos según BONUS
        steps = []
        if BONUS_ACTIVO:
            if relojes_30 >= 2:
                steps = [
                    ("reduce_icon", {"sleep":0.5}),
                    ("clock_30m_bonus", {"offset_x":100, "sleep":0.3}),
                    ("clock_30m_bonus", {"offset_x":100, "sleep":0.3}),
                    ("exit", {"template": True, "sleep":0.3}),
                ]
            elif relojes_1h >= 1:
                steps = [
                    ("reduce_icon", {"sleep":0.5}),
                    ("clock_1h_bonus", {"offset_x":100, "sleep":0.3}),
                    ("exit", {"template": True, "sleep":0.3}),
                ]
            else:
                log("❌ Sin relojes suficientes")
                return False
        else:
            if relojes_30 >= 1 and relojes_1h >= 1:
                steps = [
                    ("reduce_icon", {"sleep":0.5}),
                    ("clock_1h", {"offset_x":100, "sleep":0.3}),
                    ("arrow_left", {"sleep":0.3}),
                    ("clock_30m", {"offset_x":100, "sleep":0.3}),
                    ("exit", {"template": True, "sleep":0.3}),
                ]
            else:
                log("❌ Sin relojes suficientes")
                return False

        usados_30 = 0
        usados_1h = 0

        popup_files = [
            "assets/popups/reward.png",
            "assets/popups/offer.png",
            "assets/popups/offer2.png",
            "assets/popups/offer3.png",
        ]

        # Ejecutar pasos
        for name, opts in steps:
            for attempt in range(3):
                while POPUP_ACTIVE:
                    log("⚠️ Popup detectado durante la secuencia, pausando...")
                    time.sleep(0.3)

                log(f"🔍 Click en {name} (intento {attempt+1})...")

                if opts.get("template"):  # exit con wait_and_click
                    success = wait_and_click(f"assets/cooldown/{name}.png", confidence=0.75, timeout=3)
                    time.sleep(opts.get("sleep",0.3))

                    if success and name == "exit":
                        # Doble validación de popup tras click en exit
                        for retry in range(5):
                            popup_closed = False
                            for pp in popup_files:
                                if detect_popup(pp, confidence=0.8, timeout=0.2):
                                    log("⚠️ Popup apareció tras clicar exit, cerrando...")
                                    wait_and_click(pp, confidence=0.8, timeout=1)
                                    popup_closed = True
                                    time.sleep(0.3)
                                    break
                            if popup_closed:
                                log("🔁 Reintentando exit...")
                                wait_and_click(f"assets/cooldown/{name}.png", confidence=0.75, timeout=2)
                                time.sleep(0.3)
                            else:
                                if not detect_on_screen(name):
                                    break
                else:
                    success = click_coord_center(name, offset_x=opts.get("offset_x",0))
                    time.sleep(opts.get("sleep",0.3))

                if success:
                    if "clock_30m" in name:
                        usados_30 += 1
                    elif "clock_1h" in name:
                        usados_1h += 1

                    log(f"✅ Click en {name}")
                    break
                else:
                    log(f"⚠️ No se pudo clicar {name} en el intento {attempt+1}")
                    time.sleep(0.4)
            else:
                log(f"❌ Falló la secuencia en {name}, abortando")
                return False

        # Actualizar contadores
        relojes_30 = max(0, relojes_30 - usados_30)
        relojes_1h = max(0, relojes_1h - usados_1h)
        spin_30min.delete(0, "end")
        spin_30min.insert(0, str(relojes_30))
        spin_1h.delete(0, "end")
        spin_1h.insert(0, str(relojes_1h))

        log(f"✅ Se gastaron {usados_30} relojes de 30m y {usados_1h} de 1h. "
            f"Restan {relojes_30} de 30m y {relojes_1h} de 1h")

        log(f"🔓 Lock liberado para campamento ({camp.x},{camp.y})")
        return True

    finally:
        end_cooldown(camp_id)



def attack_camp(camp: Detection):
    cx, cy = camp.x + camp.w // 2, camp.y + camp.h // 2
    pyautogui.moveTo(cx, cy, duration=0.08)
    pyautogui.click()
    time.sleep(0.25)

    if "attack_icon" in COORDS:
        click_coord(COORDS["attack_icon"])
    time.sleep(0.5)
    
    if "confirm_attack" in COORDS:
        click_coord(COORDS["confirm_attack"])
    time.sleep(0.5)
    
    if "template_button" in COORDS:
        click_coord(COORDS["template_button"])

    if not wait_and_click("assets/attack/attack_button.png"):
        return False
    time.sleep(0.4)

    if detect_popup("assets/attack/min_troops.png", confidence=0.8, timeout=0.5):
        log("❌ Error: No hay suficientes tropas para atacar")
        wait_and_click("assets/attack/error_close.png")
        wait_and_click("assets/attack/exit2.png")
        global RUNNING
        RUNNING = False
        return False

    if horse_choice == "monedas" and "horse_gold_coins" in COORDS:
        click_coord(COORDS["horse_gold_coins"])
    elif horse_choice == "plumas" and "horse_premium" in COORDS:
        click_coord(COORDS["horse_premium"])
    else:
        log("⚠️ No se encontró coordenada para el caballo elegido")
        return False
    time.sleep(0.1)

    # Confirmar ataque
    if "confirm_attack2" in COORDS:
        click_coord(COORDS["confirm_attack2"])
        time.sleep(0.01)

    log(f"✅ Ataque completado con caballo {horse_choice}")

    return True

def detect_fire_roi(camp: Detection, confirm_frames: int = 3) -> bool:
    x, y, w, h = camp.x, camp.y, camp.w, camp.h
    margin = 6
    rx = max(0, x - margin)
    ry = max(0, y - margin)
    rw = max(1, w + margin * 2)
    rh = max(1, h + margin * 2)

    fire_count = 0
    for _ in range(confirm_frames):
        try:
            screenshot = pyautogui.screenshot(region=(rx, ry, rw, rh))
        except Exception:
            screenshot = pyautogui.screenshot()

        gray = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2GRAY)

        for templ in templates.get("fire", []):
            if templ is None:
                continue
            res = cv2.matchTemplate(gray, templ, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(res)
            if max_val >= THRESHOLD:
                fire_count += 1
                break  # ya detectamos fuego en este frame
        
        time.sleep(0.05)  # pequeña pausa entre frames

    return fire_count >= (confirm_frames // 2 + 1)  # mayoría de frames

# -------------------- Berimond --------------------

def wait_for_recruit_button(image_path, timeout=10):
    """Detecta el botón 'Reclutar' y devuelve su posición, sin hacer clic."""
    start = time.time()
    while time.time() - start < timeout:
        pos = detect_on_screen(image_path)
        if pos:
            return pos  # devuelve las coordenadas x, y
        time.sleep(0.2)
    return None


def recruit_troops(subtipo, slots=1):
    tipo_general = TROOP_TYPES.get(subtipo)
    imagen = TROOP_IMAGES.get(subtipo)
    if not tipo_general or not imagen:
        log(f"⚠️ Tropas desconocidas: {subtipo}")
        return

    # Aseguramos que slots esté entre 1 y 5
    slots = max(1, min(slots, 5))
    log(f"⚔️ Reclutando {subtipo} ({tipo_general}) con {slots} slot(s)...")

    try:
        # Abrir castillo y barracones
        wait_and_click("assets/recruit/castle_icon.png")
        time.sleep(0.2)
        wait_and_click("assets/recruit/recruit_icon.png")
        time.sleep(0.2)

        # Buscar y seleccionar tropa
        found = False
        for attempt in range(5):
            result = detect_on_screen(imagen)
            if result:
                wait_and_click(result.x, result.y)  # seleccionar tropa
                found = True
                break
            wait_and_click("assets/recruit/left_arrow.png")
            time.sleep(0.2)

        if not found:
            for attempt in range(5):
                result = detect_on_screen(imagen)
                if result:
                    wait_and_click(result.x, result.y)  # seleccionar tropa
                    found = True
                    break
                wait_and_click("assets/recruit/right_arrow.png")
                time.sleep(0.2)

        if not found:
            log(f"❌ No se pudo encontrar {subtipo}")
            return

        # Detectar botón "reclutar" una sola vez
        recruit_btn = detect_on_screen("assets/recruit/recruit.png")
        if not recruit_btn:
            log(f"❌ Botón Reclutar no encontrado")
            return

        # Mover mouse a la posición del botón y dar los clics
        pyautogui.moveTo(recruit_btn.x, recruit_btn.y)
        for i in range(slots):
            pyautogui.click()
            log(f"➡️ Reclutando {subtipo} (slot {i+1}/{slots})")
            time.sleep(1.5)  # espera para que la UI procese cada clic

        # Pasos finales
        for step in ["alliance_help", "exit_menu_barracks", "exit_castle"]:
            wait_and_click(f"assets/recruit/{step}.png")
            time.sleep(0.2)

        log(f"✅ Reclutamiento de {subtipo} completado ({slots} slot(s))")

    except Exception as e:
        log(f"❌ Error en reclutamiento: {e}")

def click_coord(coord: dict):
    x, y = coord["x"], coord["y"]
    pyautogui.moveTo(x, y, duration=0.05)
    pyautogui.click()


def attack_berimond():
    global horse_choice

    # Click en campamento (centro de pantalla)
    screen_width, screen_height = pyautogui.size()
    pyautogui.moveTo(screen_width//2, screen_height//2, duration=0.05)
    pyautogui.click()
    time.sleep(0.2)

    # Click en pre-attack steps (coordenadas fijas)
    for step in ["attack_icon","confirm_attack", "template_button", "attack_button"]:
        if step in COORDS:
            click_coord(COORDS[step])
            time.sleep(0.4)

    # Detectar min_troops
    if detect_popup("assets/attack/min_troops.png", confidence=0.8, timeout=0.5):
        log("❌ Error: No hay suficientes tropas para atacar")
        # Cerrar errores por imagen
        wait_and_click("assets/attack/error_close.png")
        wait_and_click("assets/attack/exit2.png")
        return False

    # --- Selección del caballo según GUI ---
    if horse_choice == "monedas" and "horse_monedas" in COORDS:
        click_coord(COORDS["horse_monedas"])
    elif horse_choice == "plumas" and "horse_plumas" in COORDS:
        click_coord(COORDS["horse_plumas"])
    else:
        log("⚠️ No se encontró coordenada para el caballo elegido")
        return False
    time.sleep(0.3)

    # Confirmar ataque
    if "confirm_attack2" in COORDS:
        click_coord(COORDS["confirm_attack2"])
        time.sleep(0.01)

    log(f"✅ Ataque completado con caballo {horse_choice}")
    return True

def launch_main_window_berimond():
    global root, spin_comandantes, spin_timer, combo_comandante
    global entry_delay, text_log, label_status, comandante_delays
    global ATTACK_ACTIVE

    if "comandante_delays" not in globals():
        comandante_delays = {}

    root = tk.Tk()
    root.title("Bot Berimond")
    
    # --- Barra de menú ---
    menu_bar = tk.Menu(root)
    root.config(menu=menu_bar)

    settings_menu = tk.Menu(menu_bar, tearoff=0)
    menu_bar.add_cascade(label="Settings", menu=settings_menu)
    settings_menu.add_command(label="Configurar coordenadas", command=coord_config_window)
    settings_menu.add_command(label="Configurar imágenes", command=open_image_config_window)
    settings_menu.add_command(label="Añadir imágenes recortadas de tropas", command=add_troop_images)
    
    # --- Frame principal ---
    frame = ttk.Frame(root, padding=10)
    frame.grid(row=0, column=0, sticky="nw")

    ttk.Label(frame, text="Comandantes:").grid(row=0, column=0, sticky="w")
    spin_comandantes = tk.Spinbox(frame, from_=1, to=40, width=5)
    spin_comandantes.grid(row=0, column=1, padx=5, pady=5)

    ttk.Label(frame, text="Cooldown (s):").grid(row=1, column=0, sticky="w")
    spin_timer = tk.Spinbox(frame, from_=5, to=600, width=5)
    spin_timer.grid(row=1, column=1, padx=5, pady=5)

    # --- Delay Frame ---
    delays_frame = ttk.LabelFrame(root, text="Delay por Comandante", padding=10)
    delays_frame.grid(row=0, column=1, padx=15, pady=10, sticky="n")

    ttk.Label(delays_frame, text="Selecciona comandante:").grid(row=0, column=0, sticky="w")
    combo_comandante = ttk.Combobox(delays_frame, state="readonly", width=18)
    combo_comandante.grid(row=0, column=1, padx=5, pady=5)

    ttk.Label(delays_frame, text="Delay (s):").grid(row=1, column=0, sticky="w")
    entry_delay = ttk.Entry(delays_frame, width=6)
    entry_delay.grid(row=1, column=1, padx=5, pady=5)

    # --- Log ---
    log_frame = ttk.Frame(root)
    log_frame.grid(row=1, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")
    text_log = tk.Text(log_frame, height=20, width=80, state="normal")
    text_log.grid(row=0, column=0, sticky="nsew")
    scrollbar = ttk.Scrollbar(log_frame, orient="vertical", command=text_log.yview)
    scrollbar.grid(row=0, column=1, sticky="ns")
    text_log.config(yscrollcommand=scrollbar.set)
    
    # --- Selección de caballo ---
    horse_frame = ttk.LabelFrame(root, text="Tipo de Caballo", padding=10)
    horse_frame.grid(row=2, column=0, padx=10, pady=10, sticky="w")

    horse_var = tk.StringVar(value="monedas")  # por defecto

    rb_monedas = ttk.Radiobutton(horse_frame, text="Caballo Monedas", variable=horse_var, value="monedas",
                             command=lambda: set_horse_choice(horse_var.get()))
    rb_monedas.grid(row=0, column=0, padx=5, pady=5)

    rb_plumas = ttk.Radiobutton(horse_frame, text="Caballo Plumas", variable=horse_var, value="plumas",
                            command=lambda: set_horse_choice(horse_var.get()))
    rb_plumas.grid(row=0, column=1, padx=5, pady=5)

    def set_horse_choice(choice):
        global horse_choice
        horse_choice = choice
        log(f"🐴 Seleccionado: Caballo {choice.capitalize()}")

    def log_local(msg):
        text_log.insert(tk.END, f"{msg}\n")
        text_log.see(tk.END)
        root.update_idletasks()
        print(msg)

    globals()['log'] = log_local

    # --- Label de estado ---
    label_status = ttk.Label(frame, text="", font=("Segoe UI", 12))
    label_status.grid(row=0, column=2, padx=5)

    # --- Funciones de actualización ---
    def actualizar_combo_comandante(*args):
        try:
            n = int(spin_comandantes.get())
        except:
            n = 1
        combo_comandante["values"] = [f"Comandante {i+1}" for i in range(n)]
        if n > 0:
            combo_comandante.current(0)
            actualizar_entry_delay()

    def actualizar_entry_delay(*args):
        idx = combo_comandante.current()
        if idx < 0:
            entry_delay.delete(0, tk.END)
            return
        delay = comandante_delays.get(idx, 0.0)
        entry_delay.delete(0, tk.END)
        entry_delay.insert(0, str(delay))

    def guardar_delay(*args):
        idx = combo_comandante.current()
        if idx < 0:
            return
        try:
            comandante_delays[idx] = float(entry_delay.get())
            log(f"⏱ Delay comandante {idx+1} actualizado a {comandante_delays[idx]}s")
        except:
            comandante_delays[idx] = 0.0
            log(f"⚠️ Delay comandante {idx+1} no válido, reiniciado a 0.0s")

    # --- Guardar y cargar config ---
    def guardar_config():
        guardar_delay()
        config = {}
        if os.path.exists(CONFIG_FILEB):
            try:
                with open(CONFIG_FILEB, "r", encoding="utf-8") as f:
                    config = json.load(f)
            except:
                pass
        config.update({
            "N_COMANDANTES": int(spin_comandantes.get()),
            "TIMER": int(spin_timer.get()),
            "comandante_delays": {str(k): v for k, v in comandante_delays.items()}
        })
        try:
            with open(CONFIG_FILEB, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=4)
            log(f"💾 Configuración guardada")
        except Exception as e:
            log(f"⚠️ Error guardando configuración: {e}")

    def cargar_config():
        if os.path.exists(CONFIG_FILEB):
            try:
                with open(CONFIG_FILEB, "r", encoding="utf-8") as f:
                    config = json.load(f)
            except Exception as e:
                log(f"⚠️ No se pudo cargar {CONFIG_FILEB}: {e}")
                return
            spin_comandantes.delete(0, "end")
            spin_comandantes.insert(0, config.get("N_COMANDANTES", 1))
            spin_timer.delete(0, "end")
            spin_timer.insert(0, config.get("TIMER", 10))
            delays = config.get("comandante_delays", {})
            for k, v in delays.items():
                try:
                    comandante_delays[int(k)] = float(v)
                except:
                    pass
            actualizar_combo_comandante()
            actualizar_entry_delay()
            log("💾 Configuración cargada")

    cargar_config()

    # --- Bindings ---
    spin_comandantes.config(command=actualizar_combo_comandante)  
    combo_comandante.bind("<<ComboboxSelected>>", actualizar_entry_delay)
    entry_delay.bind("<FocusOut>", guardar_delay)
    spin_timer.config(command=guardar_config)

    # --- Botones iniciar/detener ---
    def start_attacks():
        global ATTACK_ACTIVE
        if ATTACK_ACTIVE:
            return
        ATTACK_ACTIVE = True
        threading.Thread(target=attack_loop, daemon=True).start()
        log("▶️ Ataques iniciados")

    def stop_attacks():
        global ATTACK_ACTIVE
        ATTACK_ACTIVE = False
        log("⏹ Ataques detenidos")

    def attack_loop():
        while ATTACK_ACTIVE:
            attack_berimond()

    ttk.Button(frame, text="Iniciar", command=start_attacks).grid(row=2, column=0, pady=5)
    ttk.Button(frame, text="Detener", command=stop_attacks).grid(row=2, column=1, pady=5)
    
    # --- Frame para reclutamiento ---
    recruit_frame = ttk.LabelFrame(root, text="Reclutamiento de tropas", padding=10)
    recruit_frame.grid(row=3, column=0, padx=10, pady=10, sticky="w")

    current_type = tk.StringVar(value="cuerpo")
    selected_troop = tk.StringVar()
    combo_troop = ttk.Combobox(recruit_frame, state="readonly", textvariable=selected_troop, width=30)
    combo_troop.grid(row=0, column=1, padx=5, pady=5)
    image_label = tk.Label(recruit_frame)
    image_label.grid(row=1, column=1, pady=10)

    def update_troop_list(tipo):
        current_type.set(tipo)
        troops = [t for t, ttype in TROOP_TYPES.items() if ttype == tipo]
        combo_troop["values"] = troops
        if troops:
            combo_troop.current(0)
            update_troop_image(troops[0])

    def update_troop_image(troop_name):
        path = TROOP_IMAGES_ORIGINAL.get(troop_name)
        if path:
            try:
                img = Image.open(path)
                h = 100
                w = int(img.width * (h / img.height))
                img = img.resize((w, h), Image.LANCZOS)
                img_tk = ImageTk.PhotoImage(img)
                image_label.configure(image=img_tk)
                image_label.image = img_tk
            except Exception as e:
                log(f"⚠️ Error cargando imagen de {troop_name}: {e}")

    combo_troop.bind("<<ComboboxSelected>>", lambda e: update_troop_image(selected_troop.get()))
    ttk.Button(recruit_frame, text="⚔️ Melee", command=lambda: update_troop_list("cuerpo")).grid(row=0, column=0, padx=5)
    ttk.Button(recruit_frame, text="🏹 Ranged", command=lambda: update_troop_list("distancia")).grid(row=0, column=2, padx=5)
    update_troop_list("cuerpo")

    # Spinbox para seleccionar slots
    slots_var = tk.IntVar(value=1)
    ttk.Label(recruit_frame, text="Slots:").grid(row=2, column=0, sticky="e")
    spin_slots = tk.Spinbox(recruit_frame, from_=1, to=5, width=5, textvariable=slots_var)
    spin_slots.grid(row=2, column=1, sticky="w", padx=5)

    # Modificar la función de reclutamiento para usar slots
    def recruit_selected():
        tropa = selected_troop.get()
        slots = slots_var.get()  # número de slots seleccionado
        if not tropa:
            log("⚠️ No se seleccionó ninguna tropa")
            return
        log(f"➡️ Reclutando {tropa} ({current_type.get()}) con {slots} slot(s)...")
        recruit_troops(tropa, slots)  # pasar slots a la función
    ttk.Button(recruit_frame, text="Reclutar", command=recruit_selected).grid(row=2, column=0, columnspan=3, pady=10)
    # --- Guardar al cerrar ---
    def on_close():
        guardar_config()
        global ATTACK_ACTIVE
        ATTACK_ACTIVE = False
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()

# -------------------- UI: main window --------------------
global ultimo_log_ciclos
ultimo_log_ciclos = None
def launch_main_window():
    global root, combo_camps, spin_comandantes, spin_timer, combo_comandante
    global entry_delay, text_log, label_status, comandante_delays

    # Cargar templates
    load_templates()

    if "comandante_delays" not in globals():
        comandante_delays = {}

    root = tk.Tk()
    root.title(f"Bot de Campamentos - {FACTION.capitalize() if FACTION else '??'}")

    # Layout
    frame = ttk.Frame(root, padding=10)
    frame.grid(row=0, column=0, sticky="nw")

    combo_camps = ttk.Combobox(frame, state="readonly")
    combo_camps.grid(row=0, column=1, padx=5, pady=5)

    spin_comandantes = tk.Spinbox(frame, from_=1, to=15, width=5)
    spin_comandantes.grid(row=1, column=1, padx=5, pady=5)

    spin_timer = tk.Spinbox(frame, from_=5, to=600, width=5)
    spin_timer.grid(row=2, column=1, padx=5, pady=5)
    
    # Spinboxes para relojes
    ttk.Label(frame, text="Relojes 30min:").grid(row=3, column=0, sticky="w")
    global spin_30min
    spin_30min = tk.Spinbox(frame, from_=0, to=9999, width=5)
    spin_30min.grid(row=3, column=1, padx=5, pady=5)

    ttk.Label(frame, text="Relojes 1h:").grid(row=4, column=0, sticky="w")
    global spin_1h
    spin_1h = tk.Spinbox(frame, from_=0, to=9999, width=5)
    spin_1h.grid(row=4, column=1, padx=5, pady=5)

    # Spinbox para ciclos a ejecutar
    ttk.Label(frame, text="Ciclos a ejecutar:").grid(row=5, column=0, sticky="w")
    spin_ciclos = tk.Spinbox(frame, from_=1, to=1, width=5) 
    spin_ciclos.grid(row=5, column=1, padx=5, pady=5)


    ttks = [
        ("Campamento", combo_camps),
        ("Comandantes", spin_comandantes),
        ("Cooldown (s)", spin_timer),
    ]
    for i, (lbl, widget) in enumerate(ttks):
        ttk.Label(frame, text=lbl).grid(row=i, column=0, sticky="w")

    # Delay Frame
    delays_frame = ttk.LabelFrame(root, text="Delay por Comandante", padding=10)
    delays_frame.grid(row=0, column=1, padx=15, pady=10, sticky="n")

    ttk.Label(delays_frame, text="Selecciona comandante:").grid(row=0, column=0, sticky="w")
    combo_comandante = ttk.Combobox(delays_frame, state="readonly", width=18)
    combo_comandante.grid(row=0, column=1, padx=5, pady=5)

    ttk.Label(delays_frame, text="Delay (s):").grid(row=1, column=0, sticky="w")
    entry_delay = ttk.Entry(delays_frame, width=6)
    entry_delay.grid(row=1, column=1, padx=5, pady=5)
    # --- Selección de caballo ---
    horse_frame = ttk.LabelFrame(root, text="Tipo de Caballo", padding=10)
    horse_frame.grid(row=2, column=0, padx=10, pady=10, sticky="w")

    horse_var = tk.StringVar(value="monedas")

    rb_monedas = ttk.Radiobutton(horse_frame, text="Caballo Monedas", variable=horse_var, value="monedas",
                             command=lambda: set_horse_choice(horse_var.get()))
    rb_monedas.grid(row=0, column=0, padx=5, pady=5)

    rb_plumas = ttk.Radiobutton(horse_frame, text="Caballo Plumas", variable=horse_var, value="plumas",
                            command=lambda: set_horse_choice(horse_var.get()))
    rb_plumas.grid(row=0, column=1, padx=5, pady=5)

    def set_horse_choice(choice):
        global horse_choice
        horse_choice = choice
        log(f"🐴 Seleccionado: Caballo {choice.capitalize()}")
    # --- Barra de menú ---
    menu_bar = tk.Menu(root)
    root.config(menu=menu_bar)
    # Menu
    settings_menu = tk.Menu(menu_bar, tearoff=0)
    menu_bar.add_cascade(label="Settings", menu=settings_menu)
    settings_menu.add_command(label="Configurar coordenadas", command=coord_config_window)
    settings_menu.add_command(label="Configurar imágenes", command=open_image_config_window)
    
    def guardar_delay(*args):
        idx = combo_comandante.current()
        if idx < 0:
            return
        try:
            comandante_delays[idx] = float(entry_delay.get())
        except:
            comandante_delays[idx] = 0.0
        save_config()
        load_config()

    globals()['actualizar_combo_comandante'] = actualizar_combo_comandante
    globals()['actualizar_entry_delay'] = actualizar_entry_delay

    # --- Bindings para widgets ---
    spin_comandantes.config(command=actualizar_combo_comandante)
    combo_comandante.bind("<<ComboboxSelected>>", actualizar_entry_delay)
    entry_delay.bind("<FocusOut>", lambda e: guardar_delay())

    actualizar_combo_comandante()
    # Log frame
    log_frame = ttk.Frame(root)
    log_frame.grid(row=1, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")

    text_log = tk.Text(log_frame, height=20, width=80, state="normal")
    text_log.grid(row=0, column=0, sticky="nsew")

    scrollbar = ttk.Scrollbar(log_frame, orient="vertical", command=text_log.yview)
    scrollbar.grid(row=0, column=1, sticky="ns")
    text_log.config(yscrollcommand=scrollbar.set)

    def log_local(msg):
        text_log.insert(tk.END, f"{msg}\n")
        text_log.see(tk.END)
        root.update_idletasks()
        print(msg)
        
    globals()['log'] = log_local

    label_status = ttk.Label(frame, text="", font=("Segoe UI", 12))
    label_status.grid(row=0, column=2, padx=5)

    def update_status_indicator_real_time():
        idx = combo_camps.current()
        if idx < 0 or idx >= len(camps_detected):
            label_status.config(text="")
            return
        camp = camps_detected[idx]
        label_status.config(text="🔥" if camp.label == "fire" else "✅")

    combo_camps.bind("<<ComboboxSelected>>", lambda e: update_status_indicator_real_time())
    
    def actualizar_max_ciclos(*args):
        global ultimo_log_ciclos, N_COMANDANTES

        try:
            N_COMANDANTES = int(spin_comandantes.get())
            if N_COMANDANTES < 1:
                N_COMANDANTES = 1
        except:
            N_COMANDANTES = 1

        # Cantidad de relojes
        try:
            n_30 = int(spin_30min.get())
        except:
            n_30 = 0
        try:
            n_1h = int(spin_1h.get())
        except:
            n_1h = 0

        max_ciclos = calcular_max_ciclos(n_30, n_1h, N_COMANDANTES, BONUS_ACTIVO)

        if max_ciclos < 0:
            max_ciclos = 0

        spin_ciclos.config(to=max_ciclos)

        try:
            current_val = int(spin_ciclos.get())
            if current_val > max_ciclos:
                spin_ciclos.delete(0, "end")
                spin_ciclos.insert(0, str(max_ciclos))
        except:
            spin_ciclos.delete(0, "end")
            spin_ciclos.insert(0, str(max_ciclos))

        if ultimo_log_ciclos != max_ciclos:
            log(f"ℹ️ Puedes ejecutar como máximo {max_ciclos} ciclos con los relojes y comandantes actuales")
            ultimo_log_ciclos = max_ciclos

    spin_30min.config(command=actualizar_max_ciclos)
    spin_1h.config(command=actualizar_max_ciclos)
    spin_comandantes.config(command=actualizar_max_ciclos)

    bonus_var = tk.BooleanVar(value=False)
    chk_bonus = ttk.Checkbutton(frame, text="Bonus de alianza activo", variable=bonus_var,command=lambda: set_bonus(bonus_var.get()))
    chk_bonus.grid(row=6, column=0, columnspan=2, pady=5)

    def set_bonus(value):
        global BONUS_ACTIVO
        BONUS_ACTIVO = value
        actualizar_max_ciclos()
    spin_comandantes.config(command=save_config)
    spin_timer.config(command=save_config)
    combo_camps.bind("<<ComboboxSelected>>", lambda e: save_config())
   
    # ---------- UI commands ----------
    def refresh_camps():
        global camps_detected
        camps_detected = detect_camps()
        combo_camps["values"] = [f"{c.label} ({c.x},{c.y})" for c in camps_detected]
        log(f"🔍 Detectados {len(camps_detected)} campamentos")
        update_status_indicator_real_time()

    def start_cycle():
        global RUNNING, N_COMANDANTES, TIMER
        if RUNNING:
            messagebox.showwarning("Bot", "Ya está en ejecución")
            return

        idx = combo_camps.current()
        if idx < 0:
            messagebox.showerror("Error", "Selecciona un campamento")
            return

        for i in range(int(spin_comandantes.get())):
            comandante_delays.setdefault(i, comandante_delays.get(i, 0.5))

        refresh_camps()
        selected = camps_detected[idx] if idx < len(camps_detected) else None
        if selected is None:
            messagebox.showerror("Error", "No se pudo obtener las coordenadas del campamento seleccionado")
            return

        N_COMANDANTES = int(spin_comandantes.get())
        TIMER = int(spin_timer.get())

        try:
            n_30 = int(spin_30min.get())
        except:
            n_30 = 0
        try:
            n_1h = int(spin_1h.get())
        except:
            n_1h = 0

        ciclos_max = calcular_max_ciclos(n_30, n_1h, N_COMANDANTES, BONUS_ACTIVO)

        if ciclos_max <= 0:
            messagebox.showerror(
                "Error",
                f"No hay relojes suficientes para {N_COMANDANTES} comandantes.\n"
                f"Tienes {n_30} relojes 30 min y {n_1h} relojes 1 h."
            )
            log("❌ Intento de iniciar ciclo sin suficientes relojes")
            return

        log(f"ℹ️ Puedes ejecutar como máximo {ciclos_max} ciclos con {N_COMANDANTES} comandantes y tus relojes")

        try:
            ciclos = int(spin_ciclos.get())
        except:
            ciclos = ciclos_max

        if ciclos > ciclos_max:
            log(f"⚠️ Ajustando ciclos a máximo disponible: {ciclos_max}")
            ciclos = ciclos_max
            spin_ciclos.delete(0, "end")
            spin_ciclos.insert(0, str(ciclos))

        RUNNING = True
        threading.Thread(target=ciclo_ataques, args=(selected, ciclos), daemon=True).start()
        log(f"▶️ Ciclo de ataques iniciado ({ciclos} ciclos)")

    def stop_cycle():
        global RUNNING
        if RUNNING:
            RUNNING = False
            save_config()
            log("⏹ Ciclo detenido")
        else:
            log("ℹ️ No hay ciclo en ejecución")

    btns = [
        ("Detectar campamentos", refresh_camps),
        ("Iniciar", start_cycle),
        ("Detener", stop_cycle),
    ]
    for i, (lbl, cmd) in enumerate(btns):
        ttk.Button(frame, text=lbl, command=cmd).grid(row=10+i, column=0, columnspan=2, pady=5)

    # Cargar configuración
    load_config()

    # Activar los hilos
    threading.Thread(target=watcher_fire_fast, daemon=True).start()
    threading.Thread(target=watcher_popups_templates, daemon=True).start()

    # Safe close
    def on_close():
        global RUNNING, WATCHER_ACTIVE
        RUNNING = False
        WATCHER_ACTIVE = False
        log("🔴 Cerrando... esperando a que los hilos terminen")
        save_config()
        time.sleep(0.5)
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)

    root.mainloop()

# ---------- Watchers ----------
last_click_time = {}
COOLDOWN = 1  # segundo
last_popup_time = 0


def watcher_popups_templates():
    global last_click_time, POPUP_ACTIVE, last_popup_time
    while WATCHER_ACTIVE:
        screenshot = pyautogui.screenshot()
        screen_gray = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2GRAY)
        current_time = time.time()

        popup_found = False

        for name, template in offer_templates.items():
            if template is None:
                continue
            if current_time - last_click_time.get(name, 0) < COOLDOWN:
                continue

            res = cv2.matchTemplate(screen_gray, template, cv2.TM_CCOEFF_NORMED)
            loc = np.where(res >= 0.9)

            if loc[0].size > 0:
                popup_found = True
                POPUP_ACTIVE = True
                last_popup_time = current_time
                for y, x in zip(*loc):
                    h, w = template.shape
                    cx, cy = x + w // 2, y + h // 2
                    pyautogui.moveTo(cx, cy, duration=0.2)
                    pyautogui.click()
                    last_click_time[name] = current_time
                    log(f"✅ {name} clicado en ({x},{y})")
                    break
                break

        if not popup_found:
            if current_time - last_popup_time > POPUP_GRACE:
                POPUP_ACTIVE = False

        time.sleep(0.1)


def is_cooldown_locked(camp_id):
    return COOLDOWN_LOCKS.get(camp_id, False)

def start_cooldown(camp_id):
    COOLDOWN_LOCKS[camp_id] = True

def end_cooldown(camp_id):
    COOLDOWN_LOCKS[camp_id] = False

def watcher_fire_fast():

    global camps_detected, WATCHER_ACTIVE, POPUP_ACTIVE, last_popup_time

    if not camps_detected:
        return

    target_camp = camps_detected[0]
    comandante_actual = 0
    camp_id = id(target_camp)

    while WATCHER_ACTIVE:
        try:
            while POPUP_ACTIVE:
                time.sleep(0.1)

            if time.time() - last_popup_time < POPUP_GRACE:
                time.sleep(0.1)
                continue

            if is_cooldown_locked(camp_id):
                time.sleep(0.1)
                continue

            if detect_fire_roi(target_camp):
                if target_camp.label != "fire":
                    log(f"🔥 Campamento en llamas detectado en ({target_camp.x},{target_camp.y})")
                target_camp.label = "fire"

                if comandante_actual < N_COMANDANTES:
                    log(f"🔥 Enfriando con comandante {comandante_actual+1}/{N_COMANDANTES}")
                    if cool_down_camp(target_camp, comandante_actual):
                        comandante_actual += 1
                    else:
                        log(f"❌ Fallo al enfriar con comandante {comandante_actual+1}, reintentando...")
                        time.sleep(0.25)
                else:
                    comandante_actual = 0
            else:
                target_camp.label = "normal"
                comandante_actual = 0

        except Exception as e:
            log(f"⚠️ Error en watcher_fire_fast con campamento ({target_camp.x},{target_camp.y}): {e}")

        time.sleep(0.1)


# ---------- Attack cycle ----------
def ciclo_ataques(target: Detection, ciclos_max):

    global RUNNING
    ciclos_realizados = 0

    while RUNNING and ciclos_realizados < ciclos_max:
        if detect_fire_roi(target):
            log("🔥 Campamento en llamas al inicio, enfriando...")
            for comandante_idx in range(N_COMANDANTES):
                if not RUNNING:
                    break
                while detect_fire_roi(target) and RUNNING:
                    if cool_down_camp(target, comandante_idx):
                        log(f"✅ Campamento enfriado por comandante {comandante_idx+1}")
                        break
                    else:
                        log("❌ Fallo al enfriar, reintentando...")
                        time.sleep(0.5)

        log(f"⚔️ Lanzando {N_COMANDANTES} ataques al campamento en {target.x},{target.y}")
        for i in range(N_COMANDANTES):
            if not RUNNING:
                break
            if attack_camp(target):
                log(f"✅ Ataque {i+1}")
            else:
                log(f"❌ Fallo en ataque {i+1}")
            delay = comandante_delays.get(i, 0.5)
            time.sleep(delay)

        ciclos_realizados += 1
        log(f"🔁 Ciclo {ciclos_realizados}/{ciclos_max} completado")

        log(f"⏳ Iniciando cooldown de {TIMER}s")
        end_time = time.time() + TIMER
        last_log = int(TIMER)

        while RUNNING and time.time() < end_time:
            if detect_fire_roi(target):
                log("🔥 Campamento entró en llamas durante cooldown, enfriando...")
                for comandante_idx in range(N_COMANDANTES):
                    if not RUNNING:
                        break
                    while detect_fire_roi(target) and RUNNING:
                        if cool_down_camp(target, comandante_idx):
                            log(f"✅ Campamento enfriado en cooldown por comandante {comandante_idx+1}")
                            break
                        else:
                            log("❌ Fallo al enfriar en cooldown, reintentando...")
                            time.sleep(0.45)

            remaining = int(end_time - time.time())
            if remaining <= 0:
                break
            if remaining == TIMER or remaining <= last_log - 30:
                log(f"⏳ Siguiente ciclo en {remaining}s")
                last_log = remaining
            time.sleep(0.5)

    RUNNING = False
    log("⏹ Todos los ciclos completados o se detuvo la ejecución")


# -------------------- Seleccionador de Facciones --------------------
def choose_faction_window():
    def select_faction(name):
        global FACTION
        FACTION = name
        w.destroy()
        load_templates()
        if name in ["nomadas", "samurais"]:
            launch_main_window()  # GUI original
        elif name == "berimond":
            launch_main_window_berimond()  # Nueva GUI para Berimond
        else:
            messagebox.showinfo("Info", f"La facción/evento '{name}' aún no tiene GUI implementada")

    def open_settings():
        settings_win = tk.Toplevel(w)   # ✅ ventana secundaria
        settings_win.title("Settings")
        settings_win.geometry("300x200")

        ttk.Label(settings_win, text="Configuración avanzada", font=("Segoe UI", 12)).pack(pady=10)

        def launch_coords_app():
            try:
                import subprocess, sys
                subprocess.Popen([sys.executable, "coords_app.py"])
            except Exception as e:
                messagebox.showerror("Error", f"No se pudo abrir coords_app.py:\n{e}")

        ttk.Button(settings_win, text="Abrir Configuración de Coordenadas", command=launch_coords_app).pack(pady=20)

    # ✅ ventana raíz
    w = tk.Tk()
    w.title("Seleccionador de Eventos")

    ttk.Label(w, text="Selecciona el evento o NPC:", font=("Segoe UI", 14)).pack(pady=20)

    frame_buttons = ttk.Frame(w)
    frame_buttons.pack(pady=10)

    # Cargar iconos (64x64)
    icons = {
        "nomadas": ImageTk.PhotoImage(Image.open("assets/icons/nomadas.png").resize((64, 64))),
        "samurais": ImageTk.PhotoImage(Image.open("assets/icons/samurais.png").resize((64, 64))),
        "berimond": ImageTk.PhotoImage(Image.open("assets/icons/berimond.png").resize((64, 64))),
        "islas": ImageTk.PhotoImage(Image.open("assets/icons/islas.png").resize((64, 64))),
        "fortalezas": ImageTk.PhotoImage(Image.open("assets/icons/fortalezas.png").resize((64, 64))),
    }

    # Botones de facciones
    row, col = 0, 0
    for name, label in [
        ("nomadas", "Nómadas"),
        ("samurais", "Samuráis"),
        ("berimond", "Berimond"),
        ("islas", "Islas"),
        ("fortalezas", "Fortalezas")
    ]:
        btn = tk.Button(frame_buttons, text=label, image=icons[name],
                        compound="top", command=lambda n=name: select_faction(n))
        btn.grid(row=row, column=col, padx=15, pady=10)
        col += 1
        if col > 2:
            col = 0
            row += 1
    w.mainloop()

# ---------- Start ----------
if __name__ == "__main__":
    choose_faction_window()
