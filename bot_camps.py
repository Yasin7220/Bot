import time
import cv2
import numpy as np
import pyautogui
import threading
import tkinter as tk
from tkinter import ttk, messagebox
from dataclasses import dataclass
from typing import List, Optional
import json
import os

# ---------- Config ----------
FACTION: Optional[str] = None  # will be set by faction chooser

THRESHOLD = 0.8
SCALES = np.linspace(0.6, 1.4, 11)
IOU_NMS = 0.35
SAVE_DEBUG = "last_detection.png"
ROI: Optional[tuple] = None  
CONFIG_FILE = "config.json"
BONUS_ACTIVO = False  # True si el bonus está activo
POPUP_GRACE = 1.0
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

# GUI references
root = None
combo_camps = None
spin_comandantes = None
spin_timer = None
combo_comandante = None
entry_delay = None
text_log = None
label_status = None

# ---------- Utilities ----------
def safe_imread(path: str):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"⚠️ No se pudo leer la imagen: {path}")
    return img

def load_templates():
    global templates, offer_templates, attack_ui_templates

    templates = {}

    # Rutas por facción
    if FACTION == "nomadas":
        camp_paths = {
            "normal": ["assets/nomadas/camp_normal.png"],
            "fire":   ["assets/nomadas/camp_fire.png"],
        }
    elif FACTION == "samurais":
        camp_paths = {
            "normal": ["assets/samurais/camp_normal.png"],
            "fire":   ["assets/samurais/camp_fire.png"],
        }

    # Camp templates
    for label, paths in camp_paths.items():
        arr = []
        for p in paths:
            img = safe_imread(p)
            if img is not None:
                arr.append(img)
        templates[label] = arr

    # Offer templates
    offer_templates = {
        "reward": safe_imread("assets/reward.png"),
        "offer": safe_imread("assets/offer.png"),
        "offer2": safe_imread("assets/offer2.png"),
        "offer3": safe_imread("assets/offer3.png"),
        "offer4": safe_imread("assets/offer4.png"),
    }

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

def wait_and_click(image_path, confidence=0.85, timeout=3, offset_x=0):
    start = time.time()
    templ = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if templ is None:
        log(f"⚠️ Template no encontrado: {image_path}")
        return False
    while time.time() - start < timeout:
        screen = pyautogui.screenshot()
        screen_gray = cv2.cvtColor(np.array(screen), cv2.COLOR_RGB2GRAY)
        res = cv2.matchTemplate(screen_gray, templ, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        if max_val >= confidence:
            h, w = templ.shape
            cx, cy = max_loc[0] + w//2 + offset_x, max_loc[1] + h//2
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


# -------------------- Cooldown camp --------------------
def cool_down_camp(camp: Detection, comandante: int):
    camp_id = id(camp)
    if is_cooldown_locked(camp_id):
        log(f"⏳ Campamento ({camp.x},{camp.y}) ya en cooldown, ignorando")
        return False

    start_cooldown(camp_id)
    log(f"🔒 Lock activado para campamento ({camp.x},{camp.y})")

    try:
        while POPUP_ACTIVE:
            log("⏳ Esperando que el popup se cierre antes de enfriar el campamento")
            time.sleep(0.5)

        log(f"👉 Enfriando campamento en llamas ({camp.x},{camp.y})")

        cx, cy = camp.x + camp.w // 2, camp.y + camp.h // 2
        pyautogui.moveTo(cx, cy, duration=0.2)
        pyautogui.click()
        time.sleep(0.6)

        for attempt in range(3):
            if wait_and_click("assets/attack_icon.png", confidence=0.8, timeout=0.25):
                log("✅ Se encontró y clicó el botón de ataque")
                break
            log(f"⚠️ Intento {attempt+1}: botón de ataque no encontrado")
            time.sleep(0.6)
        else:
            log("❌ No se pudo abrir el menú de ataque")
            return False

        try:
            relojes_30 = int(spin_30min.get())
        except:
            relojes_30 = 0
        try:
            relojes_1h = int(spin_1h.get())
        except:
            relojes_1h = 0
        log(f"📦 Relojes disponibles: {relojes_30} de 30m, {relojes_1h} de 1h")

        steps = []
        if BONUS_ACTIVO:
            if relojes_30 >= 2:
                steps = [
                    ("assets/reduce_icon.png", {}),
                    ("assets/clock_30m.png", {"offset_x": 100}),
                    ("assets/clock_30m.png", {"offset_x": 100}),
                    ("assets/exit.png", {}),
                ]
            elif relojes_1h >= 1:
                steps = [
                    ("assets/reduce_icon.png", {}),
                    ("assets/clock_1h.png", {"offset_x": 100}),
                    ("assets/exit.png", {}),
                ]
            else:
                log("❌ Sin relojes suficientes")
                return False
        else:
            if relojes_30 >= 1 and relojes_1h >= 1:
                steps = [
                    ("assets/reduce_icon.png", {}),
                    ("assets/clock_1h.png", {"offset_x": 100}),
                    ("assets/arrow_left.png", {}),
                    ("assets/clock_30m.png", {"offset_x": 100}),
                    ("assets/exit.png", {}),
                ]
            else:
                log("❌ Sin relojes suficientes")
                return False

        usados_30 = 0
        usados_1h = 0

        for img, opts in steps:
            for attempt in range(3):
                while POPUP_ACTIVE:
                    log("⚠️ Popup detectado durante la secuencia, pausando...")
                    time.sleep(0.3)
                log(f"🔍 Buscando {img} (intento {attempt+1})...")
                if wait_and_click(img, confidence=0.75, timeout=3, offset_x=opts.get("offset_x", 0)):
                    log(f"✅ Click en {img}")
                    time.sleep(0.4)

                    if "clock_30m.png" in img:
                        usados_30 += 1
                    elif "clock_1h.png" in img:
                        usados_1h += 1
                    break
                else:
                    log(f"⚠️ No se pudo clicar {img} en el intento {attempt+1}")
                    time.sleep(0.4)
            else:
                log(f"❌ Falló la secuencia en {img}, abortando")
                return False

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

    if not safe_attack_click(): 
        return False

    pre_attack_steps = [
        "assets/confirm_attack.png",
        "assets/template_button.png"
    ]

    for step in pre_attack_steps:
        if not wait_and_click(step):
            return False

    if not wait_and_click("assets/attack_button.png"):
        return False
    time.sleep(0.4)

    if detect_popup("assets/min_troops.png", confidence=0.8, timeout=0.5):
        log("❌ Error: No hay suficientes tropas para atacar")
        wait_and_click("assets/error_close.png")
        wait_and_click("assets/exit2.png")
        global RUNNING
        RUNNING = False
        return False

    post_attack_steps = [
        "assets/horse_type.png",
        "assets/confirm_attack2.png"
    ]

    for step in post_attack_steps:
        if not wait_and_click(step):
            return False

    return True

def detect_fire_roi(camp: Detection) -> bool:
    x, y, w, h = camp.x, camp.y, camp.w, camp.h
    margin = 6
    rx = max(0, x - margin)
    ry = max(0, y - margin)
    rw = max(1, w + margin * 2)
    rh = max(1, h + margin * 2)
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
            return True
    return False

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

# -------------------- Seleccionador de Facciones --------------------
def choose_faction_window():
    def select_nomadas():
        global FACTION
        FACTION = "nomadas"
        w.destroy()
        launch_main_window()

    def select_samurais():
        global FACTION
        FACTION = "samurais"
        w.destroy()
        launch_main_window()

    w = tk.Tk()
    w.title("Selecciona Facción")
    ttk.Label(w, text="Selecciona tu facción:", font=("Segoe UI", 14)).pack(pady=20)
    frame_buttons = ttk.Frame(w)
    frame_buttons.pack(pady=10)
    btn_nomadas = ttk.Button(frame_buttons, text="Nómadas", command=select_nomadas, width=20)
    btn_nomadas.grid(row=0, column=0, padx=10)
    btn_samurais = ttk.Button(frame_buttons, text="Samuráis", command=select_samurais, width=20)
    btn_samurais.grid(row=0, column=1, padx=10)
    w.mainloop()

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
    """
    Watcher optimizado que solo monitorea el campamento seleccionado.
    Detecta fuego y llama a cool_down_camp por cada comandante disponible.
    """
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

        time.sleep(0.25)


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

# ---------- Start ----------
if __name__ == "__main__":
    choose_faction_window()
