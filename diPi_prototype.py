from picamera2 import Picamera2
from gpiozero import Button
import numpy as np
import cv2
import time
import os
import glob

# ══════════════════════════════════════════════════════════
#  PIN WIRING GUIDE
#
#  CAPTURE BUTTON        signal → pin 29  (BCM 5)
#                        ground → pin 30
#
#  TEMPERATURE UP        signal → pin 31  (BCM 6)   warmer / more orange
#                        ground → pin 34  ┐  F-F splitter
#  TINT UP               signal → pin 35  (BCM 19)  more magenta
#                        ground → pin 34  ┘
#
#  TEMPERATURE DOWN      signal → pin 33  (BCM 13)  cooler / more blue
#                        ground → pin 39  ┐  F-F splitter
#  TINT DOWN             signal → pin 37  (BCM 26)  more green
#                        ground → pin 39  ┘
# ══════════════════════════════════════════════════════════

BTN_CAPTURE  = Button(5,  pull_up=True, bounce_time=0.05)
BTN_TEMP_UP  = Button(6,  pull_up=True, bounce_time=0.05)
BTN_TEMP_DN  = Button(13, pull_up=True, bounce_time=0.05)
BTN_TINT_UP  = Button(19, pull_up=True, bounce_time=0.05)
BTN_TINT_DN  = Button(26, pull_up=True, bounce_time=0.05)

COLOUR_BTNS = [BTN_TEMP_UP, BTN_TEMP_DN, BTN_TINT_UP, BTN_TINT_DN]

# ══════════════════════════════════════════════════════════
#  COLOUR GRADE STATE
#  temperature: -1.0 (cool/blue) to +1.0 (warm/orange)
#  tint:        -1.0 (green)     to +1.0 (magenta)
# ══════════════════════════════════════════════════════════

grade = {"temperature": 0.0, "tint": 0.0}
STEP = 0.1
MIN_VAL, MAX_VAL = -1.0, 1.0

SAVE_DIR = "/home/princess"
os.makedirs(SAVE_DIR, exist_ok=True)

# ── App state ─────────────────────────────────────────────
STATE         = "camera"   # "camera" or "gallery"
gallery_imgs  = []
gallery_idx   = 0
gallery_cache = None       # rendered gallery frame, rebuilt on index change

# Debounce timers — track last time each button action fired
last_action = {
    "temp_up":  0, "temp_dn":  0,
    "tint_up":  0, "tint_dn":  0,
    "capture":  0, "combo":    0,
    "scroll":   0,
}
DEBOUNCE      = 0.15   # seconds between repeated single-presses
COMBO_DEBOUNCE = 0.5   # seconds between combo triggers


# ── Colour grading ────────────────────────────────────────

def apply_grade(img_array: np.ndarray) -> np.ndarray:
    """
    Apply temperature + tint grade to a BGR numpy array.
    BGR order: channel 0=B, 1=G, 2=R
    """
    img  = img_array.astype(np.float32)
    temp = grade["temperature"]
    tint = grade["tint"]
    TS   = 40.0   # temperature strength
    NS   = 30.0   # tint strength

    img[:, :, 0] = np.clip(img[:, :, 0] + (-temp * TS) + ( tint * NS * 0.7), 0, 255)
    img[:, :, 1] = np.clip(img[:, :, 1] + ( temp * TS * 0.2) + (-tint * NS), 0, 255)
    img[:, :, 2] = np.clip(img[:, :, 2] + ( temp * TS) + ( tint * NS * 0.7), 0, 255)
    return img.astype(np.uint8)


# ── Overlay helpers ───────────────────────────────────────

def put_text(img, text, pos, scale=0.55, thickness=1):
    """White text with black shadow for readability on any background."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    x, y = pos
    cv2.putText(img, text, (x+1, y+1), font, scale, (0,0,0),     thickness+1)
    cv2.putText(img, text, (x,   y),   font, scale, (255,255,255), thickness)

def draw_camera_overlay(frame: np.ndarray) -> np.ndarray:
    out = frame.copy()
    tw  = 'warm'    if grade['temperature'] > 0 else 'cool'    if grade['temperature'] < 0 else 'neutral'
    tn  = 'magenta' if grade['tint']        > 0 else 'green'   if grade['tint']        < 0 else 'neutral'
    put_text(out, f"Temp {grade['temperature']:+.1f} ({tw})", (10, 25))
    put_text(out, f"Tint {grade['tint']:+.1f} ({tn})",        (10, 50))
    put_text(out, "Black=Capture  Red=Temp Up  Blue=Temp Down  Green=Tint Down  Yellow=Tint Up",          (10, 75))
    return out

def draw_gallery_overlay(frame: np.ndarray) -> np.ndarray:
    out   = frame.copy()
    total = len(gallery_imgs)
    name  = os.path.basename(gallery_imgs[gallery_idx]) if gallery_imgs else "—"
    put_text(out, f"{gallery_idx+1}/{total}  {name}", (10, 25))
    put_text(out, "Temp UP/DN = scroll  |  Capture = back", (10, 50))
    return out


# ── Fit image to screen ───────────────────────────────────

def fit_to_screen(img: np.ndarray, w: int, h: int) -> np.ndarray:
    ih, iw = img.shape[:2]
    scale  = min(w / iw, h / ih)
    nw, nh = int(iw * scale), int(ih * scale)
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    yo     = (h - nh) // 2
    xo     = (w - nw) // 2
    canvas[yo:yo+nh, xo:xo+nw] = cv2.resize(img, (nw, nh))
    return canvas


# ── Gallery helpers ───────────────────────────────────────

def refresh_gallery():
    global gallery_imgs, gallery_idx
    gallery_imgs = sorted(
        glob.glob(os.path.join(SAVE_DIR, "*.jpg")),
        key=os.path.getmtime
    )
    gallery_idx = max(0, len(gallery_imgs) - 1)   # open on latest photo

def build_gallery_frame(sw, sh):
    if not gallery_imgs:
        blank = np.zeros((sh, sw, 3), dtype=np.uint8)
        put_text(blank, "No photos yet.", (10, sh // 2))
        return blank
    img = cv2.imread(gallery_imgs[gallery_idx])
    if img is None:
        img = np.zeros((sh, sw, 3), dtype=np.uint8)
    img = fit_to_screen(img, sw, sh)
    return draw_gallery_overlay(img)


# ── Input polling (called every frame) ───────────────────
# All button logic lives here — no when_pressed callbacks —
# so there is no risk of handler overwrites or threading issues.

def poll_inputs(sw, sh):
    global STATE, gallery_idx, gallery_cache

    now             = time.time()
    pressed_colours = [b for b in COLOUR_BTNS if b.is_pressed]
    combo           = len(pressed_colours) >= 2

    # ── Combo: enter gallery ──────────────────────────────
    if combo and STATE == "camera":
        if now - last_action["combo"] > COMBO_DEBOUNCE:
            last_action["combo"] = now
            refresh_gallery()
            STATE         = "gallery"
            gallery_cache = build_gallery_frame(sw, sh)
            print("  → Gallery mode")
        return   # don't process single-button actions this frame

    # ── Camera mode single-button actions ─────────────────
    if STATE == "camera":
        if BTN_TEMP_UP.is_pressed and now - last_action["temp_up"] > DEBOUNCE:
            last_action["temp_up"] = now
            grade["temperature"]   = round(min(grade["temperature"] + STEP, MAX_VAL), 2)

        if BTN_TEMP_DN.is_pressed and now - last_action["temp_dn"] > DEBOUNCE:
            last_action["temp_dn"] = now
            grade["temperature"]   = round(max(grade["temperature"] - STEP, MIN_VAL), 2)

        if BTN_TINT_UP.is_pressed and now - last_action["tint_up"] > DEBOUNCE:
            last_action["tint_up"] = now
            grade["tint"]          = round(min(grade["tint"] + STEP, MAX_VAL), 2)

        if BTN_TINT_DN.is_pressed and now - last_action["tint_dn"] > DEBOUNCE:
            last_action["tint_dn"] = now
            grade["tint"]          = round(max(grade["tint"] - STEP, MIN_VAL), 2)

        if BTN_CAPTURE.is_pressed and now - last_action["capture"] > DEBOUNCE:
            last_action["capture"] = now
            return "capture"

    # ── Gallery mode single-button actions ────────────────
    elif STATE == "gallery":
        scrolled = False

        if BTN_TEMP_UP.is_pressed and now - last_action["scroll"] > DEBOUNCE:
            last_action["scroll"] = now
            gallery_idx  = (gallery_idx + 1) % len(gallery_imgs)
            scrolled     = True

        if BTN_TEMP_DN.is_pressed and now - last_action["scroll"] > DEBOUNCE:
            last_action["scroll"] = now
            gallery_idx  = (gallery_idx - 1) % len(gallery_imgs)
            scrolled     = True

        if scrolled:
            gallery_cache = build_gallery_frame(sw, sh)

        if BTN_CAPTURE.is_pressed and now - last_action["capture"] > DEBOUNCE:
            last_action["capture"] = now
            STATE = "camera"
            print("  → Camera mode")

    return None


# ── Camera setup ──────────────────────────────────────────

cam    = Picamera2()
config = cam.create_video_configuration(
    main={"format": "RGB888", "size": (1280, 720)}
)
cam.configure(config)
cam.start()

sensor_size = cam.camera_properties["PixelArraySize"]
full_fov = (0, 0, sensor_size[0], sensor_size[1])
cam.set_controls({"ScalerCrop": full_fov})


# Fullscreen window — WINDOW_GUI_NORMAL removes OpenCV's own toolbar
WINDOW = "Camera"
cv2.namedWindow(WINDOW, cv2.WINDOW_GUI_NORMAL)
cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# Measure actual screen size from the window after first show
_probe = cam.capture_array()
cv2.imshow(WINDOW, _probe)
cv2.waitKey(1)
rect   = cv2.getWindowImageRect(WINDOW)
SW     = rect[2] if rect[2] > 0 else 480   # screen width
SH     = rect[3] if rect[3] > 0 else 320   # screen height
print(f"  Screen detected: {SW}x{SH}")

print("═══════════════════════════════════")
print("  Camera ready                     ")
print("  Temp UP/DN  → warmer / cooler    ")
print("  Tint UP/DN  → magenta / green    ")
print("  Capture     → take photo         ")
print("  Any 2 colour btns → gallery      ")
print("  Q key → quit                     ")
print("═══════════════════════════════════")


# ── Main loop ─────────────────────────────────────────────

while True:

    action = poll_inputs(SW, SH)

    if STATE == "camera":
        frame   = cam.capture_array()
        graded  = apply_grade(frame)
        display = fit_to_screen(graded, SW, SH)
        display = draw_camera_overlay(display)
        cv2.imshow(WINDOW, display)

        if action == "capture":
            filename = time.strftime("photo_%Y%m%d_%H%M%S.jpg")
            filepath = os.path.join(SAVE_DIR, filename)
            cv2.imwrite(filepath, graded, [cv2.IMWRITE_JPEG_QUALITY, 95])
            print(f"  Saved → {filepath}")
            print(f"  Grade: temp={grade['temperature']:+.1f}  tint={grade['tint']:+.1f}")

    else:  # gallery
        if gallery_cache is not None:
            cv2.imshow(WINDOW, gallery_cache)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.stop()
cv2.destroyAllWindows()
