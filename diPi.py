from picamera2 import Picamera2
from libcamera import controls
from gpiozero import Button, LED
import numpy as np
import cv2
import time
import os

# ── Buttons ───────────────────────────────────────────────
BTN_CAPTURE = Button(5,  pull_up=True, bounce_time=0.05)
BTN_TEMP    = Button(6,  pull_up=True, bounce_time=0.05)
BTN_TINT       = Button(13, pull_up=True, bounce_time=0.05)
BTN_FOCUS_NEAR = Button(19, pull_up=True, bounce_time=0.05)
BTN_FOCUS_FAR  = Button(26, pull_up=True, bounce_time=0.05)

# ── LEDs ──────────────────────────────────────────────────
LED_RED    = LED(16)
LED_YELLOW = LED(20)
LED_GREEN  = LED(21)

def leds_off():
    LED_RED.off(); LED_YELLOW.off(); LED_GREEN.off()

def led_set(red=False, yellow=False, green=False):
    LED_RED.value    = red
    LED_YELLOW.value = yellow
    LED_GREEN.value  = green

# ── Lux thresholds ─────────────────────────────────────────
LUX_RED_LOW     = 10
LUX_YELLOW_LOW  = 50
LUX_YELLOW_HIGH = 5000
LUX_RED_HIGH    = 20000

CAPTURE_GREEN_DURATION = 2.0
capture_green_until    = 0.0

_lux_history    = []
LUX_HISTORY_LEN = 8

def smooth_lux(raw):
    _lux_history.append(raw)
    if len(_lux_history) > LUX_HISTORY_LEN:
        _lux_history.pop(0)
    return sum(_lux_history) / len(_lux_history)

def update_exposure_leds(lux):
    if time.time() < capture_green_until:
        led_set(green=True)
        return
    if lux < LUX_RED_LOW or lux > LUX_RED_HIGH:
        led_set(red=True)
    elif lux < LUX_YELLOW_LOW or lux > LUX_YELLOW_HIGH:
        led_set(yellow=True)
    else:
        led_set(green=True)

# ── Temp state ─────────────────────────────────────────────
grade    = {"temperature": -1.0}
temp_dir = 1
STEP     = 0.1
MIN_VAL, MAX_VAL = -1.0, 1.0

# ── Tint state ─────────────────────────────────────────────
tint = {"value": 0.0}
tint_dir = 1
TINT_STEP = 0.1
TINT_MIN, TINT_MAX = -1.0, 1.0

# ── Focus state ────────────────────────────────────────────
focus      = {"lens_position": 0.0}
focus_dir  = 1
FOCUS_STEP = 0.3
FOCUS_MIN  = 0.0
FOCUS_MAX  = 12.0

SAVE_DIR = "/home/princess"
os.makedirs(SAVE_DIR, exist_ok=True)

# ── Hold behaviour tuning ────────────────────────────────
HOLD_DELAY = 0.4        # seconds before hold kicks in
REPEAT_RATE = 0.05      # update interval while holding

FOCUS_SPEED = 3.0       # units per second
TEMP_SPEED  = 1.5       # units per second
TINT_SPEED  = 1.5       # units per second

# ── Runtime state ──────────────────────────────────────────
freeze_until  = 0.0
frozen_frame  = None
current_lux   = 100.0

last_action = {"temp":0, "tint":0, "focus":0, "capture":0}
DEBOUNCE = 0.15

button_press_time = {
    "temp": None,
    "tint": None,
    "focus_near": None,
    "focus_far": None
}

last_repeat_time = {
    "temp": 0,
    "tint": 0,
    "focus": 0
}

# ── Helpers ───────────────────────────────────────────────
def cycle_value(current, direction, step, min_val, max_val):
    nv = round(current + direction * step, 2)
    if nv >= max_val:
        return max_val, -1
    if nv <= min_val:
        return min_val, 1
    return nv, direction

def apply_grade(img):
    img  = img.astype(np.float32)
    temp = grade["temperature"]
    tint_val = tint["value"]
    
    TS   = 40.0
    TiS = 30.0 #tint strength
    
    # Temperature (blue ↔ red)
    img[:,:,0] = np.clip(img[:,:,0] + (-temp * TS), 0,255)
    img[:,:,1] = np.clip(img[:,:,1] + ( temp * TS * 0.2), 0,255)
    img[:,:,2] = np.clip(img[:,:,2] + ( temp * TS), 0,255)

    # Tint (green ↔ magenta)
    img[:,:,1] = np.clip(img[:,:,1] + (tint_val * TiS), 0,255)
    img[:,:,0] = np.clip(img[:,:,0] - (tint_val * TiS * 0.5), 0,255)
    img[:,:,2] = np.clip(img[:,:,2] - (tint_val * TiS * 0.5), 0,255)

    return img.astype(np.uint8)

def put_text(img, text, pos, scale=0.55, thickness=1):
    font = cv2.FONT_HERSHEY_SIMPLEX
    x,y = pos
    cv2.putText(img,(text),(x+1,y+1),font,scale,(0,0,0),thickness+1)
    cv2.putText(img,(text),(x,y),font,scale,(255,255,255),thickness)

def exposure_zone_str(lux):
    if lux < LUX_RED_LOW: return "TOO DARK"
    if lux < LUX_YELLOW_LOW: return "dim"
    if lux > LUX_RED_HIGH: return "BLOWN OUT"
    if lux > LUX_YELLOW_HIGH: return "bright"
    return "good"

def draw_overlay(frame, lux):
    out = frame.copy()
    put_text(out,f"Temp {grade['temperature']:+.1f}",(10,25))
    put_text(out,f"Tint {tint['value']:+.1f}",(10,50))
    put_text(out,f"Focus {focus['lens_position']:.1f}",(10,75))
    put_text(out,f"Lux {lux:.0f} [{exposure_zone_str(lux)}]",(10,100))
    return out

def fit_to_screen(img,w,h):
    ih,iw = img.shape[:2]
    s = min(w/iw,h/ih)
    nw,nh = int(iw*s),int(ih*s)
    canvas = np.zeros((h,w,3),dtype=np.uint8)
    yo,xo = (h-nh)//2,(w-nw)//2
    canvas[yo:yo+nh,xo:xo+nw] = cv2.resize(img,(nw,nh))
    return canvas

def set_lens_position(lp):
    cam.set_controls({"LensPosition":lp})

def apply_quality_controls():
    full_res = cam.camera_properties['PixelArraySize']
    cam.set_controls({
        "AfMode": controls.AfModeEnum.Manual,
        "LensPosition": focus["lens_position"],
        "Sharpness":2.0,
        "Saturation":1.2,
        "Contrast":1.1,
        "NoiseReductionMode": controls.draft.NoiseReductionModeEnum.HighQuality,
        "ScalerCrop": (0,0,full_res[0], full_res[1]),
    })

# ── Input ─────────────────────────────────────────────────
def poll_inputs():
    global temp_dir, tint_dir
    now = time.time()
    action = None

    # ───────────────── TEMP ─────────────────
    if BTN_TEMP.is_pressed:
        if button_press_time["temp"] is None:
            button_press_time["temp"] = now

        held_time = now - button_press_time["temp"]

        # HOLD → smooth sweep
        if held_time > HOLD_DELAY:
            if now - last_repeat_time["temp"] > REPEAT_RATE:
                last_repeat_time["temp"] = now
                grade["temperature"] += temp_dir * TEMP_SPEED * REPEAT_RATE
                grade["temperature"] = max(MIN_VAL, min(MAX_VAL, grade["temperature"]))

        # TAP → cycle
        elif now - last_action["temp"] > DEBOUNCE:
            last_action["temp"] = now
            grade["temperature"], temp_dir = cycle_value(
                grade["temperature"], temp_dir, STEP, MIN_VAL, MAX_VAL
            )

    else:
        button_press_time["temp"] = None

    # ───────────────── TINT ─────────────────
    if BTN_TINT.is_pressed:
        if button_press_time["tint"] is None:
            button_press_time["tint"] = now

        held_time = now - button_press_time["tint"]

        if held_time > HOLD_DELAY:
            if now - last_repeat_time["tint"] > REPEAT_RATE:
                last_repeat_time["tint"] = now
                tint["value"] += tint_dir * TINT_SPEED * REPEAT_RATE
                tint["value"] = max(TINT_MIN, min(TINT_MAX, tint["value"]))

        elif now - last_action["tint"] > DEBOUNCE:
            last_action["tint"] = now
            tint["value"], tint_dir = cycle_value(
                tint["value"], tint_dir, TINT_STEP, TINT_MIN, TINT_MAX
            )

    else:
        button_press_time["tint"] = None

    # ───────────────── FOCUS (HOLD ONLY) ─────────────────
    focus_changed = False

    if BTN_FOCUS_NEAR.is_pressed:
        if button_press_time["focus_near"] is None:
            button_press_time["focus_near"] = now

        if now - last_repeat_time["focus"] > REPEAT_RATE:
            last_repeat_time["focus"] = now
            focus["lens_position"] -= FOCUS_SPEED * REPEAT_RATE
            focus_changed = True

    else:
        button_press_time["focus_near"] = None

    if BTN_FOCUS_FAR.is_pressed:
        if button_press_time["focus_far"] is None:
            button_press_time["focus_far"] = now

        if now - last_repeat_time["focus"] > REPEAT_RATE:
            last_repeat_time["focus"] = now
            focus["lens_position"] += FOCUS_SPEED * REPEAT_RATE
            focus_changed = True

    else:
        button_press_time["focus_far"] = None

    if focus_changed:
        focus["lens_position"] = max(FOCUS_MIN, min(FOCUS_MAX, focus["lens_position"]))
        set_lens_position(round(focus["lens_position"], 2))

    # ───────────────── CAPTURE ─────────────────
    if BTN_CAPTURE.is_pressed and now - last_action["capture"] > DEBOUNCE:
        last_action["capture"] = now
        action = "capture"

    return action

# ── Camera setup ──────────────────────────────────────────
cam = Picamera2()

still_config = cam.create_still_configuration(
    main={"format":"RGB888","size":(4608,2592)},
    lores={"format":"YUV420","size":(1280,720)},
    display="lores",
)

preview_config = cam.create_video_configuration(
    main={"format":"RGB888","size":(1280,720)},
)

cam.configure(preview_config)
cam.start()
time.sleep(0.3)
apply_quality_controls()

sensor_size = cam.camera_properties["PixelArraySize"]
full_fov = (0, 0, sensor_size[0], sensor_size[1])
cam.set_controls({"ScalerCrop": full_fov})

WINDOW="Camera"
cv2.namedWindow(WINDOW, cv2.WINDOW_GUI_NORMAL)
cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

_probe = cam.capture_array()
cv2.imshow(WINDOW,_probe)
cv2.waitKey(1)
rect=cv2.getWindowImageRect(WINDOW)
SW,SH = rect[2] or 480, rect[3] or 320

# ── Main loop ─────────────────────────────────────────────
while True:

    action = poll_inputs()
    now = time.time()

    if now < freeze_until and frozen_frame is not None:
        cv2.imshow(WINDOW,frozen_frame)
        update_exposure_leds(current_lux)

    else:
        frame = cam.capture_array()
        metadata = cam.capture_metadata()

        raw_lux = metadata.get("Lux",current_lux)
        current_lux = smooth_lux(raw_lux)
        update_exposure_leds(current_lux)

        graded = apply_grade(frame)
        display = fit_to_screen(graded,SW,SH)
        display = draw_overlay(display,current_lux)
        cv2.imshow(WINDOW,display)

        if action=="capture":

            cam.stop()
            cam.configure(still_config)
            apply_quality_controls()
            cam.start()
            time.sleep(0.2)

            full_frame = cam.capture_array()

            cam.stop()
            cam.configure(preview_config)
            apply_quality_controls()
            cam.start()

            graded_full = apply_grade(full_frame)

            filename = time.strftime("photo_%Y%m%d_%H%M%S.jpg")
            filepath = os.path.join(SAVE_DIR,filename)
            cv2.imwrite(filepath,graded_full,[cv2.IMWRITE_JPEG_QUALITY,97])

            capture_green_until = now + CAPTURE_GREEN_DURATION
            led_set(green=True)

            frozen_frame = fit_to_screen(graded_full,SW,SH)
            put_text(frozen_frame,"SAVED!",(SW//2-40,SH//2),1.2,2)

            freeze_until = now + 4.0
            cv2.imshow(WINDOW,frozen_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ── Cleanup ───────────────────────────────────────────────
leds_off()
cam.stop()
cv2.destroyAllWindows()
