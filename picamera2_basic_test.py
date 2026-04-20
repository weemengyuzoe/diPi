from picamera2 import Picamera2, Preview
from gpiozero import Button
from signal import pause
import time

cam = Picamera2()
cam_config = cam.create_preview_configuration()
cam.configure(cam_config)

button = Button(5)


def startpreview():
    cam.start_preview(Preview.QTGL)
    print("Start!")
    cam.start()
    time.sleep(5)
    cam.stop_preview()

button.when_pressed = startpreview

pause()
