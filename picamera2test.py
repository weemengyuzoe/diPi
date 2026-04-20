from picamera2 import Picamera2, Preview
import time

cam = Picamera2()
camera_config = cam.create_preview_configuration()
cam.configure(camera_config)

cam.start_preview(Preview.QTGL)
cam.start()
time.sleep(5)
cam.stop_preview()
