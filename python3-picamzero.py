from picamzero import Camera
from time import sleep

cam = Camera()
cam.start_preview()
sleep(5)
cam.stop_preview() # Stop the previews