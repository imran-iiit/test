from PIL import ImageGrab
import time
import datetime

PATH = '/Users/aniron/Desktop/BCP/' # Change this
DURATION_SECONDS = 60*30

for _ in range(30):
    # ImageGrab.grab().save("screen_capture.jpg", "JPEG")
    name = PATH + "%s.png" % datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    ImageGrab.grab().save(name)
    print('Captured: %s' % name)
    time.sleep(DURATION_SECONDS)