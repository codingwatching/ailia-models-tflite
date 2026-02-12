# ailia MODELS TFLITE TCP HOST

# prepare client
# v is video input, s is video output
# python3 yolox.py -v "127.0.0.1:8000" -s "127.0.0.1:8001"

# find utils path
import os, sys

def find_and_append_util_path():
    current_dir = os.path.abspath(os.path.dirname(__file__))
    while current_dir != os.path.dirname(current_dir):
        potential_util_path = os.path.join(current_dir, 'util')
        if os.path.exists(potential_util_path):
            sys.path.append(potential_util_path)
            return
        current_dir = os.path.dirname(current_dir)
    raise FileNotFoundError("Couldn't find 'util' directory. Please ensure it's in the project directory structure.")

find_and_append_util_path()

# connect to client via tcp
import webcamera_utils
import cv2

cap = webcamera_utils.get_capture("0")
cap2 = None
writer = webcamera_utils.get_writer("127.0.0.1:8000", 480, 640, 20)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    writer.write(frame)
    if cap2 is None:
        cap2 = webcamera_utils.get_capture("127.0.0.1:8001")
    ret, frame2 = cap2.read()
    cv2.imshow("recv", frame)
    cv2.imshow("recv", frame2)

    if cv2.waitKey(1) == 27:
        break

cap.release()
writer.release()

