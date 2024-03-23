from norfair import Detection, Tracker, Video, draw_boxes
from yolo import YOLO
import cv2


name = "nwb5"
tracker = Tracker(distance_function="euclidean", distance_threshold=50, reid_distance_threshold=50, hit_counter_max=30, initialization_delay=15)
model = YOLO(weights="bestbottle.pt", thresh=0.05)

vid = f'vids/{name}.mp4'
model.run(path=vid)


