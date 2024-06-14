from ultralytics import YOLO
from ultralytics.engine.results import Results
from norfair import Detection, Tracker, Video, draw_boxes
from torch import cuda, tensor
from typing import List
from time import time
import numpy as np
import cv2
import argparse


class BottleDetector:

    def __init__(self, weights='bottle_weights.pt'):
        """
        Description:
            initializes a new YOLO model
        Parameters:
            weights (String)    : path to the .pt file containing the weights of the model
        Returns:
            None
        """
        self.model = YOLO(weights)
            
            
    def predict_frame(self, frame, thresh):
        """
        Description:
            runs the input frame through an inference pass on the class's model
        Parameters:
            frame (String)      : video frame to be processed and tracked
            slice (bool)        : whether the frame should be sliced via sahi for
                                  processing (note that this increases accuracy for
                                  detecting small objects, but increases inference time)
        Returns:
            List[Detections]: list of norfair Detection objects for the results of
                              running the model on the input frame
        """
        result = self.model(frame, conf=thresh, verbose=False, iou=0.4)
        result = self._yolo_detections_to_norfair_detections(result)
        return result
        
    def track( self, 
                    path = "demo.mp4", 
                    skip_frames = 30, 
                    tracker = None, 
                    actual_num_bottles = None, 
                    min_time = 1.0, 
                    min_frame_dist = 0.3,
                    show = False,
                    thresh = 0.05):
        """
        Description:
            This method gathers data from the video, running the tracking algorithm to detect bottles. Post-processing 
            is performed so as to filter out false positives, making it easier to lower detection thresholds in the case
            of false negatives. 
        Parameters:
            timestamps (String)         : txt file with timestamps for when each class is present
            path (String)               : the video to test the model on, associated with timestamps
            skip_frames (int)           : number of frames to be skipped before reading the next
            tracker (Tracker)           : a norfair Tracker object to be used for object tracking
            actual_num_bottles (int)    : the number of bottles that actually appear in the video, shows error metrics is present
            min_time (float)            : the minimum number of seconds a bottle must be on screen for it to be counted
            min_frame_dist (float)      : a float between 0.0 and 1.0 representing the fraction of the screen
                                          the bottle must travel horizontally to be counted
            thresh (float)              : confidence threshold the model should use
            show (bool)                 : whether or not the video should be displayed as tracking is run
        Returns:
            float                       : the accuracy of the model on the input video
        """

        #init video
        vidcap = cv2.VideoCapture(path)
        fps = (vidcap.get(cv2.CAP_PROP_FPS)+1)//skip_frames
        total = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        #init data storage for reading into
        tracked_bottles = {}
                    
        #declare a tracker that we can use to detect the bottles
        if tracker is None:
            tracker = Tracker(
                distance_function       = "euclidean", 
                distance_threshold      = 100, 
                reid_distance_threshold = 100, 
                hit_counter_max         = 30, 
                initialization_delay    = 2)

        #This is the master for loop that will read the entire video frame by frame
        success = vidcap.grab()
        i = 0
        while success:
            start = time()
            if i % skip_frames == 0:

                _, frame = vidcap.retrieve()

                #get width to use in post-processing
                width, _, _ = frame.shape

                #run object detection and update the tracker
                nf_detections = self.predict_frame(frame, thresh)
                tracked_objects = tracker.update(detections = nf_detections)

                #display the video if 'show' is True
                if show:
                    draw_boxes(frame, tracked_objects, thickness=3, draw_ids=True)
                    cv2.imshow("VID", frame)
                    cv2.waitKey(1)

                #iterate over each actively tracked object
                for obj in tracker.get_active_objects():

                    #convert bounding boxes from the tracked object to xy coordinates
                    (x1, y1), (x2, y2) = obj.last_detection.points
                    x = int(x1+x2)//2
                    y = int(y1+y2)//2

                    #if the object is already stored in the list of objects we've creates,
                    #then we append it to the list along with x, y, and the frame number.
                    #otherwise, create a new object and store the same data as a new point
                    if obj.id in tracked_bottles:
                        pts = tracked_bottles[obj.id][1]
                        pts.append((x, y, i))
                        tracked_bottles[obj.id] = (obj.age, pts)
                    else:
                        (x1, y1), (x2, y2) = obj.last_detection.points
                        x = int(x1+x2)//2
                        y = int(y1+y2)//2
                        tracked_bottles[obj.id] = (obj.age, [(x, y, i)])

                end = time()
                diff = str(skip_frames/(1e-4+end-start))[:5]
                print(f"Tracking frame {i//skip_frames} of {total//skip_frames} at {diff}fps", end="\r", flush=True)

            success = vidcap.grab()
            i += 1
        print()
            
        
        #this is where the post-processing will take place, iterating over each object
        bottles = []
        for id in tracked_bottles.keys():
            b = tracked_bottles[id]

            #sort the object coordinates by frame on which the coordinates were added
            pts = sorted(b[1], key = lambda x : x[2])

            #if the object is both on screen for long enough and travels far enough across
            #the screen horizontally, then we count that as a detected bottle in the final
            #result
            if (b[0]/fps) > min_time and (pts[-1][0] - pts[0][0]) > min_frame_dist * width:
                bottles.append(
                    {'tracking_id' : id, 
                    'seconds_in_view' : str(b[0]/fps)[:3], 
                    'x_dist_travelled' : pts[-1][0] - pts[0][0]})

        #print our metrics and return the error
        print("num_tracked:", len(bottles))
        if actual_num_bottles is not None:
            print("num_actual:", actual_num_bottles)
            print("error:", (actual_num_bottles - len(bottles)) / actual_num_bottles)
        return bottles
        
        
    #adapted from https://github.com/tryolabs/norfair/blob/master/demos/yolov7/src/demo.py
    def _yolo_detections_to_norfair_detections(self, yolo_detections: Results, track_points = "centroid") -> List[Detection]:
        """convert detections_as_xywh to norfair detections"""
        norfair_detections: List[Detection] = []
        boxes = yolo_detections[0].boxes.cpu()

        for i, _ in enumerate(boxes.cls):

            bbox = np.array(
                [
                    [boxes.xyxy[i][0], boxes.xyxy[i][1]],
                    [boxes.xyxy[i][2], boxes.xyxy[i][3]],
                ]
            )
            scores = np.array(
                [boxes.conf[i], boxes.conf[i]]
            )
            norfair_detections.append(
                Detection(
                    points=bbox, scores=scores, label=0
                )
            )
        return norfair_detections


if __name__ == "__main__":

    model = BottleDetector()
    parser = argparse.ArgumentParser()

    parser.add_argument("path", help="the file path to the video that is to be processed")    
    parser.add_argument("-s", "--show", action="store_true", help="if this is specified, the model will show the video as it is processed")
    parser.add_argument("-n", "--numbottles", type=int, default=None, help="the number of bottles that appear in the video; if this is specified, error will be calculated")
    parser.add_argument("-f", "--frameskip", type=int, default=1, help="the n for which every nth frame will be read from the video")
    parser.add_argument("-t", "--mintime", type=float, default=1.0, help="the minimum number of seconds a bottle must be on screen for it to be counted")
    parser.add_argument("-d", "--minframedist", type=float, default=1.0, help="a float between 0.0 and 1.0 representing the fraction of the screen the bottle must travel horizontally to be counted")
    parser.add_argument("-c", "--conf", type=float, default=0.05, help="confidence threshold the model should use")

    args = parser.parse_args()

    detections = model.track(
        path=args.path, 
        actual_num_bottles=args.numbottles, 
        min_time=args.mintime, 
        min_frame_dist=args.minframedist, 
        skip_frames=args.frameskip, 
        show=args.show, 
        thresh=args.conf)
    for x in detections: print(x)
