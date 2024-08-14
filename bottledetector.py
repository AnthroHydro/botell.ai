##################################################
## Botell.ai
##
## Software package that detects and counts bottles in streams.
##
##################################################
## Author: Andrew Heller and Jason Davison
## Copyright: Copyright 2024, Botell.ai
## License: Creative Commons Attribution 4.0 International
## Version: 1.2.5
## Maintainer: Andrew Heller
## Email: abh037@gmail.com and davisonj@cua.edu
## Status: In-progress -- 8/14/2024 last update
##################################################


from ultralytics import YOLO
from ultralytics.engine.results import Results
from norfair import Detection, Tracker, Video, draw_boxes
from typing import List
from time import time
from datetime import timedelta
import numpy as np
import pandas as pd
import cv2
import argparse
import openpyxl
from openpyxl.worksheet.datavalidation import DataValidation


class BottleDetector:

    def __init__(self, weights='bottle_weights.pt'):
        """
        Description:
            initializes a new YOLO model
        Parameters:
            weights (String)    : path to the .pt file containing the weights of the model
        Returns:
            BottleDetector class object
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
        result = self.model(frame, conf=thresh, verbose=False, iou=0.7)
        result = self._yolo_detections_to_norfair_detections(result)
        return result
        
    def track( 
                self, 
                path = "demo.mp4", 
                skip_frames = 3, 
                tracker = None, 
                actual_num_bottles = None, 
                min_time = 1.0, 
                min_frame_dist = 0.3,
                show = False,
                thresh = 0.05,
                save = None,
                moving_cam=False):
        """
        Description:
            This method gathers data from the video, running the tracking algorithm to detect bottles. Post-processing 
            is performed so as to filter out false positives, making it easier to lower detection thresholds in the case
            of false negatives. 
        Parameters:
            path (String)               : the video to test the model on, associated with timestamps
            skip_frames (int)           : number of frames to be skipped before reading the next
            tracker (Tracker)           : a norfair Tracker object to be used for object tracking
            actual_num_bottles (int)    : the number of bottles that actually appear in the video, shows error metrics is present
            min_time (float)            : the minimum number of seconds a bottle must be on screen for it to be counted
            min_frame_dist (float)      : a float between 0.0 and 1.0 representing the fraction of the screen
                                          the bottle must travel horizontally to be counted
            show (bool)                 : whether or not the video should be displayed as tracking is run
            thresh (float)              : confidence threshold the model should use
            save (String)               : either None or the name under which the video should be saved with tracking annotations
        Returns:
            int                         : number of bottles that were successfully tracked and counted in the video
            float                       : the accuracy of the model on the input video
            List[tuple]                 : list of the metrics of the bottles tracked in the video
        """

        #init video
        vidcap = cv2.VideoCapture(path)
        fps = vidcap.get(cv2.CAP_PROP_FPS)
        total = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        #init data storage for reading into
        tracked_bottles = {}
                    
        #declare a tracker that we can use to detect the bottles
        if tracker is None:
            tracker = Tracker(
                distance_function       = "euclidean", 
                distance_threshold      = 200, 
                reid_distance_threshold = 300, 
                hit_counter_max         = 30, 
                initialization_delay    = 4)

        #This is the master for loop that will read the entire video frame by frame
        height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
        success = vidcap.grab()
        assert success, "Could not open video!"
        backSub = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
        i = 0
        framerate = 1.0
        diff = []
        if save is not None:
            assert save[-4:] == ".avi", "save argument must be a .avi file"
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(save, fourcc, fps/skip_frames, (width, height), isColor=True)
            if height > 640:
                writer = cv2.VideoWriter(save, fourcc, fps/skip_frames, ((640*width) // height, 640), isColor=True)
        while success:
            start = time()
            if i % skip_frames == 0:

                _, frame = vidcap.retrieve()
                if height > 640:
                        frame = cv2.resize(frame, ((640*width) // height, 640))
                mask = backSub.apply(frame)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8), iterations=2)
                mask = cv2.dilate(mask, np.ones((7, 7), np.uint8), iterations=5)
                mask = cv2.blur(mask, (height//10, height//10))
                scaled_heatmap = (((mask - mask.min()) / (mask.max() - mask.min() + 1e-7))/2) - (1/2)


                #run object detection and update the tracker
                nf_detections = self.predict_frame(frame, thresh)
                if not moving_cam:
                    to_be_deleted = []
                    for j, d in enumerate(nf_detections):
                        ((x1, y1), (x2, y2)) = d.points.astype(int)
                        nf_detections[j].scores += np.mean(scaled_heatmap[y1:y2, x1:x2])
                        if np.mean(nf_detections[j].scores) < thresh:
                            to_be_deleted.append(j)
                    nf_detections = [d for k, d in enumerate(nf_detections) if k not in to_be_deleted]    
                tracked_objects = tracker.update(detections = nf_detections)

                #display the video if 'show' is True
                annotated_frame = draw_boxes(frame, tracked_objects, thickness=3, draw_ids=True, draw_labels=True, draw_scores=True)
                if save is not None:
                    writer.write(annotated_frame)
                if show:
                    cv2.imshow("VID", annotated_frame)
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
                        confs = tracked_bottles[obj.id][0]
                        confs.append(obj.last_detection.scores[0])
                        tracked_bottles[obj.id] = (confs, pts)
                    else:
                        (x1, y1), (x2, y2) = obj.last_detection.points
                        x = int(x1+x2)//2
                        y = int(y1+y2)//2
                        tracked_bottles[obj.id] = ([obj.last_detection.scores[0]], [(x, y, i)])

                end = time()
                if (end-start) > 0.005:
                    diff.append(skip_frames/(end-start))
                if i%(skip_frames*15) == 0:
                    framerate = np.mean(diff)
                    diff = []
                print(f" > Processing video ({str(100*(i+skip_frames)/total)[:5]}%) at {str(framerate)[:5]} frames per second...", end="\r", flush=True)

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
            if not moving_cam:
                if ((pts[-1][2] - pts[0][2])/fps) > (1.5+min_time) and np.linalg.norm(np.array(pts[-1][:-1]) - np.array(pts[0][:-1])) > min_frame_dist * width:
                    bottles.append(
                        {'tracking_id' : id, 
                        'start_time' : timedelta(seconds=(pts[0][2]//fps)),
                        'end_time' : timedelta(seconds=(pts[-1][2]//fps) - 2),
                        'confs' : b[0],
                        'dist_travelled' : np.linalg.norm(np.array(pts[-1][:-1]) - np.array(pts[0][:-1]))})
            else:
                if ((pts[-1][2] - pts[0][2])/fps) > min_time:
                    bottles.append(
                        {'tracking_id' : id, 
                        'start_time' : timedelta(seconds=(pts[0][2]//fps)),
                        'end_time' : timedelta(seconds=(pts[-1][2]//fps) - 2),
                        'confs' : b[0],
                        'dist_travelled' : np.linalg.norm(np.array(pts[-1][:-1]) - np.array(pts[0][:-1]))})
        #save metrics to variables and return them
        nt = len(bottles)
        er = None
        if actual_num_bottles is not None:
            er = (actual_num_bottles - len(bottles)) / actual_num_bottles
        return nt, er, bottles
        
        
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

    print()
    parser = argparse.ArgumentParser()

    parser.add_argument("path", help="the file path to the video that is to be processed")    
    parser.add_argument("-s", "--show", action="store_true", help="if this is specified, the model will show the video as it is processed")
    parser.add_argument("-n", "--numbottles", type=int, default=None, help="the number of bottles that appear in the video; if this is specified, error will be calculated")
    parser.add_argument("-f", "--framerate", type=int, default=None, help="the frames per second at which the video should be processed")
    parser.add_argument("-t", "--mintime", type=float, default=-1.0, help="the minimum number of seconds a bottle must be on screen for it to be counted")
    parser.add_argument("-d", "--minframedist", type=float, default=0.3, help="a float between 0.0 and 1.0 representing the fraction of the screen the bottle must travel horizontally to be counted")
    parser.add_argument("-c", "--conf", type=float, default=0.02, help="confidence threshold the model should use")
    parser.add_argument("-o", "--output", type=str, default="output.txt", help="the name of the .txt the tool should write metrics to")
    parser.add_argument("-v", "--verify", type=str, default=None, help="filename with the .xlsx extension to be created for verification (a .avi file with the same name will be saved as well)")
    parser.add_argument("-m", "--movingcam", action="store_true", help="if the camera is not stationary in the video, specifying this will disable the post-processing algorithm (which assumes a still camera)")

    args = parser.parse_args()
    assert args.output[-4:] == '.txt', "--output argument must be a .txt file"

    save = None
    if args.verify: 
        assert args.verify[-5:] == ".xlsx", "--verify argument must be a .xlsx file"
        save = f"{args.verify[:-5]}.avi"


    print(" > Loading model...")
    model = BottleDetector()
    if args.framerate is not None: 
        skipframes = int((cv2.VideoCapture(args.path).get(cv2.CAP_PROP_FPS)+1)/args.framerate)
    else:
        skipframes = 1
    num_tracked, error, detections = model.track(
                                            path=args.path, 
                                            actual_num_bottles=args.numbottles, 
                                            min_time=args.mintime, 
                                            min_frame_dist=args.minframedist, 
                                            skip_frames=skipframes, 
                                            show=args.show, 
                                            thresh=args.conf,
                                            save=save,
                                            moving_cam=args.movingcam)
    error = "N/A" if error is None else error
    num_actual = "N/A" if args.numbottles is None else args.numbottles

    starttimes = []
    endtimes = []
    ids = []
    bools = []
    print(f" > Writing to {args.output}...")

    vars_dict = vars(args)

    with open(args.output, 'w') as f:
        for k in vars_dict:
            if vars_dict[k] == parser.get_default(k):
                f.write(f"{k}: {vars_dict[k]} (Default)\n")
            else:
                f.write(f"{k}: {vars_dict[k]}\n")
        f.write(f'Number of bottles tracked: {num_tracked}\n')
        f.write(f'Actual number of bottles:  {num_actual}\n')
        f.write(f'Error:  {error}\n\n')
        for i, b in enumerate(detections):
            starttimes.append(str(b['start_time']))
            endtimes.append(str(b['end_time']))
            ids.append(b['tracking_id'])
            bools.append("")
            f.write(f"Detection {i+1}:\n")
            f.write(f"\tID                 - {b['tracking_id']}\n")
            f.write(f"\tconf               - {str(np.mean(np.unique(b['confs'])))[:7]}\n")
            f.write(f"\tstart time         - {b['start_time']}\n")
            f.write(f"\tend time           - {b['end_time']}\n")
            f.write(f"\tpixels travelled   - {b['dist_travelled']}\n")

    if args.verify is not None:
        data = {'ID' : ids, 'Start' : starttimes, 'End' : endtimes, 'True?' : bools}
        df = pd.DataFrame.from_dict(data)
        df.to_excel(args.verify, sheet_name='sheet1')
        wb = openpyxl.load_workbook(args.verify)
        ws = wb['sheet1']
        dv = DataValidation(type="list", formula1='"True,False"', allow_blank=False)
        dv.error = 'Your entry is not valid'
        dv.errorTitle = 'Invalid Entry'
        ws.add_data_validation(dv)
        ran = f'E2:E{len(ids)+1}' if len(ids) > 0 else f'E2:E2'
        dv.ranges.add(ran)
        wb.save(args.verify)

    print(" > Done!")

