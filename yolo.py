from sahi import AutoDetectionModel
from sahi.prediction import PredictionResult
from sahi.predict import get_prediction, get_sliced_prediction
from norfair import Detection, Tracker, Video, draw_boxes, draw_tracked_boxes
from torch import cuda
from typing import List
import numpy as np
import cv2

class YOLO:

    def __init__(self, version='yolov8', weights='best.pt', thresh=0.2, use_gpu=True):
        """
        Description:
            initializes a new YOLO model
        Parameters:
            version (String)    : the model to be used as the predictor (e.g. 'yolov5', 'yolov8')
            weights (String)    : path to the .pt file containing the weights of the model
            use_gpu (int)       : 1 if a cuda-supported GPU is available, 0 otherwise
        Returns:
            None
        """
        device = 'cpu'
        if use_gpu and cuda.is_available():
            device = 'cuda:0'
        self.version=version
        self.model = AutoDetectionModel.from_pretrained(
            model_type=version,
            model_path=weights,
            confidence_threshold=thresh,
            device=device)
            
            
    def predict_frame(self, frame, slice=False):
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
        if slice:
            results = get_sliced_prediction(
                frame,
                self.model,
                slice_height=int(frame.shape[1] / 2),
                slice_width=int(frame.shape[1] / 2),
                overlap_height_ratio=0.2,
                overlap_width_ratio=0.2
            )
        else:
            results = get_prediction(frame, self.model)

        return self._get_detections(results.object_prediction_list)

    def run(self, tracker = None, path = "demo.mp4"):
        video = Video(input_path=path)
        if tracker is None:
            tracker = Tracker(distance_function="euclidean", distance_threshold=100)
        for i, frame in enumerate(video):
            frame = cv2.resize(frame, (640, 360))
            norfair_detections = self.predict_frame(frame)
            tracked_objects = tracker.update(detections=norfair_detections)
            draw_boxes(frame, tracked_objects)
            cv2.imshow("VID", frame)
            cv2.waitKey(1)

        
    def train(self):
        """
        Description:
            Trains the model given a .yaml data file
        Parameters:
            TBD
        Returns:
            TBD
        """
        return
        
    def statistics(self, timestamps = "timestamps.txt", path = "demo.mp4", interval = 1, tracker = None):
        """
        Description:
            TBD
        Parameters:
            timestamps (String) : txt file with timestamps for when each class is present
            path (String)       : the video to test the model on, associated with timestamps
            interval (int)      : the framerate at which the video should be read
        Returns:
            float               : the accuracy of the model on the input video
        """
        #init video
        video = Video(input_path=path)
        fps = int(video.output_fps)
        
        #init data storage for reading into
        predicted_labels = []
                    
        #read norfair tracking detections into list
        if tracker is None:
            tracker = Tracker(distance_function="euclidean", distance_threshold=100)
        for i, frame in enumerate(video):
            frame = cv2.resize(frame, (640, 360))
            nf_detections = self.predict_frame(frame)
            tracker.update(detections = nf_detections)
            predicted_labels.append([detection.label for detection in nf_detections])


            
        #read timestamps into a list where each index is a frame of the video
        labels = [[] for _ in range(len(predicted_labels))]
        N = 0
        with open(timestamps, 'r') as txt:
            for line in txt.readlines():
                start, end, label = line.split(",")
                starti = interval * ((int(start.split(":")[0]) * 60) + int(start.split(":")[1]))
                endi = interval * ((int(end.split(":")[0]) * 60) + int(end.split(":")[1]))
                for i in range(starti, endi):
                    labels[i].append(label)
                N += 1
        print("num_actual:", N)
        print("num_tracked:", tracker.total_object_count)
        return (N - tracker.total_object_count) / N
        
        
    #adapted from https://github.com/tryolabs/norfair/blob/master/demos/sahi/src/demo.py
    def _get_detections(self, object_prediction_list: PredictionResult) -> List[Detection]:
        detections = []
        for prediction in object_prediction_list:
            bbox = prediction.bbox

            detection_as_xyxy = bbox.to_voc_bbox()
            bbox = np.array(
                [
                    [detection_as_xyxy[0], detection_as_xyxy[1]],
                    [detection_as_xyxy[2], detection_as_xyxy[3]],
                ]
            )
            detections.append(
                Detection(
                    points=bbox,
                    scores=np.array([prediction.score.value for _ in bbox]),
                    label=prediction.category.id,
                )
            )
        return detections
         
