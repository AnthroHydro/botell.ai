# Usage

## Command Line

### Documentation

usage: 
  `python bottledetector.py [--help] [--show] [--numbottles] [--framerate] [--mintime] [--minframedist] [--conf] [--output] [--verify] path`

positional arguments:  
  `path`                    the file path to the video that is to be processed

optional arguments:  
  `-h`, `--help`            show this help message and exit  
  `-s`, `--show`            if this is specified, the model will show the video as it is processed  
  `-n`, `--numbottles` the number of bottles that appear in the video; if this is specified, error will be calculated  
  `-f`, `--framerate` the frames per second at which the video should be processed  
  `-t`, `--mintime` the minimum number of seconds a bottle must be on screen for it to be counted  
  `-d`, `--minframedist` a float between 0.0 and 1.0 representing the fraction of the screen the bottle must travel horizontally to be counted  
  `-c`, `--conf` confidence threshold the model should use  
  `-o`, `--output` the name of the .txt the tool should write metrics to  
  `-v`, `--verify` filename with the .xlsx extension to be created for verification (a .avi file with the same name will be saved as well)  
  
### Example

`python bottledetector.py "path/to/video.mp4" -s -f 15 -c 0.2` 

This will process the video at 15FPS (`-f 15`, shorthand for `--framerate 15`) and a confidence threshold of 0.2 (`-c 0.2`, shorthand for `--conf 0.2`) while showing the output as it processes (`-s`, shorthand for `--show`)

## Module

### Documentation

   ```BottleDetector(weights='bottle_weights.pt')```  
     
     
   ```__init__(self, weights='bottle_weights.pt')```  
           &nbsp;&nbsp;&nbsp;&nbsp;**Description**:  
           &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;initializes a new YOLO model  
           &nbsp;&nbsp;&nbsp;&nbsp;**Parameters**:  
           &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`weights` (`String`)    : path to the .pt file containing the weights of the model  
           &nbsp;&nbsp;&nbsp;&nbsp;**Returns**:  
           &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`BottleDetector` class object  
     
   ```predict_frame(self, frame, thresh)```  
           &nbsp;&nbsp;&nbsp;&nbsp;**Description**:  
           &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;runs the input frame through an inference pass on the class's model  
           &nbsp;&nbsp;&nbsp;&nbsp;**Parameters**:  
           &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`frame` (`String`)      : video frame to be processed and tracked  
           &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`thresh` (`float`)      : confidence threshold the model should use  
           &nbsp;&nbsp;&nbsp;&nbsp;**Returns**:  
           &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`List[Detections]`      : list of norfair Detection objects  
     
   ```track(self, path='demo.mp4', skip_frames=3, tracker=None, actual_num_bottles=None, min_time=1.0, min_frame_dist=0.3, show=False, thresh=0.05, save=None)```  
           &nbsp;&nbsp;&nbsp;&nbsp;**Description**:  
           &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;This method gathers data from the video, running the tracking algorithm to detect<br> 
           &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;bottles. Post-processing is performed so as to filter out false positives, making it<br>
           &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;easier to lower detection thresholds in the case of false negatives.  
           &nbsp;&nbsp;&nbsp;&nbsp;**Parameters**:  
           &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`path` (`String`)               : the video to test the model on, associated with timestamps  
           &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`skip_frames` (`int`)           : number of frames to be skipped before reading the next  
           &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`tracker` (`Tracker`)           : a norfair `Tracker` object to be used for object tracking  
           &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`actual_num_bottles` (`int`)    : the number of bottles that actually appear in the video  
           &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`min_time` (`float`)            : number of seconds a bottle must be on screen for it to be counted  
           &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`min_frame_dist` (`float`)      : horizontal fraction of the frame the bottle must travel to be counted  
           &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`show` (`bool`)                 : whether or not the video should be displayed as tracking is run  
           &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`thresh` (`float`)              : confidence threshold the model should use  
           &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`save` (`String`)               : either `None` or the name under which the annotated video should be saved  
           &nbsp;&nbsp;&nbsp;&nbsp;**Returns**:
           &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`int`                         : number of bottles that were successfully tracked and counted in the video  
           &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`float`                       : the accuracy of the model on the input video  
           &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`tuple[int, int, Dict]`       : metrics and a dictionary of the bottles tracked in the video  

### Example

```
from bottledetector import BottleDetector

model = BottleDetector()
model.track(path='path/to/video.mp4')
```
