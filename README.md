# costiabottles

See the [wiki](https://github.com/AnthroHydro/costiabottles/wiki) for installation instructions.

Example code:

```
from yolo import YOLO

model = YOLO(weights="bestbottle.pt", thresh=0.05)

#the tracker below is optional, if no tracker is created, the model will use a default one
#tracker = Tracker(distance_function="euclidean", distance_threshold=50, reid_distance_threshold=50, hit_counter_max=30, initialization_delay=15)

model.run(path='vids/demo.mp4')
```
