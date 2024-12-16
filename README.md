# botell.ai

Model weights can be found [here](https://github.com/AnthroHydro/costiabottles/releases/download/v1.0.0/bottle_weights.pt). Ensure they are placed in the same directory as `bottledetector.py`. See the [wiki](https://github.com/AnthroHydro/costiabottles/wiki) for more detailed installation instructions.

# Command Line

Example command for use as a python command line tool:

`python bottledetector.py "path/to/video.mp4" --show`

For command line tool help, run `python bottledetector.py --help` to display documentation and list of arguments.

# Module

Example code for use as an import:

```python
from bottledetector import BottleDetector

model = BottleDetector()
model.track(path='path/to/video.mp4')
```
For Python module help, after importing BottleDetector as shown above, call `help(BottleDetector)` for class documentation.




