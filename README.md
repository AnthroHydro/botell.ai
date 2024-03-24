# costiabottles

See the [wiki](https://github.com/AnthroHydro/costiabottles/wiki) for installation instructions.

Example code:

```python
from yolo import YOLO

model = YOLO(weights="bestbottle.pt", thresh=0.05)
model.run(path='demo.mp4')
```
