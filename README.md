# Usage

## import library

```python
from pathlib import Path
from tactus_yolov8.yolov8 import Yolov8
```

## download weights

download the model weights to the specified directory. mode_name must be in the following list:

"yolov8n-pose.pt", "yolov8s-pose.pt", "yolov8m-pose.pt", "yolov8l-pose.pt", "yolov8x-pose.pt", "yolov8x-pose-p6.pt", "yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt", 

```python
Yolov8.download_weights(model_dir: Path, model_name: str)
```

## load model

```python
model = Yolov8(model_dir: Path, model_name: str, device: str, half: bool = False)
```

## extract bounding boxes

```python
model.extract_bboxs(img: Union[Path, np.ndarray])
```

## extract skeletons

```python
model.extract_skeletons(img: Union[Path, np.ndarray])
```
