from typing import Union, Tuple, Dict, List
from pathlib import Path

import numpy as np
import torch
import cv2

from ultralytics.nn.autobackend import AutoBackend
from ultralytics.yolo.utils import downloads
from ultralytics.yolo.utils.checks import check_imgsz
from ultralytics.yolo.utils.torch_utils import select_device
from ultralytics.yolo.utils.ops import non_max_suppression, scale_boxes, scale_coords
from ultralytics.yolo.data.augment import LetterBox


class Yolov8:
    def __init__(self, model_dir: Path, model_name: str, device: str, half: bool = False) -> None:
        """
        instanciate the model.

        Parameters
        ----------
        model_dir : Path
            directory where to find the model weights.
        model_name : str
            name of the model.
        device : str
            the device name to run the model on.
        half : bool, optional
            use half precision (float16) forthe model, by default False
        """
        self.device = select_device(device)
        self.half = half
        self.model_name = model_name
        self.model = AutoBackend(model_dir / model_name, device=self.device, fp16=half)

    @classmethod
    def download_weights(cls, model_dir: Path, model_name: str):
        """
        download weights for a given model name to a specified directory.

        Parameters
        ----------
        model_dir : Path
            directory to save the model weights.
        model_name : str
            name of the model weights to download.
        """
        if model_name not in model_name_to_url:
            raise ValueError("invalid model name. Must be in ", ", ".join(model_name_to_url.keys()))

        url = model_name_to_url[model_name]
        downloads.safe_download(url, dir=model_dir)

    def _preprocess_img(self, img: Union[Path, np.ndarray]) -> Tuple[torch.Tensor, Tuple, Tuple]:
        """
        pad and transform an image to a tensor. Also compute the origin
        image size and the new size.

        Parameters
        ----------
        img : Union[Path, np.ndarray]
            image Path or numpy array reprenting the image.

        Returns
        -------
        Tuple[torch.Tensor, Tuple, Tuple]
            (
                tensor of shape [n_samples, 3, img_height, img_width],
                (original image height, original image width),
                (new image height, new image width),
            )
        """
        img = load_image(img)
        img0_size = img.shape[:2]
        img = correct_img_size(img, self.model.stride)
        img_size = img.shape[1:]
        imgs = img_to_tensor(img, self.device, self.half)

        return imgs, img0_size, img_size

    def _predict(self, images: torch.Tensor) -> List[torch.Tensor]:
        """
        predict on a tensor containing single or multiple images.

        Parameters
        ----------
        images : torch.Tensor
            tensor of shape [n_samples, 3, img_height, img_width]

        Returns
        -------
        List[torch.Tensor]
            result tensor for each image.
        """
        return self.model(images)

    @torch.no_grad()
    def extract_bboxs(self, img: Union[Path, np.ndarray]):
        """
        extract the bounding boxes from a given image.

        Parameters
        ----------
        img : Union[Path, np.ndarray]
            path to an image, or a numpy array representing the image.

        Returns
        -------
        Dict[str, List]
            return a dictionnary with
            - `bboxes` list containing every (x1, y1, x2, y2) coordinates.
            - `scores` list the score for each bounding box.
        """
        if "pose" in self.model_name:
            raise ValueError("wrong model selected. You can't predict bounding boxes"
                             "with a pose-prediction model.")

        _img, img0_size, img_size = self._preprocess_img(img)
        preds = self._predict(_img)
        preds = non_max_suppression(preds, conf_thres=0.25, iou_thres=0.45, classes=[0], max_det=300)

        results_bboxs = []
        results_scores = []
        for pred in preds:
            # scale the prediction back to the input image size
            pred[:, :4] = scale_boxes(img_size, pred[:, :4], img0_size).round()
            results_bboxs.extend(pred[:, :4].tolist())
            results_scores.extend(pred[:, 4].tolist())

        return {"bboxes": results_bboxs, "scores": results_scores}

    @torch.no_grad()
    def extract_skeletons(self, img: Union[Path, np.ndarray]) -> Dict[str, List]:
        """
        extract the skeletons from a given image.

        Parameters
        ----------
        img : Union[Path, np.ndarray]
            path to an image, or a numpy array representing the image.

        Returns
        -------
        Dict[str, List]
            return a dictionnary with
            - `bboxes` list containing every (x1, y1, x2, y2) coordinates.
            - `scores` list the score for each bounding box.
            - `keypoints` list containing (x1, y1, is_visible) coordinates.
        """
        if "pose" not in self.model_name:
            raise ValueError("wrong model selected. You can't predict poses"
                             "without a pose-prediction model.")

        _img, img0_size, img_size = self._preprocess_img(img)
        preds = self._predict(_img)
        preds = non_max_suppression(preds, conf_thres=0.25, iou_thres=0.7, classes=None, max_det=1000, nc=1)

        results_bboxs = []
        results_scores = []
        results_keypoints = []
        for pred in preds:
            if pred == []:
                continue
            # scale the prediction back to the input image size
            pred[:, :4] = scale_boxes(img_size, pred[:, :4], img0_size).round()
            pred[:, 6:] = scale_coords(img_size, pred[:, 6:], img0_size).round()

            results_bboxs.extend(pred[:, :4].tolist())
            results_scores.append(pred[:, 4].tolist())
            results_keypoints.extend(pred[:, 6:].tolist())

        return {"bboxes": results_bboxs, "scores": results_scores, "keypoints": results_keypoints}


def correct_img_size(img: np.ndarray, stride: int) -> np.ndarray:
    """
    pad the image if it does not have the correct size.

    Parameters
    ----------
    img : np.ndarray
        numpy array representing the image.
    stride : int
        stride to apply.

    Returns
    -------
    np.ndarray
        new resized image.
    """
    img_size = img.shape[:2]
    target_img_size = check_imgsz(img_size, stride)

    img = LetterBox(target_img_size, auto=True, stride=stride)(image=img)
    img = img.transpose((2, 0, 1))[::-1]
    img = np.ascontiguousarray(img)

    return img


def img_to_tensor(img: np.ndarray, device: torch.device, half: bool) -> torch.Tensor:
    """
    convert a numpy image to a tensor

    Parameters
    ----------
    img : np.ndarray
         numpy array representing the image.
    device : torch.device
        the torch device to store the tensor on.
    half : bool
        whether or not to use half precision.

    Returns
    -------
    torch.Tensor
        tensor of shape [n_samples, 3, img_height, img_width] storing the images.
    """
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if len(img.shape) == 3:
        img = img[None]  # expand for batch dim

    return img


def load_image(image: Union[Path, np.ndarray]) -> np.ndarray:
    """
    load the image from a Path, or do nothing if it is a numpy array.

    Parameters
    ----------
    image : Union[Path, np.ndarray]
        image Path or numpy array reprenting the image.

    Returns
    -------
    np.ndarray
        numpy array representing the image.
    """
    if isinstance(image, Path):
        return cv2.imread(str(image))

    return image


model_name_to_url = {
    "yolov8n-pose.pt": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-pose.pt",
    "yolov8s-pose.pt": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s-pose.pt",
    "yolov8m-pose.pt": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m-pose.pt",
    "yolov8l-pose.pt": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l-pose.pt",
    "yolov8x-pose.pt": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x-pose.pt",
    "yolov8x-pose-p6.pt": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x-pose-p6.pt",
    "yolov8n.pt": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt",
    "yolov8s.pt": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt",
    "yolov8m.pt": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt",
    "yolov8l.pt": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l.pt",
    "yolov8x.pt": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt",
}
