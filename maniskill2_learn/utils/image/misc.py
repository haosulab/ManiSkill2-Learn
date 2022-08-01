from typing import List, Dict
import numpy as np, cv2
from .photometric import imdenormalize

try:
    import torch
except ImportError:
    torch = None


def tensor2imgs(tensor, mean=(0, 0, 0), std=(1, 1, 1), to_rgb=True):
    """Convert tensor to 3-channel images.
    Args:
        tensor (torch.Tensor): Tensor that contains multiple images, shape (N, C, H, W).
        mean (tuple[float], optional): Mean of images. Defaults to (0, 0, 0).
        std (tuple[float], optional): Standard deviation of images. Defaults to (1, 1, 1).
        to_rgb (bool, optional): Whether the tensor was converted to RGB format in the first place.
            If so, convert it back to BGR. Defaults to True.
    Returns:
        list[np.ndarray]: A list that contains multiple images.
    """

    if torch is None:
        raise RuntimeError("pytorch is not installed")
    assert torch.is_tensor(tensor) and tensor.ndim == 4
    assert len(mean) == 3
    assert len(std) == 3

    num_imgs = tensor.size(0)
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    imgs = []
    for img_id in range(num_imgs):
        img = tensor[img_id, ...].cpu().numpy().transpose(1, 2, 0)
        img = imdenormalize(img, mean, std, to_bgr=to_rgb).astype(np.uint8)
        imgs.append(np.ascontiguousarray(img))
    return imgs


def put_text_on_image(image: np.ndarray, lines: List[str]):
    assert image.dtype == np.uint8, image.dtype
    image = image.copy()

    font_size = 0.5
    font_thickness = 1
    font = cv2.FONT_HERSHEY_SIMPLEX

    y = 0
    for line in lines:
        if isinstance(line, np.ndarray):
            continue
        text_size = cv2.getTextSize(line, font, font_size, font_thickness)[0]
        y += text_size[1] + 10
        x = 10
        cv2.putText(
            image,
            line,
            (x, y),
            font,
            font_size,
            (0, 255, 0),
            font_thickness,
            lineType=cv2.LINE_AA,
        )
    return image


def append_text_to_image(image: np.ndarray, lines: List[str]):
    r"""Appends text left to an image of size (height, width, channels).
    The returned image has white text on a black background.

    Args:
        image: the image to put text
        text: a string to display

    Returns:
        A new image with text inserted left to the input image

    See also:
        habitat.utils.visualization.utils
    """
    # h, w, c = image.shape
    font_size = 0.5
    font_thickness = 1
    font = cv2.FONT_HERSHEY_SIMPLEX
    blank_image = np.zeros(image.shape, dtype=np.uint8)

    y = 0
    for line in lines:
        if isinstance(line, np.ndarray):
            continue
        text_size = cv2.getTextSize(line, font, font_size, font_thickness)[0]
        y += text_size[1] + 10
        x = 10
        cv2.putText(
            blank_image,
            line,
            (x, y),
            font,
            font_size,
            (255, 255, 255),
            font_thickness,
            lineType=cv2.LINE_AA,
        )
    # text_image = blank_image[0 : y + 10, 0:w]
    # final = np.concatenate((image, text_image), axis=0)
    final = np.concatenate((blank_image, image), axis=1)
    return final


def put_info_on_image(image, info: Dict[str, float], extras=None, overlay=True):
    lines = [f"{k}: {v:.3f}" for k, v in info.items() if not isinstance(v, np.ndarray)]
    if extras is not None:
        lines.extend(extras)
    if overlay:
        return put_text_on_image(image, lines)
    else:
        return append_text_to_image(image, lines)
