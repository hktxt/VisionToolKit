from .utils import img_b64_to_arr
from .base import readJson


def viz_labelme(json_pth):
    data = readJson(json_pth)
    imageData = data.get('imageData')
    img = img_b64_to_arr(imageData)


def viz_labelImg(json_pth):
    pass