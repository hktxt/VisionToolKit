import numpy as np
import base64
import io
import PIL.Image


def padd2square(img, value=0):
    """
    padd image to square and keep ratio.
    :param img: numpy.array image, [h,w,c],
    :param value: padd value,
    :return: padded image.
    """
    h, w, c = img.shape
    if h >= w:
        squareImg = np.ones((h, h, c), dtype=img.dtype) * value
        p = int((h-w)/2)
        squareImg[:, p:p+w, :] = img
    else:
        squareImg = np.ones((w, w, c), dtype=img.dtype) * value
        p = int((w-h)/2)
        squareImg[p:p+h, :, :] = img
    return squareImg


# modified form: https://github.com/ultralytics/yolov3/blob/master/utils/utils.py
def xyxy2xywh(x):
    # Transform box coordinates from [x1, y1, x2, y2] (where xy1=top-left, xy2=bottom-right) to [x, y, w, h]
    y = np.zeros_like(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


# modified form: https://github.com/ultralytics/yolov3/blob/master/utils/utils.py
def xywh2xyxy(x):
    # Transform box coordinates from [x, y, w, h] to [x1, y1, x2, y2] (where xy1=top-left, xy2=bottom-right)
    y = np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def rle2mask(rle, imgshape):
    # mask = rle2mask(train['EncodedPixels'].iloc[i], img.shape)
    width = imgshape[0]
    height = imgshape[1]

    mask = np.zeros(width * height).astype(np.uint8)

    array = np.asarray([int(x) for x in rle.split()])
    starts = array[0::2]
    lengths = array[1::2]

    current_position = 0
    for index, start in enumerate(starts):
        mask[int(start):int(start + lengths[index])] = 1
        current_position += lengths[index]

    return np.flipud(np.rot90(mask.reshape(height, width), k=1))


def mask2rle(mask):
    pass


def IOU():
    pass


def ap():
    pass


def img_data_to_arr(img_data):
    f = io.BytesIO()
    f.write(img_data)
    img_arr = np.array(PIL.Image.open(f))
    return img_arr


def img_b64_to_arr(img_b64):
    img_data = base64.b64decode(img_b64)
    img_arr = img_data_to_arr(img_data)
    return img_arr