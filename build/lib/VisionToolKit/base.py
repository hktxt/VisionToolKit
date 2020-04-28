import cv2
import matplotlib.pyplot as plt


def imread(img_pth):
    """read image from path.
    Parameters
    ----------
    img_pth: string, the image path.
    Returns
    -------
    Output: numpy.ndarray, (H, W, C)
    """
    image = cv2.imread(img_pth)

    return image


def imshow(image, ratio=1, title=None, axis='on'):
    assert ratio >= 0, 'ratio should larger than 1.'
    image = cv2.resize(image, image.shape[0]*ratio, image.shape[1]*ratio)
    plt.imshow(image)
    if title:
        plt.title(title)
    if axis == 'off':
        plt.xticks([])
        plt.yticks([])
    plt.show()