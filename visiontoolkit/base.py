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


def imshow(image, title=None, axis='on'):
    plt.imshow(image)
    if title:
        plt.title(title)
    if axis == 'off':
        plt.xticks([])
        plt.yticks([])
    plt.show()