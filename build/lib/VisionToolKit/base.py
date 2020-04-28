import cv2
import matplotlib.pyplot as plt


def imread(img_pth, c='rgb'):
    """read image from path.
    Parameters
    ----------
    img_pth: string, the image path,
    c: channel,
    Returns
    -------
    Output: numpy.ndarray, (H, W, C)
    """
    assert c in ['rgb', 'bgr'], 'c = rgb or bgr?'
    image = cv2.imread(img_pth)
    if c == 'rgb':
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def imsave(img, pth):
    cv2.imwrite(pth, img)


def imshow(image, title=None, axis='on'):
    plt.imshow(image)
    if title:
        plt.title(title)
    if axis == 'off':
        plt.xticks([])
        plt.yticks([])
    plt.show()