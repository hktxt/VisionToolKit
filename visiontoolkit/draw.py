import cv2
import numpy as np
from copy import deepcopy


def draw_bbox(image, bboxes, class_ids=None, class_idx_to_name=None, scores=None,
              box_color=(255, 0, 0), text_color=(255, 255, 255), thickness=2):
    """
    draw bounding box in the image.
    Parameters
    ----------
    image:  the input image, numpy.ndarray (H, W, C).
    bboxes: bounding boxes, list: [[x1, y1, x2, y2], ..., [x1, y1, x2, y2]].
           e.g. [[97, 12, 247, 212], [[90, 17, 242, 219]]].
    class_ids: class id of every bbox, list of int, e.g. [18, 17].
    class_idx_to_name: name of class_id, dict: {17: 'cat', 18: 'dog'},
    scores: score of bbox, list. [0.99, 0.98, 0.57, ..., 0.78], should have same
    length of bboxes.
    box_color: bbox color, tuple: (255, 0, 0), or length of bboxs tuple
    text_color: text color, tuple: (255, 255, 255)
    thickness: line thickness, int, e.g. 2

    Returns
    -------
    Output: image, numpy.ndarray, (H, W, C)

    example:
    pth = 'C:/Users/Emily/Pictures/000000386298.jpg'
    bboxes = [[366.7, 80.84, 499.5, 262.68], [5.66, 138.95, 152.75, 303.83]]  # x1y1x2y2
    class_id = [18, 17]
    class_id_to_name = {17: 'cat', 18: 'dog'}
    img1 = vtk.imread(pth)
    drawed = vtk.draw_bbox(img, bboxes, class_id, class_id_to_name)
    #drawed = vtk.draw_bbox(img, bboxes, class_id)
    #drawed = vtk.draw_bbox(img, bboxes)
    vtk.imshow(drawed)

    """
    img = np.ascontiguousarray(deepcopy(image))
    if class_ids:
        assert len(bboxes) == len(class_ids), 'length error, len(bboxes) should == len(class_id).'
    if scores:
        assert len(bboxes) == len(scores), 'length error, len(bboxes) should == len(scores).'

    n = len(bboxes)
    for i in range(n):
        bbox = bboxes[i]
        x_min, y_min, x_max, y_max = list(map(int, bbox))
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), box_color, thickness)
        string = None
        if class_ids:
            class_name = class_idx_to_name[class_ids[i]] if class_idx_to_name else str(class_ids[i])
            string = class_name
        if scores:
            score = scores[i]
            if string:
                string = string + ': ' + str(score)
            else:
                string = str(score)

        if string:
            ((text_width, text_height), _) = cv2.getTextSize(string, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
            # up
            # cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), box_color, -1)
            # cv2.putText(img, string, (x_min, y_min - int(0.3 * text_height)), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
            # text_color,
            # lineType=cv2.LINE_AA)
            # down
            cv2.rectangle(img, (x_min + 1, y_min), (x_min + 1 + text_width, y_min + int(1.3 * text_height)), box_color,
                          -1)
            cv2.putText(img, string, (x_min + 1, y_min + int(1.2 * text_height)), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                        text_color,
                        lineType=cv2.LINE_AA)

    return img


def draw_mask(img, mask, alpha=0.5):
    """
        draw masks in the image.
        Parameters
        ----------
        img:  the input image, numpy.ndarray (H, W, C);
        mask: the masks, (N, H, W), N is the class number, or (H, W) for single class;
        alpha: float, Alpha of RGB (default: 0.5).

        Returns
        -------
        Output: image, numpy.ndarray, (H, W, C)
        """
    if mask.ndim != 2:
        mask = sum(mask)
    masked = (1 - alpha) * img.astype(float) + alpha * mask.astype(float)
    masked = np.clip(masked.round(), 0, 255).astype(np.uint8)
    return masked


def draw_keypoints():
    pass

