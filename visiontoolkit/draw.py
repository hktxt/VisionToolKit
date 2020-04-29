import cv2


def draw_bbox(img, bboxes, class_ids=None, class_idx_to_name=None,
              box_color=(255, 0, 0), text_color=(255, 255, 255), thickness=2):
    """
    draw bounding box in the image.
    Parameters
    ----------
    img:  the input image, numpy.ndarray (H, W, C).
    bboxes: bounding boxes, list: [[x1, y1, x2, y2], ..., [x1, y1, x2, y2]].
           e.g. [[97, 12, 247, 212], [[90, 17, 242, 219]]].
    class_ids: class id of every bbox, list of int, e.g. [18, 17].
    class_idx_to_name: name of class_id, dict: {17: 'cat', 18: 'dog'}
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
    if class_ids:
        assert len(bboxes) == len(class_ids), 'length error, len(bboxes) should == len(class_id).'

        for bbox, class_id in zip(bboxes, class_ids):
            x_min, y_min, x_max, y_max = list(map(int, bbox))
            class_name = class_idx_to_name[class_id] if class_idx_to_name else str(class_id)
            colr = box_color
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=colr, thickness=thickness)
            ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
            cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), colr, -1)
            cv2.putText(img, class_name, (x_min, y_min - int(0.3 * text_height)), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                        text_color,
                        lineType=cv2.LINE_AA)
    else:
        for bbox in bboxes:
            x_min, y_min, x_max, y_max = list(map(int, bbox))
            colr = box_color
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=colr, thickness=thickness)

    return img


def draw_mask(img, masks, class_ids=None, class_idx_to_name=None, color=None, seed=None):
    """
    draw masks in the image.
    Parameters
    ----------
    img:  the input image, numpy.ndarray (H, W, C).
    masks: the masks, (N, H, W), N is the class number.
    class_ids: class id of every bbox, list of int, e.g. [18, 17].
    class_idx_to_name: name of class_id, dict: {17: 'cat', 18: 'dog'}
    color: the color for drawing, tuple, length of N,
    seed: random seed, if color is not specified seed will be used for generate fixed color.

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
    assert masks.max() == 1, 'masks should be one-hot encoded, larger than 1 should not be accepted.'

