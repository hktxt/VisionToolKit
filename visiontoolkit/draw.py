import cv2


BOX_COLOR = (255, 0, 0)
TEXT_COLOR = (255, 255, 255)


def draw_bbox(img, bboxes, class_ids=None, class_idx_to_name=None, color=BOX_COLOR, thickness=2):
    """
    draw bounding box in the image.
    Parameters
    ----------
    img:  the input image, numpy.ndarray (H, W, C).
    bboxes: bounding boxes, list: [[x1, y1, x2, y2], ..., [x1, y1, x2, y2]].
           e.g. [[97, 12, 247, 212], [[90, 17, 242, 219]]].
    class_ids: class id of every bbox, list of int, e.g. [18, 17].
    class_idx_to_name: name of class_id, dict: {17: 'cat', 18: 'dog'}
    color: bbox color, tuple: (255, 0, 0), or length of bboxs tuple
    thickness: line thickness, int, e.g. 2

    Returns
    -------
    Output: image, numpy.ndarray, (H, W, C)
    """
    if class_ids:
        assert len(bboxes) == len(class_ids), 'length error, len(bboxes) should == len(class_id).'

        for bbox, class_id in zip(bboxes, class_ids):
            x_min, y_min, x_max, y_max = list(map(int, bbox))
            class_name = class_idx_to_name[class_id]
            colr = color
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=colr, thickness=thickness)
            ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
            cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), BOX_COLOR, -1)
            cv2.putText(img, class_name, (x_min, y_min - int(0.3 * text_height)), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                        TEXT_COLOR,
                        lineType=cv2.LINE_AA)
    else:
        for bbox in bboxes:
            x_min, y_min, x_max, y_max = list(map(int, bbox))
            colr = color
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=colr, thickness=thickness)

    return img