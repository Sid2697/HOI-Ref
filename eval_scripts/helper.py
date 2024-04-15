import re

def multiple_boxes_iou(gt_boxes, pred_boxes):
    """
    Calculate the IoU score between multiple ground truth and predicted bounding boxes.

    Parameters:
    - gt_boxes: List of ground truth bounding boxes.
    - pred_boxes: List of predicted bounding boxes.

    Returns:
    - List of IoU scores between each ground truth and predicted bounding box pair.
    """
    iou_scores = []
    for gt_box in gt_boxes:
        for pred_box in pred_boxes:
            pred_box = bb_normalize(pred_box, image_original_size=(100, 100), image_new_size=(1920, 1080))
            iou_scores.append(calculate_iou(gt_box, pred_box))
    return iou_scores


def bb_normalize(
        bbox,
        image_original_size=(1440, 1080),
        image_new_size=(100, 100),
):
    """
    This function normalises the bounding box coordinates to the new image size.
    It is based on the minigpt implementation. Where the authors hardcode the
    new image size to 100x100.
    For Ego4D dataset, the image size is 1440x1080.
    For EPIC-KITCHENS-100 dataset, the image size is 456x256. This is because of
    the resizing done when generating the HOI information.
    For VISOR dataset, the image size is 1920x1080.
    """
    if type(bbox[0]) is str:
        bbox = [float(x) for x in bbox]
    normalised_bbox = [
        bbox[0] / image_original_size[0] * image_new_size[0],
        bbox[1] / image_original_size[1] * image_new_size[1],
        bbox[2] / image_original_size[0] * image_new_size[0],
        bbox[3] / image_original_size[1] * image_new_size[1]
    ]
    return [int(x) for x in normalised_bbox]


def extract_bounding_boxes(sentence):
    pattern = r'{<(-?\d+(?:\.\d+)?)><(-?\d+(?:\.\d+)?)><(-?\d+(?:\.\d+)?)><(-?\d+(?:\.\d+)?)>}'
    matches = re.finditer(pattern, sentence)

    bounding_boxes = []
    for match in matches:
        bounding_box = [float(match.group(i)) for i in range(1, 5)]
        bounding_boxes.append(bounding_box)

    return bounding_boxes


def calculate_iou(box1, box2):
    """
    Calculate the Intersection over Union (IoU) score for two bounding boxes.

    Parameters:
    - box1: Tuple (x1, y1, x2, y2) representing the coordinates of the first bounding box.
    - box2: Tuple (x1, y1, x2, y2) representing the coordinates of the second bounding box.

    Returns:
    - IoU score (float) between 0.0 and 1.0.
    """
    # Calculate the coordinates of the intersection rectangle
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    # Calculate the area of intersection
    intersection_area = max(0, x2_inter - x1_inter + 1) * max(0, y2_inter - y1_inter + 1)

    # Calculate the area of union
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
    union_area = box1_area + box2_area - intersection_area

    # Calculate IoU score
    iou_score = intersection_area / union_area

    return iou_score
