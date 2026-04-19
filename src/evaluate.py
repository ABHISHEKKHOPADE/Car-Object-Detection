def calculate_iou(box1, box2):
    x1, y1, x1_max, y1_max = box1
    x2, y2, x2_max, y2_max = box2

    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1_max, x2_max)
    yi2 = min(y1_max, y2_max)

    inter_area = max(0, xi2-xi1) * max(0, yi2-yi1)

    box1_area = (x1_max-x1)*(y1_max-y1)
    box2_area = (x2_max-x2)*(y2_max-y2)

    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area