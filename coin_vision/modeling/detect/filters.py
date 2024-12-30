def filter_duplicates(results, iou_threshold=0.5):
    if not results:
        return []

    filtered_boxes = []
    for box in results[0].boxes.xyxy:
        xmin, ymin, xmax, ymax = map(int, box)

        duplicate = False
        for existing_box in filtered_boxes:
            if _calculate_iou((xmin, ymin, xmax, ymax), existing_box) > iou_threshold:
                print('removing duplicate')
                duplicate = True
                break

        if not duplicate:
            filtered_boxes.append((xmin, ymin, xmax, ymax))

    return filtered_boxes

def _calculate_iou(box1, box2):
    # Calculate the intersection over union (IoU) for two bounding boxes
    x1, y1, x2, y2 = box1
    x1_, y1_, x2_, y2_ = box2

    # Calculate the intersection area
    inter_x1 = max(x1, x1_)
    inter_y1 = max(y1, y1_)
    inter_x2 = min(x2, x2_)
    inter_y2 = min(y2, y2_)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

    # Calculate the area of both bounding boxes
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_ - x1_) * (y2_ - y1_)

    # Calculate the IoU
    iou = inter_area / float(box1_area + box2_area - inter_area)
    return iou

def filter_round_or_oval_detections(filtered_boxes, aspect_ratio_range=(0.8, 1.2)):
    filtered_round_boxes = []
    min_ratio, max_ratio = aspect_ratio_range

    for box in filtered_boxes:
        x1, y1, x2, y2 = box  # Unpack the coordinates
        width = x2 - x1
        height = y2 - y1
        aspect_ratio = width / height

        # Check if the aspect ratio falls within the desired range
        if min_ratio <= aspect_ratio <= max_ratio:
            filtered_round_boxes.append((x1, y1, x2, y2))

    return filtered_round_boxes

