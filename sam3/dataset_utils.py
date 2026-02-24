import xml.etree.ElementTree as ET
from xml.dom import minidom
import numpy as np
import cv2
import torch


def save_dataset_to_cvat_xml(
    dataset_items: list, 
    output_path: str, 
    label_map: dict = None, 
    score_threshold: float = 0.5,
    approx_epsilon: float = 2.0
):
    """
    Saves annotations to CVAT XML 1.1 with Grouping (associating Box + Polygon).
    """
    
    root = ET.Element("annotations")
    ET.SubElement(root, "version").text = "1.1"

    def to_numpy(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.array(x)

    for img_id, item in enumerate(dataset_items):
        filename = item['filename']
        ann = item['annotations']
        
        h = int(ann['original_height'])
        w = int(ann['original_width'])
        
        image_node = ET.SubElement(root, "image", {
            "id": str(img_id), 
            "name": filename,
            "width": str(w),
            "height": str(h)
        })

        boxes = to_numpy(ann['boxes'])
        scores = to_numpy(ann['scores'])
        
        has_masks = 'masks' in ann
        if has_masks:
            masks = to_numpy(ann['masks'])
            if masks.ndim == 4: masks = masks.squeeze(1)

        # Iterate through detections
        for i, score in enumerate(scores):
            if score < score_threshold:
                continue

            # 1. GENERATE UNIQUE GROUP ID
            # We use a unique ID for this specific object instance within the image.
            # (i + 1) ensures it's a positive integer distinct from 0.
            group_id = str(i + 1)

            class_id = int(ann['labels'][i]) if 'labels' in ann else 0
            label_name = label_map[class_id] if label_map and class_id in label_map else "object"

            # 2. ADD BOX (with group_id)
            x1, y1, x2, y2 = boxes[i]
            ET.SubElement(image_node, "box", {
                "label": label_name,
                "group_id": group_id,  # <--- LINKING KEY
                "xtl": f"{x1:.2f}",
                "ytl": f"{y1:.2f}",
                "xbr": f"{x2:.2f}",
                "ybr": f"{y2:.2f}",
                "z_order": "0",
                "occluded": "0",
                "source": "manual"
            })

            # 3. ADD POLYGON (with same group_id)
            if has_masks:
                mask = masks[i]
                mask_uint8 = (mask > 0).astype(np.uint8) * 255
                contours = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
                
                for contour in contours:
                    if cv2.contourArea(contour) < 2048: continue
                    contour = cv2.approxPolyDP(contour, approx_epsilon, True)

                    points_str = ";".join([f"{pt[0][0]:.2f},{pt[0][1]:.2f}" for pt in contour])
                    
                    ET.SubElement(image_node, "polygon", {
                        "label": label_name,
                        "group_id": group_id,  # <--- SAME LINKING KEY
                        "points": points_str,
                        "z_order": "0",
                        "occluded": "0",
                        "source": "manual"
                    })

    xml_str = minidom.parseString(ET.tostring(root)).toprettyxml(indent="  ")
    with open(output_path, "w") as f:
        f.write(xml_str)
    
    print(f"Dataset saved to {output_path}")
