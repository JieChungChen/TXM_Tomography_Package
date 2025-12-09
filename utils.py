import os
import numpy as np
import xml.etree.ElementTree as ET


def from_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    center_ps = []

    for img_tag in root.findall("image"):
        img_name = img_tag.get("name")
        img_name = os.path.basename(img_name)

        box_tag = img_tag.find("box")
        if box_tag is None:
            continue

        xtl, ytl = float(box_tag.get("xtl")), float(box_tag.get("ytl"))
        xbr, ybr = float(box_tag.get("xbr")), float(box_tag.get("ybr"))
        cx = (xtl + xbr) / 2
        cy = (ytl + ybr) / 2
        h = ybr - ytl
        w = xbr - xtl
        center_ps.append([cx, cy, h, w])

    return center_ps


def min_max_normalize(arr, max_percentile=100, min_percentile=0, to_8bit=False):
    """Min-max normalize an array based on given percentiles."""
    max_val = np.percentile(arr, max_percentile)
    min_val = np.percentile(arr, min_percentile)
    normalized = (arr - min_val) / (max_val - min_val)
    normalized = np.clip(normalized, 0, 1)
    if to_8bit:
        normalized = (normalized * 255).astype(np.uint8)
    return normalized