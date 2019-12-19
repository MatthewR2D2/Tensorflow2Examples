
class YoloConfiguration:
    CLASSES = "data/classes/coco.names"
    ANCHORS = "data/anchors/basline_anchors.txt"
    STRIDES = [8, 16, 32]
    ANCHOR_PER_SCALE = 3
    IOU_LOSS_THRESH = 0.5
