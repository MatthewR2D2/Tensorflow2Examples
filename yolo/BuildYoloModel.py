import core.utils as utils
from core.yolo import YOLOv3, decode
import tensorflow as tf


class YoloTf2Model:
    def __init__(self, H, W, D):
        self.H = H
        self.W = W
        self.D = D

    def build_model(self):
        input_layer = tf.keras.layers.Input([self.W, self.H, self.D])
        feature_maps = YOLOv3(input_layer)

        bbox_tensors = []
        for i, fm in enumerate(feature_maps):
            bbox_tensor = decode(fm, i)
            bbox_tensors.append(bbox_tensor)

        model = tf.keras.Model(input_layer, bbox_tensors)
        utils.load_weights(model, "yoloweights/yolov3.weights")
        #print(model.summary())
        return model
