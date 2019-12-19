import core.utils as utils
from BuildYoloModel import YoloTf2Model
import tensorflow as tf
import cv2
import numpy as np

input_size = 416
depth = 3
yt2 = YoloTf2Model(input_size, input_size, depth)
model = yt2.build_model()

# Test
test_path = 'data/testimage/T1.jpg'
original_image = cv2.imread(test_path)
original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
original_image_size = original_image.shape[:2]

image_data = utils.image_preporcess(np.copy(original_image), [input_size, input_size])
image_data = image_data[np.newaxis, ...].astype(np.float32)

pred_bbox = model.predict(image_data)
pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
pred_bbox = tf.concat(pred_bbox, axis=0)
bboxes = utils.postprocess_boxes(pred_bbox, original_image_size, input_size, 0.3)
bboxes = utils.nms(bboxes, 0.45, method='nms')

image = utils.draw_bbox(original_image, bboxes)
#image = Image.fromarray(image)
cv2.imshow("Test", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

