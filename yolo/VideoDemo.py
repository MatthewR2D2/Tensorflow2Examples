import core.utils as utils
from BuildYoloModel import YoloTf2Model
import tensorflow as tf
import cv2
import numpy as np

input_size = 416
depth = 3
yt2 = YoloTf2Model(input_size, input_size, depth)
model = yt2.build_model()

video_path = "data/testimage/t1.mp4"

cap = cv2.VideoCapture(video_path)

while True:
    ret, frame = cap.read()
    if ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    else:
        raise ValueError("No image!")

    frame_size = frame.shape[:2]
    image_data = utils.image_preporcess(np.copy(frame), [input_size, input_size])
    image_data = image_data[np.newaxis, ...].astype(np.float32)
    pred_bbox = model.predict(image_data)
    pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
    pred_bbox = tf.concat(pred_bbox, axis=0)
    bboxes = utils.postprocess_boxes(pred_bbox, frame_size, input_size, 0.90)
    bboxes = utils.nms(bboxes, 0.45, method='nms')
    image = utils.draw_bbox(frame, bboxes)
    result = np.asarray(image)

    cv2.putText(result, text="Test", org=(50, 70), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1, color=(255, 0, 0), thickness=2)
    cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
    result = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imshow("result", result)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()


