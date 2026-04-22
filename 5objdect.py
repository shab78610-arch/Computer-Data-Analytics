import numpy as np
import os
import tensorflow as tf
import cv2
# Load the pre-trained model
MODEL_NAME = 'ssd_mobilenet_v2_coco_2018_03_29'
PATH_TO_CKPT = os.path.join(MODEL_NAME, 'frozen_inference_graph.pb')
NUM_CLASSES = 90
# Load detection graph
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
# Load label map
PATH_TO_LABELS = r"C:\cv practicals\mscoco_label_map.pbtxt"
category_index = {} 
with open(PATH_TO_LABELS, 'r') as f: 
    lines = f.readlines() 
    for line in lines: 
        if 'id:' in line: 
            id_index = int(line.strip().split(':')[1]) 
        if 'display_name:' in line: 
            name = line.strip().split(':')[1].strip().strip('"') 
            category_index[id_index] = {'name': name} 
# Function to perform object detection
def detect_objects(image):
    with detection_graph.as_default():
        with tf.compat.v1.Session(graph=detection_graph) as sess:
# Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_expanded = np.expand_dims(image, axis=0)
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
# Each box represents a part of the image where a particular object was detected.
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
# Each score represents the level of confidence for each of the objects. 
# The score is shown on the result image, together with the class label.
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
# Actual Detetion
            (boxes, scores, classes, num_detections) = sess.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: image_expanded})
# Visualization of the results of a detection
            for i in range(len(scores[0])):
                if scores[0][i] > 0.5: # confidence threshold as needed
                    class_id = int(classes[0][i])
                    class_name = category_index.get(class_id)['name']
                    score = float(scores[0][i])

                    ymin, xmin, ymax, xmax = boxes[0][i]
                    left = int(xmin * image.shape[1])
                    right = int(xmax * image.shape[1])
                    top = int(ymin * image.shape[0])
                    bottom = int(ymax * image.shape[0])

                    cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
                    cv2.putText(image,'{}: {:.2f}'.format(class_name, score),(left, max(top - 10, 20)),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0, 255, 0),2)
    return image

# Object detection on an image
input_image = cv2.imread(r"C:\cv practicals\IMG.jpeg")
output_image = detect_objects(input_image)
cv2.imshow('Object Detection', output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
