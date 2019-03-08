import os
import sys
import time
import tensorflow as tf
import numpy as np
import cv2
import tensorflow.contrib.tensorrt as trt
from tf_trt_models.detection import download_detection_model
from tf_trt_models.detection import build_detection_graph
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


# WIDTH = 1920
# HEIGHT = 1080
# WIDTH = 640
# HEIGHT = 480
WIDTH = int(sys.argv[3])
HEIGHT = int(sys.argv[4])

# Define the video stream
cap = cv2.VideoCapture(sys.argv[1])

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(sys.argv[2], fourcc, 12.0, (WIDTH, HEIGHT))

# What model to download.
MODEL = 'ssdlite_mobilenet_v2_coco'
DATA_DIR = './data/'
CONFIG_FILE = MODEL + '.config'
CHECKPOINT_FILE = 'model.ckpt'

if int(sys.argv[5]) == 0:
    PATH_TO_LABELS = '../third_party/models/research/object_detection/data/' +\
        'mscoco_label_map.pbtxt'
    OPTIMIZED_MODEL_FILE = 'optimized_model.pbtxt'
    NUM_CLASSES = 90
else:
    PATH_TO_LABELS = 'object_detection.pbtxt'
    OPTIMIZED_MODEL_FILE = 'optimized_model_2.pbtxt'
    NUM_CLASSES = 1
if not os.path.exists(os.path.join(DATA_DIR, OPTIMIZED_MODEL_FILE)):
    print('Creating optimized graph...')
    # Download model and build frozen graph
    config_path, checkpoint_path = download_detection_model(MODEL, 'data')
    frozen_graph, input_names, output_names = build_detection_graph(
        config=config_path,
        checkpoint=checkpoint_path,
        score_threshold=0.3,
        batch_size=1
    )
    # Optimize with TensorRT
    trt_graph = trt.create_inference_graph(
        input_graph_def=frozen_graph,
        outputs=output_names,
        max_batch_size=1,
        max_workspace_size_bytes=1 << 25,
        precision_mode='FP16',
        minimum_segment_size=50
    )
    with tf.gfile.GFile(
            os.path.join(DATA_DIR, OPTIMIZED_MODEL_FILE), 'wb') as f:
        f.write(trt_graph.SerializeToString())
else:
    print('Loaded optimized graph')
    trt_graph = tf.GraphDef()
    with open(os.path.join(DATA_DIR, OPTIMIZED_MODEL_FILE), 'rb') as f:
        trt_graph.ParseFromString(f.read())

# Loading label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(
    label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Create session and load graph
tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True

tf_sess = tf.Session(config=tf_config)

tf.import_graph_def(trt_graph, name='')
tf_input = tf_sess.graph.get_tensor_by_name('image_tensor:0')
tf_scores = tf_sess.graph.get_tensor_by_name('detection_scores:0')
tf_boxes = tf_sess.graph.get_tensor_by_name('detection_boxes:0')
tf_classes = tf_sess.graph.get_tensor_by_name('detection_classes:0')
tf_num_detections = tf_sess.graph.get_tensor_by_name('num_detections:0')

total_time = 0
num_iterations = 0

while True:
    start_time = time.time()
    # Read frame from camera
    ret, image_np = cap.read()
    if not ret:
        break
    image_np = cv2.resize(image_np, (WIDTH, HEIGHT))
    image_np_expanded = np.expand_dims(image_np, axis=0)
    (boxes, scores, classes, num_detections) = tf_sess.run(
        [tf_boxes, tf_scores, tf_classes, tf_num_detections],
        feed_dict={tf_input: image_np_expanded})
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=8)
    # Calculate FPS
    elapsed_time = time.time() - start_time
    cv2.putText(image_np, '%.3g fps' % (1 / elapsed_time),
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2)
    # Display output
    out.write(image_np)

    total_time += elapsed_time
    num_iterations += 1

cap.release()
out.release()

print('average fps: {}'.format(num_iterations / total_time))
