import cv2
import numpy as np
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils

# Load pre-trained model and labels
PATH_TO_MODEL_DIR = 'saved_model'
PATH_TO_LABELS = 'mscoco_label_map.pbtxt'

detect_fn = tf.saved_model.load(PATH_TO_MODEL_DIR)

category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

def detect_objects(image_np):
    # Convert image to tensor
    input_tensor = tf.convert_to_tensor(image_np)
    input_tensor = input_tensor[tf.newaxis, ...]

    detections = detect_fn(input_tensor)
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # Convert detection classes to ints
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    # Overlay the bounding boxes on the image
    image_np_with_detections = image_np.copy()
    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_detections,
        detections['detection_boxes'],
        detections['detection_classes'],
        detections['detection_scores'],
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=200,
        min_score_thresh=.30,
        agnostic_mode=False)

    return image_np_with_detections

# Live stream from webcam
video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    # Detect objects in the frame
    frame_with_detections = detect_objects(frame)

    # Display the resulting frame with detections
    cv2.imshow('Object Detection', frame_with_detections)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()

# Detect objects in images from a folder
IMAGE_FOLDER_PATH = '/images'

for image_name in os.listdir(IMAGE_FOLDER_PATH):
    image_path = os.path.join(IMAGE_FOLDER_PATH, image_name)
    image_np = cv2.imread(image_path)
    image_np_with_detections = detect_objects(image_np)
    cv2.imshow('Object Detection in Image', image_np_with_detections)
    cv2.waitKey(0)

cv2.destroyAllWindows()
