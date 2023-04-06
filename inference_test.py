import glob
import random
from io import BytesIO
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image

from models.research.object_detection.utils import ops
from models.research.object_detection.utils import visualization_utils as viz
from models.research.object_detection.utils.label_map_util import create_category_index_from_labelmap

LABELS_PATH = './annotations/label_map.pbtxt'
MODEL_PATH = './inference_graph/saved_model'

CATEGORY_IDX = create_category_index_from_labelmap(LABELS_PATH, use_display_name=True)

def load_image(path):
    image_data = tf.io.gfile.GFile(path, 'rb').read()
    image = Image.open(BytesIO(image_data))
    width, height = image.size
    shape = (height, width, 3)
    image = np.array(image.getdata())
    image = image.reshape(shape).astype('uint8')

    return image


def run_inference(net, image):
    image = np.asarray(image)
    
    input_tensor = tf.convert_to_tensor(image)
    input_tensor = input_tensor[tf.newaxis, ...]

    # forward pass
    model = net.signatures['serving_default']
    result = model(input_tensor)

    # extract detections
    num_detections = int(result.pop('num_detections'))
    result = { key: value[0, :num_detections].numpy() for key, value in result.items() }
    result['num_detections'] = num_detections
    result['detection_classes'] = result['detection_classes'].astype('int64')

    # use mask if available
    if 'detection_masks' in result:
        detection_masks_reframed = ops.reframe_box_masks_to_image_masks(result['detection_masks'], result['detection_boxes'], image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5, tf.uint8)
        result['detection_masks_reframed'] = detection_masks_reframed.numpy()
    
    return result


def get_image_with_boxes(model, path):
    image = load_image(path)
    annotation = run_inference(model, image)
    masks = annotation.get('detection_masks_reframed', None)
    viz.visualize_boxes_and_labels_on_image_array(
        image,
        annotation['detection_boxes'],
        annotation['detection_classes'],
        annotation['detection_scores'],
        CATEGORY_IDX,
        instance_masks=masks,
        use_normalized_coordinates=True,
        line_thickness=5)
    
    return image


if __name__ == '__main__':
    model = tf.saved_model.load(MODEL_PATH)

    image_paths = list(glob.glob('images/test/*.jpeg'))
    image_paths = random.choices(image_paths, k=6)
    images = [get_image_with_boxes(model, path) for path in image_paths]

    for index, image in enumerate(images):
        row, col = int(index / 3), index % 3
        plt.imshow(image)
        plt.axis('off')
        plt.savefig('test_out_' + str(index + 1) + '.jpeg', dpi=1500)
