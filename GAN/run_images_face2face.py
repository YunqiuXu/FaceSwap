#!/usr/bin/env python

# Modified by Yunqu Xu, run on image folder
# Get combined results
# Run: python run_images_face2face.py

import cv2
import numpy as np
import tensorflow as tf
from glob import glob

# image path
input_path = "test_img/landmark/*.png"
label_path = "test_img/label/*.jpg"
output_path = "test_img/output/"

# label path
model_path_100 = "frozen_model_100.pb"
model_path_200 = "frozen_model_200.pb"
model_path_300 = "frozen_model_300.pb"
model_path_400 = "frozen_model_400.pb"
model_path_500 = "frozen_model_500.pb"

def resize(image):
    CROP_SIZE = 64
    """Crop and resize image for pix2pix."""
    height, width, _ = image.shape
    if height != width:
        # crop to correct ratio
        size = min(height, width)
        oh = (height - size) // 2
        ow = (width - size) // 2
        cropped_image = image[oh:(oh + size), ow:(ow + size)]
        image_resize = cv2.resize(cropped_image, (CROP_SIZE, CROP_SIZE))
        return image_resize


def load_graph(frozen_graph_filename):
    """Load a (frozen) Tensorflow model into memory."""
    graph = tf.Graph()
    with graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(frozen_graph_filename, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    return graph


def main(model_path):

    results = []
    # TensorFlow
    graph = load_graph(model_path)
    image_tensor = graph.get_tensor_by_name('image_tensor:0')
    output_tensor = graph.get_tensor_by_name('generate_output/output:0')
    sess = tf.Session(graph=graph)

    # skip frame
    # count = 0
    for path in sorted(glob(input_path)):

        img_name = path[-7:]
        print "Processing GAN" + img_name

        frame = cv2.imread(path)
        frame_resize = cv2.resize(frame, (64, 64))
        combined_image = np.concatenate([frame_resize, frame_resize], axis=1)
        image_rgb = cv2.cvtColor(combined_image, cv2.COLOR_BGR2RGB)  # OpenCV uses BGR instead of RGB
        generated_image = sess.run(output_tensor, feed_dict={image_tensor: image_rgb})
        image_bgr = cv2.cvtColor(np.squeeze(generated_image), cv2.COLOR_RGB2BGR)
        image_bgr = cv2.resize(image_bgr, (128, 128))
        results.append(image_bgr)

    sess.close()
    return results


if __name__ == '__main__':

    # get landmarks
    landmarks = []
    for path in sorted(glob(input_path)):
        img_name = path[-7:]
        print "Processing landmark: " + img_name
        frame = cv2.imread(path)
        frame = cv2.resize(frame, (128, 128))
        landmarks.append(frame)

    # get results
    results_100 = main(model_path_100)
    results_200 = main(model_path_200)
    results_300 = main(model_path_300)
    results_400 = main(model_path_400)
    results_500 = main(model_path_500)

    # get labels
    labels = []
    for path in sorted(glob(label_path)):
        img_name = path[-7:]
        print "Processing label: " + img_name
        frame = cv2.imread(path)
        frame = cv2.resize(frame, (128, 128))
        labels.append(frame)

    # combine all
    for i in range(9):
        img_result = np.concatenate([landmarks[i], results_100[i], results_200[i], results_300[i], results_400[i], results_500[i], labels[i]], axis = 1)
        cv2.imwrite(output_path + str(i+1) + ".jpg", img_result)


