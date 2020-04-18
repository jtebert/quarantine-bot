"""
Image detection using the Intel RealSense camera and Coral EdgeTPU

This code is based on the examples provided by Google

- Tensorflow Lite: git clone https://github.com/google-coral/tflite
- Edge TPU simple camera examples:
  https://github.com/google-coral/examples-camera/

And a repository where someone used RealSense + edgutpu library (a different
API, it looks like):
- Realsense-Object-Detection-Public:
  https://github.com/samhoff20/Realsense-Object-Detection-Public
"""

import argparse
import os
import time

from PIL import Image, ImageDraw
import numpy as np
import imutils
import pyrealsense2 as rs
import cv2
import tflite_runtime.interpreter as tflite

import utils


class RealSenseCamera:
    def __init__(self, show_debug=False):
        self._pipeline = rs.pipeline()
        self._config = rs.config()

        self._depth_resolution = (640, 480)
        self._color_resolution = (640, 480)
        self._depth_fps = 30
        self._color_fps = 60

        self._config.enable_stream(
            rs.stream.depth,
            self._depth_resolution[0], self._depth_resolution[1],
            rs.format.z16,
            self._depth_fps)
        self._config.enable_stream(
            rs.stream.color,
            self._color_resolution[0],
            self._color_resolution[1],
            rs.format.bgr8,
            self._color_fps
        )

        self._profile = self._pipeline.start(self._config)
        # depth_sensor = self._profile.get_device().first_depth_sensor()
        # depth_scale = self._depth_scale()

        if show_debug:
            print("Depth Input: {} x {} at {} fps".format(
                self._depth_resolution[0],
                self._depth_resolution[1],
                self._depth_fps
            ))
            print("Color Input: {} x {} at {} fps".format(
                self._color_resolution[0],
                self._color_resolution[1],
                self._color_fps
            ))

    def get_frames(self):
        # (blocking?) Get the color and depth frames
        frames = self._pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        return depth_frame, color_frame


def detect(args):
    # Realsense camera feed
    cam = RealSenseCamera(show_debug=args.info)

    # Set up interpreter & labels
    print('Loading {} with {} labels.'.format(args.model, args.labels))
    interpreter = utils.make_interpreter(args.model)
    interpreter.allocate_tensors()
    labels = utils.load_labels(args.labels)

    start_time = time.time()
    x = 1  # displays the frame rate every 1 second
    counter = 0

    while True:
        depth_frame, color_frame = cam.get_frames()
        if not depth_frame or not color_frame:
            continue

        # create numpy array of depth and color frames
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        # resize image based upon argument and create a copy to annotate and display
        color_image = imutils.resize(color_image, width=args.width)
        orig = color_image
        color_image = Image.fromarray(color_image)

        # Do the inference
        start_inference = time.time()

        utils.set_input(interpreter, color_image)
        interpreter.invoke()
        objs = utils.get_output(interpreter,
                                score_threshold=args.threshold,
                                top_k=args.top_k)
        end_inference = time.time()

        # Draw the bounding boxes
        cv2_img = np.array(color_image)
        utils.draw_objects(cv2_img, objs, labels)

        # Draw the image + bounding boxes
        cv2.namedWindow('Real Sense Object Detection', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('Real Sense Object Detection', cv2_img)

        if args.info > 0:
            # create counter for measuring inference time and frames per second
            counter += 1
            if (time.time() - start_time) > x:
                print("FPS: ", counter / (time.time() - start_time))
                fps = []
                fps.append(counter / (time.time() - start_time))
                inference_time = []
                inference_time.append(end_inference - start_inference)
                counter = 0
                start_time = time.time()

        key = cv2.waitKey(1) & 0xFF


if __name__ == '__main__':
    # default_model_dir = '../../models'
    default_model_dir = 'models'
    default_model = 'mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite'
    default_labels = 'coco_labels.txt'

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', help='.tflite model path',
                        default=os.path.join(default_model_dir, default_model))
    parser.add_argument('-l', '--labels', help='label file path',
                        default=os.path.join(default_model_dir, default_labels))
    parser.add_argument('-t', '--threshold', type=float, default=0.5,
                        help='classifier score threshold')
    parser.add_argument('-k', '--top_k', type=int, default=5,
                        help='number of categories with highest score to display')
    parser.add_argument("-w", "--width", type=int, default=640,
                        help="Size of image as it's input into the model")
    parser.add_argument("-i", "--info", type=int, default=0,
                        help="Print debug info to console")

    args = parser.parse_args()

    detect(args)
