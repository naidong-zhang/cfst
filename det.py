#!/usr/bin/env python3
# coding:utf-8

import numpy as np
import cv2

class Detector(object):
    def __init__(self, tf_configfile='models/opencv_face_detector.pbtxt',
                    tf_modelfile='models/opencv_face_detector_uint8.pb'):
        tf_net = cv2.dnn.readNetFromTensorflow(tf_modelfile, tf_configfile)
        self.net = tf_net

    def detect_faces(self, im, conf_threshold=0.2):
        h, w, _c = im.shape
        blob = cv2.dnn.blobFromImage(im, 1.0, (300, 300), [104, 117, 123], False, False)
        self.net.setInput(blob)
        det = self.net.forward()
        # det_boxes = list()
        # for i in range(det.shape[2]):
        #     confidence = det[0, 0, i, 2]
        #     if confidence > conf_threshold:
        #         x1 = det[0, 0, i, 3] * w
        #         y1 = det[0, 0, i, 4] * h
        #         x2 = det[0, 0, i, 5] * w
        #         y2 = det[0, 0, i, 6] * h
        #         det_boxes.append((x1,y1,x2,y2,confidence))
        # det_boxes = np.array(det_boxes, dtype=np.float32)
        idx = np.where( det[:,:,:,2] > conf_threshold)
        det = det[idx]
        a = np.logical_and(det[:,3] < 1.0, det[:,5] > 0.0)
        b = np.logical_and(det[:,4] < 1.0, det[:,6] > 0.0)
        idx = np.where(np.logical_and(a, b))
        det = det[idx]
        det[:,3::2] *= w
        det[:,4::2] *= h
        det_boxes = np.empty(det[:,2:].shape, dtype=np.float32)
        # round coordinates
        det_boxes[:,:4] = np.round(det[:,3:])
        det_boxes[:,4] = det[:,2]
        # make x0, y0 > 0
        det_boxes[:,:2] = np.maximum(det_boxes[:,:2], 0)
        # det_boxes.shape is (n,5)
        return det_boxes

