#!/usr/bin/env python3
# coding:utf-8

import numpy as np
# from scipy.special import expit
from skimage.feature import local_binary_pattern
import cv2
import click

from det import Detector
from extractor import cal_face_patch, cal_aligned_face, cal_feature

DEBUG = True
detector = Detector()

KPOINTS = np.array([
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041] ], dtype=np.float32 )

E0PX, E0PY = KPOINTS[0]
E1PX, E1PY = KPOINTS[1]
NPX, NPY = KPOINTS[2]
M0PX, M0PY = KPOINTS[3]
M1PX, M1PY = KPOINTS[4]

MOUTH_W = M1PX - M0PX
MOUTH_W *= 1.236
MOUTH_BXC = (M0PX + M1PX) * 0.5
MOUTH_BYC = (M0PY + M1PY) * 0.5
MOUTH_H = MOUTH_BYC - NPY

MOUTH_BX0, MOUTH_BX1 = MOUTH_BXC - 0.5 * MOUTH_W, MOUTH_BXC + 0.5 * MOUTH_W
MOUTH_BY0, MOUTH_BY1 = MOUTH_BYC - 0.5 * MOUTH_H, MOUTH_BYC + 0.5 * MOUTH_H

EYE_W = (E1PX - E0PX) * 0.5
EYE_W *= 1.236
EYE_H = MOUTH_H * 0.618
EYE0_W = NPX - E0PX
EYE0_W = (EYE0_W + EYE_W) * 0.5
EYE1_W = E1PX - NPX
EYE1_W = (EYE1_W + EYE_W) * 0.5
EYE0_H = EYE_H * EYE0_W / EYE_W
EYE1_H = EYE_H * EYE1_W / EYE_W

EYE0_BX0, EYE0_BX1 = E0PX - 0.5 * EYE0_W, E0PX + 0.5 * EYE0_W
EYE0_BY0, EYE0_BY1 = E0PY - 0.5 * EYE0_H, E0PY + 0.5 * EYE0_H
EYE1_BX0, EYE1_BX1 = E1PX - 0.5 * EYE1_W, E1PX + 0.5 * EYE1_W
EYE1_BY0, EYE1_BY1 = E1PY - 0.5 * EYE1_H, E1PY + 0.5 * EYE1_H

NOSE_BX0, NOSE_BX1 = NPX - 0.618 * EYE1_W, NPX + 0.618 * EYE0_W
NOSE_BY1 = MOUTH_BY0
NOSE_BY0 = min(EYE0_BY1, EYE1_BY1)

MOUTH_BX0 = np.round(MOUTH_BX0).astype(np.int)
MOUTH_BX1 = np.round(MOUTH_BX1).astype(np.int)
MOUTH_BY0 = np.round(MOUTH_BY0).astype(np.int)
MOUTH_BY1 = np.round(MOUTH_BY1).astype(np.int)
NOSE_BX0 = np.round(NOSE_BX0).astype(np.int)
NOSE_BX1 = np.round(NOSE_BX1).astype(np.int)
NOSE_BY0 = np.round(NOSE_BY0).astype(np.int)
NOSE_BY1 = np.round(NOSE_BY1).astype(np.int)
EYE0_BX0 = np.round(EYE0_BX0).astype(np.int)
EYE0_BX1 = np.round(EYE0_BX1).astype(np.int)
EYE0_BY0 = np.round(EYE0_BY0).astype(np.int)
EYE0_BY1 = np.round(EYE0_BY1).astype(np.int)
EYE1_BX0 = np.round(EYE1_BX0).astype(np.int)
EYE1_BX1 = np.round(EYE1_BX1).astype(np.int)
EYE1_BY0 = np.round(EYE1_BY0).astype(np.int)
EYE1_BY1 = np.round(EYE1_BY1).astype(np.int)


def cal_hist(image):
    hist, _edges = np.histogram(image, bins=256, range=(0,256))
    return hist

def get_lbp_feature(im):
    if len(im.shape) == 3 and im.shape[2] == 3:
        im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    else:
        im_gray = im
    radius = 1
    n_points = 8
    lbp = local_binary_pattern(im_gray, n_points, radius)
    hist_all = []
    N = 2
    H, W = im_gray.shape
    h = H // N
    w = W // N
    for i in range(N):
        for j in range(N):
            hist = cal_hist(lbp[i*h:min((i+1)*h,H),j*w:min((j+1)*w,W)])
            hist_all.append(hist)
    hist_all = np.concatenate(hist_all)
    return hist_all

def cal_lbp_simi(hist0, hist1):
    hist_inter = np.minimum(hist0, hist1)
    sum_i = np.sum(hist_inter)
    sum0 = np.sum(hist0)
    sum1 = np.sum(hist1)
    if sum0 == sum1:
        simi = sum_i / sum0
    else:
        simi = 2 * sum_i / (sum0 + sum1)
    return simi

def postprocess(simi):
    simi = simi ** 0.9
    # simi = expit( (simi - 0.5) * 10 )
    return simi

def cal_head(filepath, debug_i):
    im = cv2.imread(filepath)
    assert im is not None

    det_boxes = detector.detect_faces(im, conf_threshold=0.3)
    assert det_boxes.shape[0] == 1
    x0,y0,x1,y1,_conf = det_boxes[0]
    x0,y0,x1,y1 = map(int, (x0,y0,x1,y1))
    box = (x0,y0,x1,y1)
    patch, box = cal_face_patch(im, box)
    if DEBUG:
        cv2.imwrite('test-patch-{}.png'.format(debug_i), patch)
    aligned_face, _padded_face = cal_aligned_face(patch, box)
    assert aligned_face is not None
    if DEBUG:
        cv2.imwrite('test-face-{}.png'.format(debug_i), aligned_face)

    mouth_box = aligned_face[MOUTH_BY0:MOUTH_BY1+1, MOUTH_BX0:MOUTH_BX1+1]
    eye0_box = aligned_face[EYE0_BY0:EYE0_BY1+1, EYE0_BX0:EYE0_BX1+1]
    eye1_box = aligned_face[EYE1_BY0:EYE1_BY1+1, EYE1_BX0:EYE1_BX1+1]
    nose_box = aligned_face[NOSE_BY0:NOSE_BY1+1, NOSE_BX0:NOSE_BX1+1]
    if DEBUG:
        face = aligned_face.copy()
        for px, py in KPOINTS:
            cv2.circle(face, (px,py), 2, (127,255,0))
        cv2.rectangle(face, (MOUTH_BX0,MOUTH_BY0), (MOUTH_BX1,MOUTH_BY1), (0,255,0))
        cv2.rectangle(face, (NOSE_BX0,NOSE_BY0), (NOSE_BX1,NOSE_BY1), (0,255,0))
        cv2.rectangle(face, (EYE0_BX0,EYE0_BY0), (EYE0_BX1,EYE0_BY1), (0,255,0))
        cv2.rectangle(face, (EYE1_BX0,EYE1_BY0), (EYE1_BX1,EYE1_BY1), (0,255,0))
        cv2.imwrite('test-kpoints-{}.png'.format(debug_i), face)
        cv2.imwrite('test-mouth-{}.png'.format(debug_i), mouth_box)
        cv2.imwrite('test-eye0-{}.png'.format(debug_i), eye0_box)
        cv2.imwrite('test-eye1-{}.png'.format(debug_i), eye1_box)
        cv2.imwrite('test-nose-{}.png'.format(debug_i), nose_box)

    mouth_feature = get_lbp_feature(mouth_box)
    nose_feature = get_lbp_feature(nose_box)
    eye0_feature = get_lbp_feature(eye0_box)
    eye1_feature = get_lbp_feature(eye1_box)
    face_feature = cal_feature(aligned_face)
    return face_feature, mouth_feature, nose_feature, eye0_feature, eye1_feature


@click.command()
@click.option('--head1', default='dengchao.jpg', help='image1 filepath')
@click.option('--head2', default='sunli.jpg', help='image2 filepath')
def main(head1, head2):
    face0, mouth0, nose0, eye00, eye10 = cal_head(head1, 0)
    face1, mouth1, nose1, eye01, eye11 = cal_head(head2, 1)
    face_similarity = np.dot(face0, face1) * 0.5 + 0.5
    print(face_similarity)
    mouth_similarity = cal_lbp_simi(mouth0, mouth1)
    nose_similarity = cal_lbp_simi(nose0, nose1)
    eye0_similarity = cal_lbp_simi(eye00, eye01)
    eye1_similarity = cal_lbp_simi(eye10, eye11)
    eye01_similarity = cal_lbp_simi(eye00, eye11)
    eye10_similarity = cal_lbp_simi(eye10, eye01)
    eye_similarity = max(eye0_similarity, eye1_similarity, eye01_similarity, eye10_similarity)
    print(mouth_similarity, nose_similarity, eye_similarity, eye0_similarity, eye1_similarity, eye01_similarity, eye10_similarity)

if __name__ == '__main__':
    main()
