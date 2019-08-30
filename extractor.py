#!/usr/bin/env python3
#coding:utf-8

import os
import click
import numpy as np
import cv2
from infer import Aligner, recog_preprocess, Recognizer

# from editor import DBPATH
DBPATH = './db_jav'
FACES_FOLDER = os.path.join('data','faces')
FEATURE_DATA = os.path.join('data', 'features')
MTCNN_PATH = 'models/mtcnn-model'
GPU = 0
RNET_THRESHOLD = 0.0
ONET_THRESHOLD = 0.2
MODEL_PATH = 'models/model-y1-test2/model,0'
FEATURE_DIMENSION = 128

aligner = None
recognizer = None

def cal_face_patch(image, box):
    # convert_to_square
    x0,y0,x1,y1 = box
    h = y1 - y0 + 1
    w = x1 - x0 + 1
    max_side = max(h, w)
    # pad
    max_side += max_side // 2

    px0 = x0 + w // 2 - max_side // 2
    py0 = y0 + h // 2 - max_side // 2
    px1 = px0 + max_side - 1
    py1 = py0 + max_side - 1
    px0 = max(px0, 0)
    py0 = max(py0, 0)
    patch = image[py0:py1+1,px0:px1+1]

    bx0 = x0 - px0
    by0 = y0 - py0
    bx1 = x1 - px0
    by1 = y1 - py0
    box2 = bx0,by0,bx1,by1
    return patch, box2

def cal_aligned_face(patch, box, padded_too=False):
    global aligner
    aligned_face, padded_face = None, None

    if aligner is None:
        aligner = Aligner(MTCNN_PATH, GPU)

    box_conf = (*box, 0.9)
    det_box = np.array(box_conf, dtype=np.float32)
    det_box = np.expand_dims(det_box, axis=0)
    ret = aligner.get_box_kpoints(patch,
                            rnet_threshold=RNET_THRESHOLD,
                            onet_threshold=ONET_THRESHOLD,
                            det_box=det_box)
    if ret is not None:
        box, kpoints = ret
        _is_aligned, aligned_face = recog_preprocess(patch, box, kpoints)
    if ret is None or padded_too:
        box, kpoints = det_box[0][:4], None
        _is_aligned, padded_face = recog_preprocess(patch, box, kpoints)

    return aligned_face, padded_face

def cal_feature(face_image):
    global recognizer
    if recognizer is None:
        recognizer = Recognizer(MODEL_PATH, GPU)
    feature = recognizer.get_feature(face_image)
    return feature



def get_knn(feature, t, faces_filenames, k=10):
    ids, dises = t.get_nns_by_vector(feature, k, include_distances=True)
    ret = [(faces_filenames[i], dis * 0.5 + 0.5) for i, dis in zip(ids, dises)]
    return ret


def generate_all_faces(dst_folder=FACES_FOLDER):
    import psycopg2

    if os.path.exists(dst_folder):
        raise FileExistsError('please remove {} first!'.format(dst_folder))

    os.mkdir(dst_folder)
    conn = psycopg2.connect('dbname=db_jav')
    cur = conn.cursor()
    cur.execute("SELECT face_id, x0, y0, x1, y1, banggo_id FROM face;")
    face_id_x0_y0_x1_y1_banggo_id = cur.fetchall()
    i = 0
    n = len(face_id_x0_y0_x1_y1_banggo_id)
    for face_id, x0, y0, x1, y1, banggo_id in face_id_x0_y0_x1_y1_banggo_id:
        cur.execute("SELECT raw_cover_filepath FROM raw WHERE id={};".format(banggo_id))
        raw_cover_filepath = cur.fetchone()[0]
        im_raw = cv2.imread(os.path.join(DBPATH, raw_cover_filepath), cv2.IMREAD_COLOR)
        assert im_raw is not None, raw_cover_filepath
        cur.execute("SELECT x0, y0, x1, y1 FROM roi WHERE banggo_id={};".format(banggo_id))
        rx0, ry0, rx1, ry1 = cur.fetchone()
        im_roi = im_raw[ry0:ry1+1,rx0:rx1+1]
        box = (x0,y0,x1,y1)
        patch, box = cal_face_patch(im_roi, box)
        aligned_face, padded_face = cal_aligned_face(patch, box, True)
        if aligned_face is not None:
            cv2.imwrite(os.path.join(dst_folder, '{}_1.png'.format(face_id)), aligned_face)
        if padded_face is not None:
            cv2.imwrite(os.path.join(dst_folder, '{}_0.png'.format(face_id)), padded_face)
        i += 1
        if i % 100 == 0:
            print('generate_all_faces:{}/{}'.format(i, n), end='\r', flush=True)
    cur.close()
    conn.close()
    global aligner
    aligner = None
    print('close database', flush=True)

def generate_all_features(faces_folder=FACES_FOLDER, feature_data=FEATURE_DATA):
    import pickle
    import annoy

    feature_file = feature_data + '.ann'
    if os.path.exists(feature_file):
        raise FileExistsError('please remove {} first!'.format(feature_file))

    faces_filenames = os.listdir(faces_folder)
    with open(feature_data + '.pkl', 'wb') as f:
        pickle.dump(faces_filenames, f)
    t = annoy.AnnoyIndex(FEATURE_DIMENSION, metric='dot')
    i = 0
    for fn in faces_filenames:
        fp = os.path.join(faces_folder, fn)
        face_image = cv2.imread(fp, cv2.IMREAD_COLOR)
        feature = cal_feature(face_image)
        assert feature.size == FEATURE_DIMENSION
        t.add_item(i, feature)
        i += 1
    t.build(FEATURE_DIMENSION)
    t.save(feature_file)
    global recognizer
    recognizer = None
    print('save {} features from {} to {}.'.format(i, faces_folder, feature_file), flush=True)

def test(image_file, debug=True):
    import pickle
    import annoy
    from det import Detector

    im = cv2.imread(image_file, cv2.IMREAD_COLOR)
    detector = Detector()
    det_boxes = detector.detect_faces(im, conf_threshold=0.3)
    assert det_boxes.shape[0] == 1
    x0,y0,x1,y1,_conf = det_boxes[0]
    x0,y0,x1,y1 = map(int, (x0,y0,x1,y1))
    box = (x0,y0,x1,y1)
    patch, box = cal_face_patch(im, box)
    if debug:
        cv2.imwrite('test-patch.png', patch)
    aligned_face, padded_face = cal_aligned_face(patch, box)
    face_image = aligned_face if aligned_face is not None else padded_face
    if debug:
        cv2.imwrite('test-face.png', face_image)
    feature = cal_feature(face_image)

    t = annoy.AnnoyIndex(FEATURE_DIMENSION, metric='dot')
    t.load(FEATURE_DATA + '.ann')
    with open(FEATURE_DATA + '.pkl', 'rb') as f:
        faces_filenames = pickle.load(f)
    ret = get_knn(feature, t, faces_filenames)
    print(ret)
    global aligner
    aligner = None
    global recognizer
    recognizer = None



@click.command()
@click.option('--run', default='', help='function name')
@click.option('--test-image', default='', help='test image filepath')
def main(run, test_image):
    if run == 'generate_all_faces':
        generate_all_faces()
    elif run == 'generate_all_features':
        generate_all_features()
    elif run == 'test':
        test(test_image)

if __name__ == '__main__':
    main()
