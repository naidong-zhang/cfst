import os
import base64

from flask import render_template, session, redirect, url_for, current_app, flash, request, jsonify
# from werkzeug.utils import secure_filename
from . import cfst
from .errors import bad_request

import numpy as np
import cv2

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def _allowed_file(filename):
    return '.' in filename and filename.rsplit('.',1)[1].lower() in ALLOWED_EXTENSIONS

def _cv2_read_raw(raw_data):
    enc = np.frombuffer(raw_data, dtype=np.uint8)
    im = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    assert im is not None
    return im

def _preprocess(im):
    im1 = cv2.bilateralFilter(im, 15, 20, 5)
    return im1

def _cv2_write_raw(im, format='.png'):
    ret, enc = cv2.imencode(format, im)
    assert ret == True
    enc = np.squeeze(enc)
    raw_data = enc.tobytes()
    return raw_data

@cfst.route('/detect/', methods=['POST'])
def detect_faces():
    b64_portrait_raw = request.json.get('portrait')
    if b64_portrait_raw is None or len(b64_portrait_raw) == 0:
        return jsonify({'error': 'No Photo File!'})
    portrait_raw = base64.b64decode(b64_portrait_raw)

    try:
        portrait_im = _cv2_read_raw(portrait_raw)
    except AssertionError:
        return jsonify({'error': 'Not an Image file!'})
        # return redirect(url_for('.cfst'))

    h,w,_ = portrait_im.shape
    # test-code
    # bboxes = np.array([[124,71,204,171],[254,29,314,97]])
    bboxes = current_app.models.detect_faces(portrait_im, w=w, h=h)
    bboxes = bboxes.tolist()

    return jsonify({'bboxes': bboxes})

@cfst.route('/align/', methods=['POST'])
def align_face():
    b64_patch_raw = request.json.get('patch')
    if b64_patch_raw is None or len(b64_patch_raw) == 0:
        return jsonify({'error': 'No patch data!'})
    bbox = request.json.get('bbox')
    if bbox is None or len(bbox) != 4:
        return jsonify({'error': 'Bad bbox!'})
    patch_raw = base64.b64decode(b64_patch_raw)

    patch_im = _cv2_read_raw(patch_raw)
    # test-code
    # aligned_face = patch_im[:112,:112]
    aligned_face = current_app.models.align_face(patch_im, bbox)
    if aligned_face is None:
        return jsonify({'aligned': False})

    face_raw = _cv2_write_raw(aligned_face)
    b64_face_raw = base64.b64encode(face_raw).decode()

    return jsonify({'face': b64_face_raw})

@cfst.route('/similarity/', methods=['POST'])
def cal_simi():
    b64_face_me = request.json.get('face_me')
    if b64_face_me is None or len(b64_face_me) == 0:
        return jsonify({'error': 'No my face!'})
    b64_face_spouse = request.json.get('face_spouse')
    if b64_face_spouse is None or len(b64_face_spouse) == 0:
        return jsonify({'error': "No my spouse's face!"})
    aligned = request.json.get('aligned')

    face_me_raw = base64.b64decode(b64_face_me)
    face_spouse_raw = base64.b64decode(b64_face_spouse)

    face_me_im = _cv2_read_raw(face_me_raw)
    face_spouse_im = _cv2_read_raw(face_spouse_raw)

    face_me_im = _preprocess(face_me_im)
    face_spouse_im = _preprocess(face_spouse_im)

    # test-code
    # face_simi = 0.18281828
    face_simi = current_app.models.cal_face_simi(face_me_im, face_spouse_im)
    if aligned:
        # test-code
        # mouth_simi, nose_simi, eye_simi = 0.4, 0.6, 0.7
        mouth_simi, nose_simi, eye_simi = current_app.models.cal_lbp_simi(face_me_im, face_spouse_im)
        synthetic_simi = (eye_simi + nose_simi + mouth_simi) * 0.2 + face_simi * 0.4
        return jsonify({'face_simi': face_simi, 'mouth_simi': mouth_simi, 'nose_simi': nose_simi, 'eye_simi': eye_simi, 'syn_simi': synthetic_simi})
    return jsonify({'face_simi': face_simi})

@cfst.route('/')
def cfst(face0=''):
    return render_template('cfst.html')
