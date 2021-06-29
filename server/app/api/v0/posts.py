import base64
import numpy as np
import cv2

from flask import jsonify, request, url_for, redirect, current_app
from . import api
from .errors import bad_request

def _cv2_read_raw(raw_data):
    enc = np.frombuffer(raw_data, dtype=np.uint8)
    im = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    assert im is not None
    return im

def _cv2_write_raw(im, format='.png'):
    ret, enc = cv2.imencode(format, im)
    assert ret == True
    enc = np.squeeze(enc)
    raw_data = enc.tobytes()
    return raw_data

def _preprocess(im):
    im1 = cv2.bilateralFilter(im, 15, 20, 5)
    return im1

@api.route('/login/', methods=['POST'])
def login():
    code = request.json.get('code')
    print(code)
    return jsonify({'res': code})

@api.route('/detect/', methods=['POST'])
def detect_faces():
    b64_portrait_raw = request.json.get('portrait')
    if b64_portrait_raw is None or len(b64_portrait_raw) == 0:
        return bad_request('no portrait data')
    portrait_raw = base64.b64decode(b64_portrait_raw)
    w = request.json.get('w')
    h = request.json.get('h')
    w, h = int(w), int(h)

    portrait_im = _cv2_read_raw(portrait_raw)
    bboxes = current_app.models.detect_faces(portrait_im, w=w, h=h)
    bboxes = bboxes.tolist()

    return jsonify({'bboxes': bboxes})

@api.route('/align/', methods=['POST'])
def get_aligned_face():
    b64_patch_raw = request.json.get('patch')
    if b64_patch_raw is None or len(b64_patch_raw) == 0:
        return bad_request('no patch data')
    bbox = request.json.get('bbox')
    if bbox is None or len(bbox) != 4:
        return bad_request('bad bbox')
    patch_raw = base64.b64decode(b64_patch_raw)

    patch_im = _cv2_read_raw(patch_raw)
    aligned_face = current_app.models.align_face(patch_im, bbox)
    if aligned_face is None:
        return jsonify({'aligned': False})

    face_raw = _cv2_write_raw(aligned_face)
    b64_face_raw = base64.b64encode(face_raw).decode()

    return jsonify({'face': b64_face_raw})

@api.route('/similarity/', methods=['POST'])
def cal_simi():
    b64_face_me = request.json.get('face_me')
    if b64_face_me is None or len(b64_face_me) == 0:
        return bad_request('no my face')
    b64_face_spouse = request.json.get('face_spouse')
    if b64_face_spouse is None or len(b64_face_spouse) == 0:
        return bad_request("no my spouse's face")
    aligned = request.json.get('aligned')

    face_me_raw = base64.b64decode(b64_face_me)
    face_spouse_raw = base64.b64decode(b64_face_spouse)

    face_me_im = _cv2_read_raw(face_me_raw)
    face_spouse_im = _cv2_read_raw(face_spouse_raw)

    face_me_im = _preprocess(face_me_im)
    face_spouse_im = _preprocess(face_spouse_im)

    face_simi = current_app.models.cal_face_simi(face_me_im, face_spouse_im)
    if aligned:
        mouth_simi, nose_simi, eye_simi = current_app.models.cal_lbp_simi(face_me_im, face_spouse_im)
        synthetic_simi = (eye_simi + nose_simi + mouth_simi) * 0.2 + face_simi * 0.4
        return jsonify({'face_simi': face_simi, 'mouth_simi': mouth_simi, 'nose_simi': nose_simi, 'eye_simi': eye_simi, 'syn_simi': synthetic_simi})
    return jsonify({'face_simi': face_simi})







@api.route('/posts/')
def get_posts():
    page = request.args.get('page', 1, type=int)
    pagination = Post.query.paginate(
        page, per_page=current_app.config['FLASKY_POSTS_PER_PAGE'],
        error_out=False)
    posts = pagination.items
    prev = None
    if pagination.has_prev:
        prev = url_for('api.get_posts', page=page-1)
    next = None
    if pagination.has_next:
        next = url_for('api.get_posts', page=page+1)
    return jsonify({
        'posts': [post.to_json() for post in posts],
        'prev': prev,
        'next': next,
        'count': pagination.total
    })


@api.route('/posts/<int:id>')
def get_post(id):
    post = Post.query.get_or_404(id)
    return jsonify(post.to_json())


@api.route('/posts/', methods=['POST'])
def new_post():
    post = Post.from_json(request.json)
    post.author = g.current_user
    db.session.add(post)
    db.session.commit()
    return jsonify(post.to_json()), 201, \
        {'Location': url_for('api.get_post', id=post.id)}


@api.route('/posts/<int:id>', methods=['PUT'])
def edit_post(id):
    post = Post.query.get_or_404(id)
    if g.current_user != post.author and \
            not g.current_user.can(Permission.ADMIN):
        return forbidden('Insufficient permissions')
    post.body = request.json.get('body', post.body)
    db.session.add(post)
    db.session.commit()
    return jsonify(post.to_json())
