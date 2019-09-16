import base64

from flask import jsonify, request, url_for, current_app
from . import api
from .errors import bad_request

@api.route('/detect/', methods=['POST'])
def detect_faces():
    b64_portrait_raw = request.json.get('portrait')
    if b64_portrait_raw is None or len(b64_portrait_raw) == 0:
        return bad_request('no portrait data')
    portrait_raw = base64.b64decode(b64_portrait_raw)
    w = request.json.get('w')
    h = request.json.get('h')
    w, h = int(w), int(h)

    bboxes = current_app.models.detect_faces(portrait_raw, w=w, h=h)
    bboxes = bboxes.tolist()

    return jsonify({'bboxes': bboxes})

@api.route('/align/', methods=['POST'])
def get_aligned_face():
    b64_portrait_raw = request.json.get('portrait')
    if b64_portrait_raw is None or len(b64_portrait_raw) == 0:
        return bad_request('no portrait data')
    bbox = request.json.get('bbox')
    if bbox is None or len(bbox) != 4:
        return bad_request('bad bbox')
    portrait_raw = base64.b64decode(b64_portrait_raw)

    face_raw = current_app.models.get_aligned_face(portrait_raw, bbox)
    b64_face_raw = base64.b64encode(face_raw).decode()

    return jsonify({'face': b64_face_raw})



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
