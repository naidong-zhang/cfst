import os
import base64

from flask import render_template, session, redirect, url_for, current_app
# from werkzeug.utils import secure_filename
from . import main
from .forms import PortraitForm


@main.route('/', methods=['GET', 'POST'])
def index():
    portrait_form = PortraitForm()
    b64_portrait_raw_with_boxes = ''
    img_size = 0,0

    b64_face_raw = ''
    face_feature = None

    b64_face1_raw = ''
    face1_feature = None

    face_simi = 0
    mouth_simi = 0
    nose_simi = 0
    eye_simi = 0
    synthetic_simi = 0

    if portrait_form.validate_on_submit():
        fs = portrait_form.portrait.data
        tf = fs.stream
        portrait0_raw = tf.read()
        bboxes, portrait0_raw_with_boxes, img_size = current_app.models.detect_faces(portrait0_raw)
        b64_portrait_raw_with_boxes = base64.b64encode(portrait0_raw_with_boxes).decode()
        session['got_bboxes'] = True

        id_ = portrait_form.bbox_id.data
        bbox = bboxes[id_]
        face_raw = current_app.models.get_aligned_face(portrait0_raw, bbox)
        b64_face_raw = base64.b64encode(face_raw).decode()

        face_feature = current_app.models.cal_feature(face_raw)

        face1_raw = portrait_form.face.data.stream.read()
        b64_face1_raw = base64.b64encode(face1_raw).decode()
        face1_feature = current_app.models.cal_feature(face1_raw)

        mouth, nose, eye0, eye1 = current_app.models.cal_lbp_features(face_raw)
        mouth1, nose1, eye01, eye11 = current_app.models.cal_lbp_features(face1_raw)

        face_simi = current_app.models.cal_face_similarity(face_feature, face1_feature)
        mouth_simi = current_app.models.cal_lbp_similarity(mouth, mouth1)
        nose_simi = current_app.models.cal_lbp_similarity(nose, nose1)
        eye0_simi = current_app.models.cal_lbp_similarity(eye0, eye01)
        eye1_simi = current_app.models.cal_lbp_similarity(eye1, eye11)
        eye01_simi = current_app.models.cal_lbp_similarity(eye0, eye11)
        eye10_simi = current_app.models.cal_lbp_similarity(eye1, eye01)
        eye_simi = max(eye0_simi, eye1_simi, eye01_simi, eye10_simi)
        synthetic_simi = (eye_simi + nose_simi + mouth_simi + face_simi) / 4


        face_feature = face_feature[:3]
        face1_feature = face1_feature[:3]
        # return redirect(url_for('.index'))

    return render_template('index.html', portrait_form=portrait_form, got_bboxes=session.get('got_bboxes', False),
                            portrait_b64_me=b64_portrait_raw_with_boxes, width=img_size[0], height=img_size[1],
                            face_b64_me=b64_face_raw, face_feature=str(face_feature),
                            face1_b64_spouse=b64_face1_raw, face1_feature=str(face1_feature),
                            face_simi=face_simi, mouth_simi=mouth_simi, nose_simi=nose_simi, eye_simi=eye_simi, synthetic_simi=synthetic_simi)
