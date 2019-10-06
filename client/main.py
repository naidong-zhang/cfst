#!/usr/bin/env python3
#coding:utf-8
import os
import base64
import json

import numpy as np
import cv2

import kivy
kivy.require('1.11.1')
from kivy.app import App
# from kivy.core.window import Window
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.popup import Popup
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.label import Label
from kivy.properties import ObjectProperty, BooleanProperty
from kivy.uix.image import Image
from kivy.graphics.texture import Texture
from kivy.network.urlrequest import UrlRequest
# from kivy.clock import Clock
from kivy.animation import Animation

from utils import cal_pts, EYE_PT0, EYE_PT1, NOSE_PT0, NOSE_PT1, MOUTH_PT0, MOUTH_PT1


SERVER_URL = 'http://192.168.1.101:5000/api/v0'


def _bright(im_rgb, v):
    if v > 0:
        hsv = cv2.cvtColor(im_rgb, cv2.COLOR_RGB2HSV)
        hsv[..., 2] = np.where(hsv[..., 2] < (255 - v), hsv[..., 2] + v, 255)
        im_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    elif v < 0:
        # dark all
        v = -v
        hsv = cv2.cvtColor(im_rgb, cv2.COLOR_RGB2HSV)
        hsv[..., 2] = np.where(hsv[..., 2] > v, hsv[..., 2] - v, 0)
        im_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return im_rgb


class LoadDialog(FloatLayout):
    load = ObjectProperty(None)
    cancel = ObjectProperty(None)


class SelectDialog(FloatLayout):
    portrait = ObjectProperty(None)
    select = ObjectProperty(None)
    cancel = ObjectProperty(None)

    def __init__(self, portrait_rgb, bboxes, **kwargs):
        super(SelectDialog, self).__init__(**kwargs)
        self.portrait_rgb = portrait_rgb
        self.bboxes = bboxes
        self.cur_bbox_id = 0
        self.touch_pos_in_cv2 = (0, 0)

        self.portrait.bind(on_touch_down=self.callback_select)
        self.reload_portrait()

    def callback_select(self, instance, touch):
        if self.collide_point(touch.x, touch.y):
            w, h = self.portrait.size
            w, h = float(w), float(h)
            rw, rh = self.portrait.texture.size
            rw, rh = float(rw), float(rh)

            tx, ty = self.portrait.to_local(touch.x, touch.y, relative=True)
            sx, sy = tx / w, ty / h

            rratio = rh / rw
            ratio = h / w
            if ratio <= rratio:
                # x axis
                px = rw * ((tx - w * 0.5) / (rw * h / rh) + 0.5)
                py = rh * (1 - sy)
            else:
                # y axis
                px = rw * sx
                py = rh * ((h * 0.5 - ty) / (rh * w / rw) + 0.5)
            self.touch_pos_in_cv2 = int(round(px)), int(round(py))
            x, y = self.touch_pos_in_cv2
            for i, bbox in enumerate(self.bboxes):
                x0,y0,x1,y1 = bbox
                if x0 <= x <= x1 and y0 <= y <= y1:
                    self.cur_bbox_id = i
                    break

            self.reload_portrait()
            return True

    def reload_portrait(self):
        im = self.portrait_rgb.copy()
        im = _bright(im, -50)
        for i, bbox in enumerate(self.bboxes):
            x0,y0,x1,y1 = bbox
            if i == self.cur_bbox_id:
                bord_color = (255,255,0)
                brightness = 50
            else:
                bord_color = (0,255,0)
                brightness = 20
            im = cv2.rectangle(im, (x0-1,y0-1), (x1+1,y1+1), bord_color, 2)
            region = im[y0:y1+1, x0:x1+1]
            im[y0:y1+1, x0:x1+1] = _bright(region, brightness)

        # for debug
        # im = cv2.circle(im, self.touch_pos_in_cv2, 5, (255,0,0), -1)

        h,w,_c = self.portrait_rgb.shape
        texture = Texture.create(size=(w,h), colorfmt='rgb')
        texture.blit_buffer(im.ravel(), colorfmt='rgb', bufferfmt='ubyte', mipmap_generation=False)
        texture.flip_vertical()
        self.portrait.texture = texture


class AnimImage(Image):
    def __init__(self, **kwargs):
        super(AnimImage, self).__init__(**kwargs)
        self._cv2_im_rgb = None
        self._eye_brightness = 0.
        self._nose_brightness = 0.
        self._mouth_brightness = 0.
        self._face_brightness = 0.

    @property
    def cv2_im_rgb(self):
        return self._cv2_im_rgb

    @cv2_im_rgb.setter
    def cv2_im_rgb(self, im_rgb):
        w,h = 112,112
        assert im_rgb.shape[:2] == (h,w)
        texture = Texture.create(size=(w,h), colorfmt='rgb')
        texture.blit_buffer(im_rgb.ravel(), colorfmt='rgb', bufferfmt='ubyte', mipmap_generation=False)
        texture.flip_vertical()
        self.texture = texture

        self._cv2_im_rgb = im_rgb
        # reload new aligned face
        self._eye_brightness = 0.
        self._nose_brightness = 0.
        self._mouth_brightness = 0.
        self._face_brightness = 0.

    def _set_brightness(self, part, v):
        if self._cv2_im_rgb is None:
            return

        im = self._cv2_im_rgb.copy()

        if part == 'eye':
            if v != 0:
                eye_box = im[EYE_PT0[1]:EYE_PT1[1]+1, EYE_PT0[0]:EYE_PT1[0]+1]
                im[EYE_PT0[1]:EYE_PT1[1]+1, EYE_PT0[0]:EYE_PT1[0]+1] = _bright(eye_box, v)
                im = cv2.rectangle(im, EYE_PT0, EYE_PT1, (0,255,0), thickness=2)
        elif part == 'nose':
            if v != 0:
                nose_box = im[NOSE_PT0[1]:NOSE_PT1[1]+1, NOSE_PT0[0]:NOSE_PT1[0]+1]
                im[NOSE_PT0[1]:NOSE_PT1[1]+1, NOSE_PT0[0]:NOSE_PT1[0]+1] = _bright(nose_box, v)
                im = cv2.rectangle(im, NOSE_PT0, NOSE_PT1, (0,255,0), thickness=2)
        elif part == 'mouth':
            if v != 0:
                mouth_box = im[MOUTH_PT0[1]:MOUTH_PT1[1]+1, MOUTH_PT0[0]:MOUTH_PT1[0]+1]
                im[MOUTH_PT0[1]:MOUTH_PT1[1]+1, MOUTH_PT0[0]:MOUTH_PT1[0]+1] = _bright(mouth_box, v)
                im = cv2.rectangle(im, MOUTH_PT0, MOUTH_PT1, (0,255,0), thickness=2)
        elif part == 'face':
            im = _bright(im, v)

        w,h = 112,112
        texture = Texture.create(size=(w,h), colorfmt='rgb')
        texture.blit_buffer(im.ravel(), colorfmt='rgb', bufferfmt='ubyte', mipmap_generation=False)
        texture.flip_vertical()
        self.texture = texture

    @property
    def eye_brightness(self):
        return self._eye_brightness

    @eye_brightness.setter
    def eye_brightness(self, v):
        self._set_brightness('eye', v)
        self._eye_brightness = v

    @property
    def nose_brightness(self):
        return self._nose_brightness

    @nose_brightness.setter
    def nose_brightness(self, v):
        self._set_brightness('nose', v)
        self._nose_brightness = v

    @property
    def mouth_brightness(self):
        return self._mouth_brightness

    @mouth_brightness.setter
    def mouth_brightness(self, v):
        self._set_brightness('mouth', v)
        self._mouth_brightness = v

    @property
    def face_brightness(self):
        return self._face_brightness

    @face_brightness.setter
    def face_brightness(self, v):
        self._set_brightness('face', v)
        self._face_brightness = v

class AnimLabel(Label):
    def __init__(self, **kwargs):
        super(AnimLabel, self).__init__(**kwargs)
        self._full_text = ''
        self._percentage = 0.

    @property
    def full_text(self):
        return self._full_text

    @full_text.setter
    def full_text(self, text):
        self._full_text = text
        # update full_text, so percentage reset
        self.percentage = 0.

    @property
    def percentage(self):
        return self._percentage

    @percentage.setter
    def percentage(self, v):
        v = 0 if v < 0 else v
        v = 1 if v > 1 else v
        p = round(v * len(self._full_text))
        placeholder = self._full_text[p:]
        lines = placeholder.split('\n')
        lines = [' ' * len(line) for line in lines]
        placeholder = '\n'.join(lines)
        self.text = self._full_text[:p] + placeholder
        self._percentage = v


class CFST(BoxLayout):
    face_me = ObjectProperty(None)
    face_spouse = ObjectProperty(None)
    btn_run = ObjectProperty(None)
    eye_simi_result_title = ObjectProperty(None)
    eye_simi_result = ObjectProperty(None)
    nose_simi_result_title = ObjectProperty(None)
    nose_simi_result = ObjectProperty(None)
    mouth_simi_result_title = ObjectProperty(None)
    mouth_simi_result = ObjectProperty(None)
    face_simi_result_title = ObjectProperty(None)
    face_simi_result = ObjectProperty(None)
    syn_simi_result_title = ObjectProperty(None)
    syn_simi_result = ObjectProperty(None)

    HEADERS = {'Content-type': 'application/json'}

    def __init__(self, **kwargs):
        super(CFST, self).__init__(**kwargs)

        self.popup = None
        self.portrait_rgb = None

        self.is_me = True
        self.got_me = False
        self.got_spouse = False

        self.b64_face_me = ''
        self.b64_face_spouse = ''

        self.anims = list()

    def input_my_photo(self):
        self._input_photo(is_me=True)

    def input_spouse_photo(self):
        self._input_photo(is_me=False)

    def _input_photo(self, is_me):
        self.is_me = is_me
        content = LoadDialog(load=self._detect, cancel=self._dismiss_popup)
        title = 'INPUT MY PHOTO' if is_me else "INPUT MY SPOUSE'S PHOTO"
        self.popup = Popup(title=title, content=content, size_hint=(0.9, 0.9))
        self.popup.open()

    def _dismiss_popup(self):
        self.popup.dismiss()

    def _detect(self, path, filenames):
        # load portrait
        fp = os.path.join(path, filenames[0])
        with open(fp, 'rb') as f:
            data = f.read()
            raw = np.frombuffer(data, dtype=np.uint8)
            im = cv2.imdecode(raw, cv2.IMREAD_COLOR)
            assert im is not None

        self.portrait_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        h, w = im.shape[:2]
        if h > 300 or w > 300:
            im = cv2.resize(im, (300,300))
        ret, raw = cv2.imencode('.jpg', im)
        assert ret == True
        raw = np.squeeze(raw)
        data1 = raw.tobytes()
        if len(data) < len(data1):
            data1 = data

        self._dismiss_popup()

        # send portrait
        b64_portrait_raw = base64.b64encode(data1).decode()
        body = json.dumps({'portrait': b64_portrait_raw, 'w': w, 'h': h})
        _req = UrlRequest(SERVER_URL + '/detect/', req_headers=self.HEADERS, req_body=body,
                            on_success=self.req_detect,
                            on_error=lambda req, err: self._pop_error(err),
                            on_failure=lambda req, res: self._pop_error(res))

    def req_detect(self, req, result):
        bboxes = result.get('bboxes')
        assert bboxes is not None
        if len(bboxes) == 0:
            self._pop_error('no face detected!')
            return
        elif len(bboxes) == 1:
            return self._align(bboxes[0])

        content = SelectDialog(portrait_rgb=self.portrait_rgb, bboxes=bboxes, select=self._align, cancel=self._dismiss_popup)
        title = 'SELECT MY FACE' if self.is_me else "SELECT MY SPOUSE'S FACE"
        self.popup = Popup(title=title, content=content, size_hint=(0.9, 0.9))
        self.popup.open()

    def _pop_error(self, message):
        popup = Popup(title='error', content=Label(text=str(message)), size_hint=(0.9, 0.4))
        popup.open()

    @staticmethod
    def _cal_face_patch(image, box):
        # convert_to_square
        x0,y0,x1,y1 = box
        h = y1 - y0 + 1
        w = x1 - x0 + 1
        max_side = max(h, w)
        # pad
        max_side += max_side // 2

        square = np.zeros((max_side, max_side, image.shape[2]), dtype=np.uint8)

        px0 = x0 + w // 2 - max_side // 2
        py0 = y0 + h // 2 - max_side // 2
        px1 = px0 + max_side - 1
        py1 = py0 + max_side - 1

        src = image[max(py0,0):py1+1, max(px0,0):px1+1]
        h, w = src.shape[:2]
        square[max(-py0,0):max(-py0,0)+h, max(-px0,0):max(-px0,0)+w] = src

        bx0 = x0 - px0
        by0 = y0 - py0
        bx1 = x1 - px0
        by1 = y1 - py0
        box2 = bx0,by0,bx1,by1
        return square, box2

    def _align(self, bbox):
        patch_rgb, box = self._cal_face_patch(self.portrait_rgb, bbox)
        patch = cv2.cvtColor(patch_rgb, cv2.COLOR_RGB2BGR)
        ret, raw = cv2.imencode('.png', patch)
        assert ret == True
        raw = np.squeeze(raw)
        data = raw.tobytes()

        self._dismiss_popup()

        # send patch
        b64_patch_raw = base64.b64encode(data).decode()
        body = json.dumps({'patch': b64_patch_raw, 'bbox': box})
        _req = UrlRequest(SERVER_URL + '/align/', req_headers=self.HEADERS, req_body=body,
                            on_success=self.req_align,
                            on_error=lambda req, err: self._pop_error(err),
                            on_failure=lambda req, res: self._pop_error(res))

    def req_align(self, req, result):
        b64_face_raw = result.get('face')
        if b64_face_raw is None and result.get('aligned') == False:
            self._pop_error('face direction is bad!')
            return

        # show face
        face_raw = base64.b64decode(b64_face_raw)
        raw = np.frombuffer(face_raw, dtype=np.uint8)
        im = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        assert im is not None
        im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        if self.is_me:
            self.face_me.cv2_im_rgb = im_rgb
            self.got_me = True
            self.b64_face_me = b64_face_raw
        else:
            self.face_spouse.cv2_im_rgb = im_rgb
            self.got_spouse = True
            self.b64_face_spouse = b64_face_raw
        self.btn_simi.text = 'ANALYZE (ONLY $1.99)'
        self._refresh_btn_simi()

    def _refresh_btn_simi(self):
        if self.got_me and self.got_spouse:
            self.btn_simi.disabled = False
        else:
            self.btn_simi.disabled = True

        for anim, widget in self.anims:
            anim.cancel(widget)

        self.eye_simi_result_title.full_text = ''
        self.eye_simi_result.full_text = ''
        self.nose_simi_result_title.full_text = ''
        self.nose_simi_result.full_text = ''
        self.mouth_simi_result_title.full_text = ''
        self.mouth_simi_result.full_text = ''
        self.face_simi_result_title.full_text = ''
        self.face_simi_result.full_text = ''
        self.syn_simi_result_title.full_text = ''
        self.syn_simi_result.full_text = ''

        self.face_me.eye_brightness = 0
        self.face_me.nose_brightness = 0
        self.face_me.mouth_brightness = 0
        self.face_me.face_brightness = 0
        self.face_spouse.eye_brightness = 0
        self.face_spouse.nose_brightness = 0
        self.face_spouse.mouth_brightness = 0
        self.face_spouse.face_brightness = 0

    def simi(self):
        self.btn_simi.text = 'ANALYZING...'
        self._refresh_btn_simi()
        # send face pair
        body = json.dumps({'face_me': self.b64_face_me, 'face_spouse': self.b64_face_spouse, 'aligned': True})
        _req = UrlRequest(SERVER_URL + '/similarity/', req_headers=self.HEADERS, req_body=body,
                            on_success=self.req_simi,
                            on_error=lambda req, err: self._pop_error(err),
                            on_failure=lambda req, res: self._pop_error(res))

    def _bind_next_anim(self, *next_anim_next_widget):
        self.anims = [(anim, widget) for anim, widget in self.anims if len(anim.animated_properties) != 0]
        for next_anim, next_widget in next_anim_next_widget:
            next_anim.start(next_widget)
            self.anims.append((next_anim, next_widget))

    def _update_btn_simi(self):
        self.btn_simi.text = 'ANALYZE (ONLY $1.59)'

    def req_simi(self, req, result):
        face_simi = result.get('face_simi')
        eye_simi = result.get('eye_simi')
        assert eye_simi is not None
        nose_simi = result.get('nose_simi')
        mouth_simi = result.get('mouth_simi')
        syn_simi = result.get('syn_simi')

        bt0 = 1.
        bt1 = 0.3
        tt = 0.2
        st = 0.5
        tt2 = 0.3
        st2 = 1.

        binds = list()
        for image in (self.face_me, self.face_spouse):
            image_anim = Animation(eye_brightness=100., duration=bt0, step=1./30, transition='in_elastic')
            image_anim += Animation(eye_brightness=0., duration=bt1, step=1./30, transition='in_elastic')
            binds.append((image_anim, image))

        self.eye_simi_result_title.full_text = 'EYES SIMILARITY: '
        self.eye_simi_result.full_text = '{:.2%}'.format(eye_simi)
        title_anim = Animation(percentage=1., duration=tt, step=1./30, transition='linear')
        image_anim.on_complete = lambda _w: self._bind_next_anim((title_anim, self.eye_simi_result_title))
        score_anim = Animation(percentage=1., duration=st, step=1./30, transition='linear')
        title_anim.on_complete = lambda _w: self._bind_next_anim((score_anim, self.eye_simi_result))

        binds2 = list()
        for image in (self.face_me, self.face_spouse):
            image_anim = Animation(nose_brightness=100., duration=bt0, step=1./30, transition='in_elastic')
            image_anim += Animation(nose_brightness=0., duration=bt1, step=1./30, transition='in_elastic')
            binds2.append((image_anim, image))
        score_anim.on_complete = lambda _w: self._bind_next_anim(*binds2)

        self.nose_simi_result_title.full_text = 'NOSE SIMILARITY: '
        self.nose_simi_result.full_text = '{:.2%}'.format(nose_simi)
        title_anim2 = Animation(percentage=1., duration=tt, step=1./30, transition='linear')
        image_anim.on_complete = lambda _w: self._bind_next_anim((title_anim2, self.nose_simi_result_title))
        score_anim2 = Animation(percentage=1., duration=st, step=1./30, transition='linear')
        title_anim2.on_complete = lambda _w: self._bind_next_anim((score_anim2, self.nose_simi_result))

        binds3 = list()
        for image in (self.face_me, self.face_spouse):
            image_anim = Animation(mouth_brightness=100., duration=bt0, step=1./30, transition='in_elastic')
            image_anim += Animation(mouth_brightness=0., duration=bt1, step=1./30, transition='in_elastic')
            binds3.append((image_anim, image))
        score_anim2.on_complete = lambda _w: self._bind_next_anim(*binds3)

        self.mouth_simi_result_title.full_text = 'MOUTH SIMILARITY: '
        self.mouth_simi_result.full_text = '{:.2%}'.format(mouth_simi)
        title_anim3 = Animation(percentage=1., duration=tt, step=1./30, transition='linear')
        image_anim.on_complete = lambda _w: self._bind_next_anim((title_anim3, self.mouth_simi_result_title))
        score_anim3 = Animation(percentage=1., duration=st, step=1./30, transition='linear')
        title_anim3.on_complete = lambda _w: self._bind_next_anim((score_anim3, self.mouth_simi_result))

        binds4 = list()
        for image in (self.face_me, self.face_spouse):
            image_anim = Animation(face_brightness=100., duration=bt0, step=1./30, transition='in_elastic')
            image_anim += Animation(face_brightness=0., duration=bt1, step=1./30, transition='in_elastic')
            binds4.append((image_anim, image))
        score_anim3.on_complete = lambda _w: self._bind_next_anim(*binds4)

        self.face_simi_result_title.full_text = 'FACE SIMILARITY: '
        self.face_simi_result.full_text = '{:.2%}'.format(face_simi)
        title_anim4 = Animation(percentage=1., duration=tt, step=1./30, transition='linear')
        image_anim.on_complete = lambda _w: self._bind_next_anim((title_anim4, self.face_simi_result_title))
        score_anim4 = Animation(percentage=1., duration=st, step=1./30, transition='linear')
        title_anim4.on_complete = lambda _w: self._bind_next_anim((score_anim4, self.face_simi_result))

        self.syn_simi_result_title.full_text = 'SYNTHETIC SIMILARITY: '
        self.syn_simi_result.full_text = '{:.3%}'.format(syn_simi)
        title_anim5 = Animation(percentage=1., duration=tt2, step=1./30, transition='linear')
        score_anim4.on_complete = lambda _w: self._bind_next_anim((title_anim5, self.syn_simi_result_title))
        score_anim5 = Animation(percentage=1., duration=st2, step=1./30, transition='linear')
        title_anim5.on_complete = lambda _w: self._bind_next_anim((score_anim5, self.syn_simi_result))

        score_anim5.on_complete = lambda _w: self._update_btn_simi()

        for anim, image in binds:
            anim.start(image)
            self.anims.append((anim, image))
        #todo: better to change to monospace; need to change label behaviour to show letters at fix place



class CFSTApp(App):
    def build(self):
        return CFST()

if __name__ == '__main__':
    app = CFSTApp()
    app.run()
