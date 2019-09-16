#!/usr/bin/env python3
#coding:utf-8
import os
import base64
import json

import numpy as np
import cv2

import kivy
kivy.require('1.10.1')
from kivy.app import App
# from kivy.core.window import Window
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.popup import Popup
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.label import Label
from kivy.properties import ObjectProperty, BooleanProperty
from kivy.graphics.texture import Texture
from kivy.network.urlrequest import UrlRequest


SERVER_URL = 'http://127.0.0.1:5000/api/v0'


class LoadDialog(FloatLayout):
    load = ObjectProperty(None)
    cancel = ObjectProperty(None)
    is_me = BooleanProperty()

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
        h,w,_c = self.portrait_rgb.shape
        texture = Texture.create(size=(w,h))

        im = self.portrait_rgb.copy()
        im = self._bright(im, -50)
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
            im[y0:y1+1, x0:x1+1] = self._bright(region, brightness)

        # for debug
        # im = cv2.circle(im, self.touch_pos_in_cv2, 5, (255,0,0), -1)
        texture.blit_buffer(im.ravel(), colorfmt='rgb', bufferfmt='ubyte', mipmap_generation=False)
        texture.flip_vertical()
        self.portrait.texture = texture

    @staticmethod
    def _bright(im_rgb, v):
        if v > 0:
            hsv = cv2.cvtColor(im_rgb, cv2.COLOR_RGB2HSV)
            hsv[..., 2] = np.where(hsv[..., 2] < (255 - v), hsv[..., 2] + v, 255)
            im_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        else:
            # dark all
            v = -v
            hsv = cv2.cvtColor(im_rgb, cv2.COLOR_RGB2HSV)
            hsv[..., 2] = np.where(hsv[..., 2] > v, hsv[..., 2] - v, 0)
            im_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        return im_rgb



class CFST(BoxLayout):
    face_me = ObjectProperty(None)
    face_spouse = ObjectProperty(None)
    btn_run = ObjectProperty(None)

    HEADERS = {'Content-type': 'application/json'}

    def __init__(self, **kwargs):
        super(CFST, self).__init__(**kwargs)

        self.popup = None
        self.portrait_rgb = None
        self.got_me = False
        self.got_spouse = False

    def input_my_photo(self):
        self._input_photo(is_me=True)

    def input_spouse_photo(self):
        self._input_photo(is_me=False)

    def _input_photo(self, is_me):
        content = LoadDialog(load=self._detect, cancel=self._dismiss_popup, is_me=is_me)
        self.popup = Popup(title='INPUT MY PHOTO', content=content, size_hint=(0.9, 0.9))
        self.popup.open()

    def _dismiss_popup(self):
        self.popup.dismiss()

    def _detect(self, path, filenames, is_me):
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
                            on_success=self.req_detect)

    def req_detect(self, req, result):
        bboxes = result.get('bboxes')
        assert bboxes is not None

        content = SelectDialog(portrait_rgb=self.portrait_rgb, bboxes=bboxes, select=self._align, cancel=self._dismiss_popup)
        self.popup = Popup(title='SELECT ONE FACE', content=content, size_hint=(0.9, 0.9))
        self.popup.open()

    def _align(self, bbox):
        print(bbox)
        self._dismiss_popup()





    def debug(self):
        # r = requests.post(SERVER_URL + '/detect/', json={'portrait': b64_portrait_raw})
        # receive bboxes
        # assert r.status_code == 200
        # bboxes = r.json().get('bboxes')
        # assert len(bboxes) == 1
        # if len(bboxes) == 1:
        #     bbox = bboxes[0]
        # draw bboxes
        # select bbox
        # send bbox
        r = requests.post(SERVER_URL + '/align/', json={'portrait': b64_portrait_raw,
                                                        'bbox': bbox})
        # receive face
        assert r.status_code == 200
        b64_face_raw = r.json().get('face')
        # show face
        face_raw = base64.b64decode(b64_face_raw)
        raw = np.frombuffer(face_raw, dtype=np.uint8)
        im = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        assert im is not None
        im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        w,h = 112,112
        texture = Texture.create(size=(w,h))
        texture.blit_buffer(im_rgb.ravel(), colorfmt='rgb', bufferfmt='ubyte', mipmap_generation=False)
        texture.flip_vertical()

        # show face
        if is_me:
            self.face_me.texture = texture
            self.got_me = True
        else:
            self.face_spouse.texture = texture
            self.got_spouse = True
        self._refresh_btn_run()


    def _refresh_btn_run(self):
        if self.got_me and self.got_spouse:
            self.btn_run.disabled = False
        else:
            self.btn_run.disabled = True


    def run(self):
        pass

class CFSTApp(App):
    def build(self):
        return CFST()

if __name__ == '__main__':
    app = CFSTApp()
    app.run()
