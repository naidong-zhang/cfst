#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""FileChooserThumbView
====================

The FileChooserThumbView widget is similar to FileChooserIconView,
but if possible it shows a thumbnail instead of a normal icon.

Usage
-----

You can set some properties in order to control its performance:

* **thumbsize:** The size of the thumbnails. It defaults to 64d
"""

# Thanks to allan-simon for making the code more readable and less "spaghetti" :)

import os
import mimetypes
#(enable for debugging)
import traceback
from multiprocessing import Process, Pipe
from queue import Empty
from os.path import join, exists, dirname
import io

import numpy as np
import cv2

# from kivy.config import Config
# fonts = ['data/fonts/Roboto-Regular.ttf', 'NotoSansCJK-Regular.ttc']
# Config.set('kivy', 'default_font', fonts)

from kivy.app import App
from kivy.lang import Builder
from kivy.metrics import dp
from kivy.utils import QueryDict
from kivy.properties import StringProperty
from kivy.properties import DictProperty
from kivy.properties import ObjectProperty
from kivy.properties import BooleanProperty
from kivy.properties import NumericProperty
from kivy.uix.filechooser import FileChooserController
from kivy.core.image import Image as CoreImage
from kivy.clock import Clock

Builder.load_string('''
<FileChooserThumbView>:
    on_entry_added: stacklayout.add_widget(args[1])
    on_entries_cleared: stacklayout.clear_widgets()
    scrollview: scrollview

    ScrollView:
        id: scrollview
        pos: root.x - dp(50), root.y - dp(150)
        size: root.size
        size_hint: None, None
        do_scroll_x: False

        Scatter:
            do_rotation: False
            do_scale: False
            do_translation: False
            size_hint_y: None
            height: stacklayout.height
            StackLayout:
                id: stacklayout
                width: scrollview.width
                size_hint_y: None
                height: self.minimum_height
                spacing: '10dp'
                padding: '10dp'

[FileThumbEntry@Widget]:
    image: image
    locked: False
    path: ctx.path
    selected: self.path in ctx.controller().selection
    size_hint: None, None

    on_touch_down: self.collide_point(*args[1].pos) and ctx.controller().entry_touched(self, args[1])
    on_touch_up: self.collide_point(*args[1].pos) and ctx.controller().entry_released(self, args[1])
    size: ctx.controller().thumbsize + dp(52), ctx.controller().thumbsize + dp(52)

    canvas:
        Color:
            rgba: 1, 1, 1, 1 if self.selected else 0
        BorderImage:
            border: 8, 8, 8, 8
            pos: root.pos
            size: root.size
            source: 'atlas://data/images/defaulttheme/filechooser_selected'

    AsyncImage:
        id: image
        size: ctx.controller().thumbsize, ctx.controller().thumbsize
        pos: root.x + dp(24), root.y + dp(40)
    Label:
        text: ctx.name
        text_size: (ctx.controller().thumbsize, self.height)
        halign: 'center'
        shorten: True
        size: ctx.controller().thumbsize, '16dp'
        pos: root.center_x - self.width / 2, root.y + dp(16)

    ''')

DEFAULT_THEME = 'atlas://data/images/defaulttheme/'
FILE_ICON = DEFAULT_THEME + 'filechooser_file'
FOLDER_ICON = DEFAULT_THEME + 'filechooser_folder'

MAX_THUMB_SIZE = 256
DT = 0.2

def get_thumbimage(im, thumb_size=MAX_THUMB_SIZE):
    hw = np.array(im.shape[:2], dtype=np.uint)
    if np.any(hw > thumb_size):
        k = (hw // thumb_size + 1).max()
        hw = hw // k
        im = cv2.resize(im, tuple(hw[::-1]))
    return im


# test if the file is a supported picture
# file
def is_picture(mime, name):
    if mime is None:
        return False

    return "image/" in mime and (
            "jpeg" in mime or
            "jpg" in mime or
            "gif" in mime or
            "png" in mime
        ) and not name.endswith(".jpe")


def get_mime(fileName):
    try:
        mime = mimetypes.guess_type(fileName)[0]
        if mime is None:
            return ""
        return mime
    except TypeError:
        return ""

    return ""

def _need_process(ctx):
    'is image'
    if ctx.isdir:
        return False

    try:
        mime = get_mime(ctx.name)

        # if it's a picture, we don't need to do
        # any transormation
        if is_picture(mime, ctx.name):
            return True
    except:
        traceback.print_exc()
        return False

    return False

class FileChooserThumbView(FileChooserController):
    '''Implementation of :class:`FileChooserController` using an icon view
    with thumbnails.
    '''
    _ENTRY_TEMPLATE = 'FileThumbEntry'

    thumbsize = NumericProperty(dp(64))
    """The size of the thumbnails. It defaults to 64dp.
    """

    _thumbs = dict()
    scrollview = ObjectProperty(None)


    def __init__(self, **kwargs):
        super(FileChooserThumbView, self).__init__(**kwargs)
        app = App.get_running_app()
        self.thumbdir = app.user_data_dir
        if not exists(self.thumbdir):
            os.mkdir(self.thumbdir)

        app.bind(on_stop=self.save_cache)

        self.thumbnail_generator = ThreadedThumbnailGenerator()
        self.ev_get_ret = None

    def save_cache(self, *args):
        self.thumbnail_generator.clear()
        Clock.unschedule(self.ev_get_ret)
        # try:
        #     shutil.rmtree(self.thumbdir, ignore_errors=True)
        # except:
        #     traceback.print_exc()

    def _update_files(self, *args, **kwargs):
        self._thumbs.clear()
        self.thumbnail_generator.clear()
        if self.ev_get_ret is not None:
            Clock.unschedule(self.ev_get_ret)

        super(FileChooserThumbView, self)._update_files(*args, **kwargs)

        self.ev_get_ret = Clock.schedule_interval(self.get_ret, DT)
    def _create_entry_widget(self, ctx):
        # instantiate the widget
        widget = super(FileChooserThumbView, self)._create_entry_widget(ctx)

        kctx = QueryDict(ctx)
        # default icon
        is_image = _need_process(kctx)
        if not is_image:
            widget.image.source = FOLDER_ICON if kctx.isdir else FILE_ICON
        else:
            self._thumbs[kctx.path] = widget.image
            # schedule generation for later execution
            self.thumbnail_generator.put(kctx.path)
        return widget

    def get_ret(self, _dt):
        self.thumbnail_generator.run()
        while self._thumbs:
            try:
                ctx_path, data = self.thumbnail_generator.get()
            except Empty:
                return True

            try:
                image = self._thumbs.pop(ctx_path)
            except KeyError:
                return True

            if data is None:
                image.source = ctx_path
            else:
                img = CoreImage(data, ext='jpg')
                image._coreimage = img
                image._on_tex_change()





class ThreadedThumbnailGenerator(object):
    """
    Class that runs thumbnail generators in a another thread and
    asynchronously updates image widgets
    """
    def __init__(self):
        self.task_recv, self.task_send = Pipe(duplex=False)
        self.ret_recv, self.ret_send = Pipe(duplex=False)
        self.thread = None

    def put(self, fn):
        self.task_send.send(fn)

    def get(self):
        if self.ret_recv.poll():
            return self.ret_recv.recv()
        else:
            raise Empty

    def clear(self):
        while self.task_recv.poll():
            self.task_recv.recv()
        while self.ret_recv.poll():
            self.ret_recv.recv()

    def run(self):
        if self.thread is None or not self.thread.is_alive():
            self.thread = Process(target=self._loop, daemon=True)
            self.thread.start()

    def _loop(self):
        while True:
            # call user function that generates the thumbnail
            if self.task_recv.poll(timeout=DT):
                fn = self.task_recv.recv()
            else:
                break
            im = cv2.imread(fn)
            if im is None:
                self.ret_send.send((fn, None))
            else:
                im = get_thumbimage(im)
                ret, buf = cv2.imencode('.jpg', im)
                assert ret == True
                buf = buf.squeeze()
                data = io.BytesIO(buf)
                self.ret_send.send((fn, data))



if __name__ == "__main__":
    # from kivy.base import runTouchApp
    from kivy.uix.boxlayout import BoxLayout
    from kivy.uix.label import Label

    def setlabel(instance, value):
        instance.mylabel.text = "[b]Selected:[/b] {0}".format(value)

    class MyApp(App):
        def build(self):
            box = BoxLayout(orientation="vertical")
            fileChooser = FileChooserThumbView(thumbsize=128)
            fileChooser.rootpath = os.path.expanduser('~')
            label = Label(markup=True, size_hint_y=None)
            fileChooser.mylabel = label
            fileChooser.bind(selection=setlabel)

            box.add_widget(fileChooser)
            box.add_widget(label)

            return  box

    # runTouchApp(box)
    app = MyApp()
    app.run()
