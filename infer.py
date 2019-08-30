#!/usr/bin/env python3
# coding:utf-8

import os
import argparse
from time import time

import numpy as np
from sklearn.preprocessing import normalize
from skimage import transform
import cv2
import mxnet as mx

DEBUG = False

IMAGE_SIZE = (112,112)

def parse_args():
    parser = argparse.ArgumentParser(description='face alignment and recognition')
    # parser.add_argument('imagepath', help='face image filepath')
    parser.add_argument('--tf-configfile', default='/home/farid/git/code_db/src/models/opencv_face_detector.pbtxt', help='face detection config path')
    parser.add_argument('--tf-modelfile', default='/home/farid/git/code_db/src/models/opencv_face_detector_uint8.pb', help='face detection model path')
    parser.add_argument('--conf-threshold', default=0.7, type=float, help='face detection confidence threshold')
    parser.add_argument('--gpu', default=0, type=int, help='gpu id')
    parser.add_argument('--mtcnn-path', default='/home/farid/git/code_db/src/models/mtcnn-model', help='face alignment(mtcnn) model path')
    parser.add_argument('--rnet-threshold', default=0.0, type=float, help='MTCNN RNet threshold')
    parser.add_argument('--onet-threshold', default=0.2, type=float, help='MTCNN ONet threshold')
    parser.add_argument('--model-path', default='/home/farid/git/code_db/src/models/model-y1-test2/model,0', help='face recognition model path')
    return parser.parse_args()

def adjust_input(in_data):
    """
        adjust the input from (h, w, c) to ( 1, c, h, w) for network input

    Parameters:
    ----------
        in_data: numpy array of shape (h, w, c)
            input data
    Returns:
    -------
        out_data: numpy array of shape (1, c, h, w)
            reshaped array
    """
    if in_data.dtype is not np.dtype('float32'):
        out_data = in_data.astype(np.float32)
    else:
        out_data = in_data

    out_data = out_data.transpose((2,0,1))
    out_data = np.expand_dims(out_data, 0)
    out_data = (out_data - 127.5)*0.0078125
    return out_data

def nms(boxes, overlap_threshold, mode='Union'):
    """
        non max suppression

    Parameters:
    ----------
        box: numpy array n x 5
            input bbox array
        overlap_threshold: float number
            threshold of overlap
        mode: float number
            how to compute overlap ratio, 'Union' or 'Min'
    Returns:
    -------
        index array of the selected bbox
    """
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # if the bounding boxes integers, convert them to floats
    if boxes.dtype.kind == "i":
        boxes = boxes.astype(np.float32)

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1, y1, x2, y2, score = [boxes[:, i] for i in range(5)]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(score)

    # keep looping while some indexes still remain in the indexes list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        inter = w * h
        if mode == 'Min':
            overlap = inter / np.minimum(area[i], area[idxs[:last]])
        else:
            overlap = inter / (area[i] + area[idxs[:last]] - inter)
        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlap > overlap_threshold)[0])))

    return pick

def convert_to_square(bbox):
    """
        convert bbox to square

    Parameters:
    ----------
        bbox: numpy array , shape n x 5
            input bbox

    Returns:
    -------
        square bbox
    """
    square_bbox = bbox.copy()

    h = bbox[:, 3] - bbox[:, 1] + 1
    w = bbox[:, 2] - bbox[:, 0] + 1
    max_side = np.maximum(h,w)
    square_bbox[:, 0] = bbox[:, 0] + w*0.5 - max_side*0.5
    square_bbox[:, 1] = bbox[:, 1] + h*0.5 - max_side*0.5
    square_bbox[:, 2] = square_bbox[:, 0] + max_side - 1
    square_bbox[:, 3] = square_bbox[:, 1] + max_side - 1
    return square_bbox

def calibrate_box(bbox, reg):
    """
        calibrate bboxes

    Parameters:
    ----------
        bbox: numpy array, shape n x 5
            input bboxes
        reg:  numpy array, shape n x 4
            bboxex adjustment

    Returns:
    -------
        bboxes after refinement

    """
    w = bbox[:, 2] - bbox[:, 0] + 1
    w = np.expand_dims(w, 1)
    h = bbox[:, 3] - bbox[:, 1] + 1
    h = np.expand_dims(h, 1)
    reg_m = np.hstack([w, h, w, h])
    aug = reg_m * reg
    bbox[:, 0:4] = bbox[:, 0:4] + aug
    return bbox

def pad(bboxes, w, h):
    """
        pad the the bboxes, alse restrict the size of it

    Parameters:
    ----------
        bboxes: numpy array, n x 5
            input bboxes
        w: float number
            width of the input image
        h: float number
            height of the input image
    Returns :
    ------s
        dy, dx : numpy array, n x 1
            start point of the bbox in target image
        edy, edx : numpy array, n x 1
            end point of the bbox in target image
        y, x : numpy array, n x 1
            start point of the bbox in original image
        ex, ex : numpy array, n x 1
            end point of the bbox in original image
        tmph, tmpw: numpy array, n x 1
            height and width of the bbox

    """
    bboxes = bboxes.astype(np.int32)
    tmpw, tmph = bboxes[:, 2] - bboxes[:, 0] + 1,  bboxes[:, 3] - bboxes[:, 1] + 1
    num_box = bboxes.shape[0]

    dx , dy= np.zeros((num_box, ), dtype=np.int32), np.zeros((num_box, ), dtype=np.int32)
    edx, edy  = tmpw - 1, tmph - 1

    x, y, ex, ey = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]

    tmp_index = np.where(ex > w-1)
    edx[tmp_index] = tmpw[tmp_index] + w - 2 - ex[tmp_index]
    ex[tmp_index] = w - 1

    tmp_index = np.where(ey > h-1)
    edy[tmp_index] = tmph[tmp_index] + h - 2 - ey[tmp_index]
    ey[tmp_index] = h - 1

    tmp_index = np.where(x < 0)
    dx[tmp_index] = 0 - x[tmp_index]
    x[tmp_index] = 0

    tmp_index = np.where(y < 0)
    dy[tmp_index] = 0 - y[tmp_index]
    y[tmp_index] = 0

    return_list = [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph]
    return_list = [item.astype(np.int32) for item in return_list]

    return  return_list

class MtcnnDetector(object):
    """
        Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Neural Networks
        see https://github.com/kpzhang93/MTCNN_face_detection_alignment
        this is a mxnet version
    """
    def __init__(self,
                 model_folder,
                 ctx=mx.cpu()):
        """
            Initialize the detector

            Parameters:
            ----------
                model_folder : string
                    path for the models
                threshold : float number
                    detect threshold for 3 stages
        """
        # load 4 models from folder
        models = ['det1', 'det2', 'det3','det4']
        models = [ os.path.join(model_folder, f) for f in models]

        # self.PNets = []
        # workner_net = mx.model.FeedForward.load(models[0], 1, ctx=ctx)
        # self.PNets.append(workner_net)

        self.RNet = mx.model.FeedForward.load(models[1], 1, ctx=ctx)
        self.ONet = mx.model.FeedForward.load(models[2], 1, ctx=ctx)
        self.LNet = mx.model.FeedForward.load(models[3], 1, ctx=ctx)

    def detect_face(self, img, accurate_landmark=False,
                    rnet_threshold=0.7, onet_threshold=0.8,
                    det_bboxes=None):
        """
            detect face over img
        Parameters:
        ----------
            img: numpy array, bgr order of shape (h, w, 3)
                input image
        Retures:
        -------
            bboxes: numpy array, n x 5 (x1,y2,x2,y2,score)
                bboxes
            points: numpy array, n x 10 (x1, x2 ... x5, y1, y2 ..y5)
                landmarks
        """

        # check input
        height, width, _c = img.shape
        if det_bboxes is None:
            total_boxes = np.array( [ [0.0, 0.0, width, height, 0.9] ] ,dtype=np.float32)
        else:
            total_boxes = det_bboxes

        #############################################
        # second stage
        #############################################
        num_box = total_boxes.shape[0]
        assert num_box == 1

        # pad the bbox
        # [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = pad(total_boxes, width, height)
        # (3, 24, 24) is the input shape for RNet
        input_buf = np.zeros((num_box, 3, 24, 24), dtype=np.float32)

        for i in range(num_box):
            # tmp = np.zeros((tmph[i], tmpw[i], 3), dtype=np.uint8)            # tmp[dy[i]:edy[i]+1, dx[i]:edx[i]+1, :] = img[y[i]:ey[i]+1, x[i]:ex[i]+1, :]
            b = total_boxes[i].astype(np.int32)
            tmp = img[b[1]:b[3]+1, b[0]:b[2]+1, :]
            if tmp.size == 0:
                return None
            if DEBUG:
                # cv2.imwrite('s2-b{}-img.png'.format(i), img[y[i]:ey[i]+1, x[i]:ex[i]+1, :])
                cv2.imwrite('s2-b{}-tmp.png'.format(i), tmp)
            input_buf[i, :, :, :] = adjust_input(cv2.resize(tmp, (24, 24)))

        output = self.RNet.predict(input_buf)

        # filter the total_boxes with threshold
        passed = np.where(output[1][:, 1] > rnet_threshold)
        total_boxes = total_boxes[passed]

        if total_boxes.size == 0:
            return None

        total_boxes[:, 4] = output[1][passed, 1].reshape((-1,))
        reg = output[0][passed]

        # nms
        pick = nms(total_boxes, 0.7, 'Union')
        total_boxes = total_boxes[pick]
        total_boxes = calibrate_box(total_boxes, reg[pick])
        total_boxes = convert_to_square(total_boxes)
        total_boxes[:, 0:4] = np.round(total_boxes[:, 0:4])

        #############################################
        # third stage
        #############################################
        num_box = total_boxes.shape[0]

        # pad the bbox
        # [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = pad(total_boxes, width, height)
        # (3, 48, 48) is the input shape for ONet
        input_buf = np.zeros((num_box, 3, 48, 48), dtype=np.float32)

        for i in range(num_box):
            # tmp = np.zeros((tmph[i], tmpw[i], 3), dtype=np.float32)
            # tmp[dy[i]:edy[i]+1, dx[i]:edx[i]+1, :] = img[y[i]:ey[i]+1, x[i]:ex[i]+1, :]
            b = total_boxes[i].astype(np.int32)
            tmp = img[b[1]:b[3]+1, b[0]:b[2]+1, :]
            if tmp.size == 0:
                return None
            if DEBUG:
                # cv2.imwrite('s3-b{}-img.png'.format(i), img[y[i]:ey[i]+1, x[i]:ex[i]+1, :])
                cv2.imwrite('s3-b{}-tmp.png'.format(i), tmp)
            input_buf[i, :, :, :] = adjust_input(cv2.resize(tmp, (48, 48)))

        output = self.ONet.predict(input_buf)

        # filter the total_boxes with threshold
        passed = np.where(output[2][:, 1] > onet_threshold)
        total_boxes = total_boxes[passed]

        if total_boxes.size == 0:
            return None

        total_boxes[:, 4] = output[2][passed, 1].reshape((-1,))
        reg = output[1][passed]
        points = output[0][passed]

        # compute landmark points
        bbw = total_boxes[:, 2] - total_boxes[:, 0] + 1
        bbh = total_boxes[:, 3] - total_boxes[:, 1] + 1
        points[:, 0:5] = np.expand_dims(total_boxes[:, 0], 1) + np.expand_dims(bbw, 1) * points[:, 0:5]
        points[:, 5:10] = np.expand_dims(total_boxes[:, 1], 1) + np.expand_dims(bbh, 1) * points[:, 5:10]

        # nms
        total_boxes = calibrate_box(total_boxes, reg)
        pick = nms(total_boxes, 0.7, 'Min')
        total_boxes = total_boxes[pick]
        points = points[pick]

        if not accurate_landmark:
            return total_boxes, points

        #############################################
        # extended stage
        #############################################
        num_box = total_boxes.shape[0]
        patchw = np.maximum(total_boxes[:, 2]-total_boxes[:, 0]+1, total_boxes[:, 3]-total_boxes[:, 1]+1)
        patchw = np.round(patchw*0.25)

        # make it even
        patchw[np.where(np.mod(patchw,2) == 1)] += 1

        input_buf = np.zeros((num_box, 15, 24, 24), dtype=np.float32)
        for i in range(5):
            x, y = points[:, i], points[:, i+5]
            x, y = np.round(x-0.5*patchw), np.round(y-0.5*patchw)
            # [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = pad(np.vstack([x, y, x+patchw-1, y+patchw-1]).T,
 # width,
 # height)
            bs = np.vstack([x, y, x+patchw-1, y+patchw-1]).T
            for j in range(num_box):
                # tmpim = np.zeros((tmpw[j], tmpw[j], 3), dtype=np.float32)
                # tmpim[dy[j]:edy[j]+1, dx[j]:edx[j]+1, :] = img[y[j]:ey[j]+1, x[j]:ex[j]+1, :]
                b = bs[j].astype(np.int32)
                tmpim = img[b[1]:b[3]+1, b[0]:b[2]+1, :]
                if tmpim.size == 0:
                    return None
                if DEBUG:
                    # cv2.imwrite('s4-k{}-b{}-img.png'.format(i,j), img[y[j]:ey[j]+1, x[j]:ex[j]+1, :])
                    cv2.imwrite('s4-k{}-b{}-tmp.png'.format(i,j), tmpim)
                input_buf[j, i*3:i*3+3, :, :] = adjust_input(cv2.resize(tmpim, (24, 24)))

        output = self.LNet.predict(input_buf)

        pointx = np.zeros((num_box, 5))
        pointy = np.zeros((num_box, 5))

        for k in range(5):
            # do not make a large movement
            tmp_index = np.where(np.abs(output[k]-0.5) > 0.35)
            output[k][tmp_index[0]] = 0.5

            pointx[:, k] = np.round(points[:, k] - 0.5*patchw) + output[k][:, 0]*patchw
            pointy[:, k] = np.round(points[:, k+5] - 0.5*patchw) + output[k][:, 1]*patchw

        points = np.hstack([pointx, pointy])
        # points = points.astype(np.int32)

        return total_boxes, points


def get_model(ctx, model_str, layer):
    _vec = model_str.split(',')
    assert len(_vec)==2
    prefix = _vec[0]
    epoch = int(_vec[1])
    print('loading recog model',prefix, epoch)
    sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
    all_layers = sym.get_internals()
    sym = all_layers[layer+'_output']
    model = mx.mod.Module(symbol=sym, context=ctx, label_names=None)
    #model.bind(data_shapes=[('data', (args.batch_size, 3, image_size[0], image_size[1]))], label_shapes=[('softmax_label', (args.batch_size,))])
    model.bind(data_shapes=[('data', (1, 3, IMAGE_SIZE[0], IMAGE_SIZE[1]))])
    model.set_params(arg_params, aux_params)
    return model

def timefunc(func):
    count = 0
    total_time = 0
    def wrapper(*args, **kwargs):
        nonlocal total_time
        nonlocal count

        t0 = time()

        det = func(*args, **kwargs)

        if det is None:
            return det
        elif isinstance(det, tuple):
            if len(det) > 0 and det[0] is None:
                return det

        dt = time() - t0
        total_time += dt
        count += 1
        if count % 1000 == 0:
            print('mean run time of {}:{:.2f}ms'.format(func.__name__, 1000 * total_time / count))
        return det
    return wrapper

@timefunc
def recog_preprocess(img, bbox=None, landmark=None):
    is_aligned = False
    M = None
    if landmark is not None:
        src = np.array([
                [38.2946, 51.6963],
                [73.5318, 51.5014],
                [56.0252, 71.7366],
                [41.5493, 92.3655],
                [70.7299, 92.2041] ], dtype=np.float32 )
        dst = landmark.astype(np.float32)

        tform = transform.SimilarityTransform()
        tform.estimate(dst, src)
        M = tform.params[0:2,:]

    if M is None:
        if bbox is None: #use center crop
            det = np.zeros(4, dtype=np.int32)
            det[0] = int(img.shape[1]*0.0625)
            det[1] = int(img.shape[0]*0.0625)
            det[2] = img.shape[1] - det[0]
            det[3] = img.shape[0] - det[1]
        else:
            det = bbox
        # MARGIN = 44
        x0,y0,x1,y1 = det
        h = y1 - y0 + 1
        w = x1 - x0 + 1
        max_side = max(h, w)
        max_side += max_side // 10
        x0 = x0 + w // 2 - max_side // 2
        y0 = y0 + h // 2 - max_side // 2
        x1 = x0 + max_side - 1
        y1 = y0 + max_side - 1
        bb = np.zeros(4, dtype=np.int32)
        # bb[0] = np.maximum(det[0]-MARGIN/2, 0)
        # bb[1] = np.maximum(det[1]-MARGIN/2, 0)
        # bb[2] = np.minimum(det[2]+MARGIN/2, img.shape[1])
        # bb[3] = np.minimum(det[3]+MARGIN/2, img.shape[0])
        bb[0] = max(x0, 0)
        bb[1] = max(y0, 0)
        bb[2] = min(x1, img.shape[1])
        bb[3] = min(y1, img.shape[0])
        ret = img[bb[1]:bb[3],bb[0]:bb[2],:]
        ret = cv2.resize(ret, (IMAGE_SIZE[1], IMAGE_SIZE[0]))
        return is_aligned, ret
    else: #do align using landmark
        warped = cv2.warpAffine(img, M, (IMAGE_SIZE[1],IMAGE_SIZE[0]), borderValue=0.0)
        is_aligned = True
        return is_aligned, warped

@timefunc
def forward0(im, rnet_threshold, onet_threshold, conf_threshold=0.2):
    det_bboxes = det(tf_net, im, conf_threshold=conf_threshold)
    try:
        ret = aligner.get_box_kpoints(im, det_bboxes=det_bboxes,
                                        rnet_threshold=rnet_threshold,
                                        onet_threshold=onet_threshold)
    except Exception as e:
        print('error on:{}, when run: aligner.get_box_kpoints(), message:{}'.format(fp, e))
        return None
    if ret is None:
        print('error on:{}, cannot find box or kpoints'.format(fp))
        return None
    box, kpoints = ret

    if len(box) != 1:
        print('error on:{}, find {} of boxes'.format(fp, len(box)))
        return None

    # im_show = im.copy()
    # for b, p in zip(box, kpoints):
    #     im_show = cv2.rectangle(im_show, (b[0],b[1]), (b[2],b[3]), (0,255,0), 1)
    #     assert p.shape == (5,2)
    #     for px, py in p:
    #         im_show = cv2.circle(im_show, (int(px),int(py)), 2, (0,255,0), 1)
    # cv2.imwrite('box-kpoints.png', im_show)
    for i, (b, p) in enumerate(zip(box, kpoints)):
        aligned_face = face_preprocess(im, box[i], kpoints[i])
    #     cv2.imwrite('aligned-{}.png'.format(i), aligned_face)
    feature = recognizer.get_feature(aligned_face)
    return feature

def cal_vgg2_features_and_save():
    # import pdb
    from easydict import EasyDict
    import pickle

    tf_configfile = '/home/zhangnd/git/insightface1/deploy/opencv_face_detector.pbtxt'
    tf_modelfile = '/home/zhangnd/git/insightface1/deploy/opencv_face_detector_uint8.pb'
    tf_net = cv2.dnn.readNetFromTensorflow(tf_modelfile, tf_configfile)

    args = parse_args()
    aligner = Aligner(args)
    recognizer = Recognizer(args)

    # im = cv2.imread(args.imagepath, cv2.IMREAD_COLOR)
    BASE_DIR = '/media/zhangnd/DATA/data/VGG-FACE2/test'
    CONF_THRESHOLD = 0.9
    folders = os.listdir(BASE_DIR)
    results = list()
    num = len(folders)
    for i, folder in enumerate(folders, 1):
        print('calculating features:{}/{}'.format(i, num))
        dp = os.path.join(BASE_DIR, folder)
        for fn in os.listdir(dp):
            fp = os.path.join(dp, fn)
            im = cv2.imread(fp, cv2.IMREAD_COLOR)
            assert im is not None
            feature = forward0(im, rnet_threshold=args.rnet_threshold,
                                  onet_threshold=args.onet_threshold,
                                  conf_threshold=CONF_THRESHOLD)
            item = EasyDict(feature=feature, fp=fp, folder=folder, fn=fn)
            results.append(item)

    with open('vgg2-r34.pkl', 'wb') as f:
        pickle.dump(results, f)




class Detector(object):
    def __init__(self, tf_configfile, tf_modelfile):
        tf_net = cv2.dnn.readNetFromTensorflow(tf_modelfile, tf_configfile)
        self.net = tf_net

    @timefunc
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
        det_boxes[:,:4] = det[:,3:]
        det_boxes[:,4] = det[:,2]
        # det_boxes.shape is (n,5)
        return det_boxes

class Aligner(object):
    def __init__(self, mtcnn_path, gpu):
        self.detector = MtcnnDetector(model_folder=mtcnn_path, ctx=mx.gpu(int(gpu)))

    @timefunc
    def get_box_kpoints(self, face_img,
                            rnet_threshold,
                            onet_threshold,
                            det_box=None):
        ret = self.detector.detect_face(face_img, accurate_landmark=True,
                                        rnet_threshold=rnet_threshold,
                                        onet_threshold=onet_threshold,
                                        det_bboxes=det_box)
        if ret is None:
            return None
        bbox, points = ret
        if bbox.shape[0] == 0:
            return None
        bbox = bbox[0,0:4]
        points = points[0,:].reshape((2,5)).T
        # points = np.stack([ps.reshape((2,5)).T for ps in points])
        return bbox, points

class Recognizer(object):
    def __init__(self, model_path, gpu):
        layer = 'fc1'
        ctx = mx.gpu(int(gpu))
        _vec = model_path.split(',')
        assert len(_vec)==2
        prefix = _vec[0]
        epoch = int(_vec[1])
        print('loading recog model',prefix, epoch)
        sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
        all_layers = sym.get_internals()
        sym = all_layers[layer+'_output']
        model = mx.mod.Module(symbol=sym, context=ctx, label_names=None)
        #model.bind(data_shapes=[('data', (args.batch_size, 3, image_size[0], image_size[1]))], label_shapes=[('softmax_label', (args.batch_size,))])
        model.bind(data_shapes=[('data', (1, 3, IMAGE_SIZE[0], IMAGE_SIZE[1]))])
        model.set_params(arg_params, aux_params)
        self.model = model

    @timefunc
    def get_feature(self, aligned):
        aligned = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
        aligned = np.transpose(aligned, (2,0,1))
        input_blob = np.expand_dims(aligned, axis=0)
        data = mx.nd.array(input_blob)
        db = mx.io.DataBatch(data=(data,))
        self.model.forward(db, is_train=False)
        embedding = self.model.get_outputs()[0].asnumpy()
        embedding = normalize(embedding).flatten()
        return embedding

@timefunc
def forward(im, conf_threshold, rnet_threshold, onet_threshold):
    # detect
    det_boxes = detector.detect_faces(im, conf_threshold=conf_threshold)
    if DEBUG:
        print(det_boxes.shape)
        print(det_boxes)
    # det_boxes need no expand, just expand image
    # select arbitrary(confidence?) box, shape is (1,5)
    if det_boxes.shape[0] == 0:
        print('detect no valid box.')
        return None, None
    det_box = det_boxes[:1]
    if DEBUG:
        im1 = im.copy()
        x1,y1,x2,y2,conf = det_box[0]
        im1 = cv2.rectangle(im1, (x1,y1), (x2,y2), (0,255,0), 1)
        cv2.imwrite('im-det_box.png', im1)
        x1,y1,x2,y2 = map(int, (x1,y1,x2,y2))
        cv2.imwrite('det_box.png', im[y1:y2+1,x1:x2+1])
    # align
    ret = aligner.get_box_kpoints(im,
                                rnet_threshold=rnet_threshold,
                                onet_threshold=onet_threshold,
                                det_box=det_box)
    if ret is not None:
        box, kpoints = ret
        if DEBUG:
            print(box.shape, kpoints.shape)
            print(box)
            print(kpoints)
        if DEBUG:
            im1 = im.copy()
            x1,y1,x2,y2 = box
            im1 = cv2.rectangle(im1, (x1,y1), (x2,y2), (0,255,0), 1)
            assert kpoints.shape == (5,2)
            for px, py in kpoints:
                im1 = cv2.circle(im1, (int(px),int(py)), 2, (0,255,0), 1)
            cv2.imwrite('im-box-kpoints.png', im1)
    else:
        box, kpoints = det_box[0][:4], None
        print('alignment failed. kpoints is None.')
    # recog
    is_aligned, aligned_face = recog_preprocess(im, box, kpoints)
    if DEBUG:
        cv2.imwrite('aligned_face-{:d}.png'.format(is_aligned), aligned_face)
    feature = recognizer.get_feature(aligned_face)
    if DEBUG:
        print(feature.shape)
    return is_aligned, feature

if __name__ == '__main__':
    args = parse_args()
    # init
    # detect
    conf_threshold = args.conf_threshold
    detector = Detector(args.tf_configfile, args.tf_modelfile)
    # align
    rnet_threshold = args.rnet_threshold
    onet_threshold = args.onet_threshold
    aligner = Aligner(args.mtcnn_path, args.gpu)
    # recog
    recognizer = Recognizer(args.model_path, args.gpu)

    DEBUG = True
    if DEBUG:
        # read image
        im = cv2.imread('/home/farid/git/code_db/src/WANZ777.jpg', cv2.IMREAD_COLOR)
        assert im is not None
        is_aligned, feature = forward(im, conf_threshold, rnet_threshold, onet_threshold)
    else:
        exemptions = ['/media/zhangnd/DATA/data/VGG-FACE2/test/n000001/0029_01.jpg',
                    '/media/zhangnd/DATA/data/VGG-FACE2/test/n000001/0188_01.jpg',
                    '/media/zhangnd/DATA/data/VGG-FACE2/test/n000624/0436_01.jpg',
                    '/media/zhangnd/DATA/data/VGG-FACE2/test/n000001/0038_01.jpg']
        BASE_DIR = '/media/zhangnd/DATA/data/VGG-FACE2/test'
        folders = os.listdir(BASE_DIR)
        folders.sort()
        for folder in folders:
            dp = os.path.join(BASE_DIR, folder)
            files = os.listdir(dp)
            files.sort()
            for fn in files:
                fp = os.path.join(dp, fn)
                im = cv2.imread(fp, cv2.IMREAD_COLOR)
                assert im is not None
                is_aligned, feature = forward(im, conf_threshold, rnet_threshold, onet_threshold)
                if not is_aligned and fp not in exemptions:
                    raise ValueError(fp)
        print('done')
