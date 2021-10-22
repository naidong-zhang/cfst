import os
import io
import numpy as np
from sklearn.preprocessing import normalize
from skimage import transform
from skimage.feature import local_binary_pattern
import cv2
from openvino.inference_engine import IECore


IMAGE_SIZE = (112,112)

# TF_CONFIGFILE = 'models/opencv_face_detector.pbtxt'
# TF_MODELFILE = 'models/opencv_face_detector_uint8.pb'
# MTCNN_PATH = 'models/mtcnn-model'
RNET_THRESHOLD = 0.0
ONET_THRESHOLD = 0.2
# MODEL_PATH = 'models/model-y1-test2/model-0000'
FEATURE_DIMENSION = 128




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




models = None

ie = IECore()
ie.set_config({'EXCLUSIVE_ASYNC_REQUESTS':'YES'}, device_name='CPU')
ie.set_config({'CPU_THROUGHPUT_STREAMS':'CPU_THROUGHPUT_AUTO'}, device_name='CPU')

def create_singleton_models(tf_configfile, tf_modelfile, mtcnn_path, model_path):
    global models
    if models is None:
        class Models(object):
            def __init__(self):
                self.detector = Detector(tf_configfile, tf_modelfile)
                self.aligner = Aligner(mtcnn_path)
                self.recognizer = Recognizer(model_path)

            def detect_faces(self, im, w, h, conf_threshold=0.3):
                det_boxes = self.detector.detect_faces(im, w, h, conf_threshold)
                bboxes = det_boxes[:,:4].astype(np.int)
                return bboxes

            def align_face(self, patch, box):
                # already patched on client side
                # patch, box = cal_face_patch(im, box)
                aligned_face, _padded_face = cal_aligned_face(self.aligner, patch, box)
                return aligned_face

            def cal_face_simi(self, face0_im, face1_im):
                face0_feature = cal_feature(self.recognizer, face0_im)
                face1_feature = cal_feature(self.recognizer, face1_im)
                face_simi = self._cal_face_similarity(face0_feature, face1_feature)
                return face_simi

            def cal_lbp_simi(self, face0_im, face1_im):
                'only for aligned face'
                mouth0_lbp, nose0_lbp, eye00_lbp, eye01_lbp = self._cal_lbp_features(face0_im)
                mouth1_lbp, nose1_lbp, eye10_lbp, eye11_lbp = self._cal_lbp_features(face1_im)

                mouth_simi = self._cal_lbp_similarity(mouth0_lbp, mouth1_lbp)
                nose_simi = self._cal_lbp_similarity(nose0_lbp, nose1_lbp)
                eye00_simi = self._cal_lbp_similarity(eye00_lbp, eye10_lbp)
                eye11_simi = self._cal_lbp_similarity(eye01_lbp, eye11_lbp)
                eye01_simi = self._cal_lbp_similarity(eye00_lbp, eye11_lbp)
                eye10_simi = self._cal_lbp_similarity(eye01_lbp, eye10_lbp)
                eye_simi = max(eye00_simi, eye11_simi, eye01_simi, eye10_simi)

                return mouth_simi, nose_simi, eye_simi

            @staticmethod
            def _cal_lbp_features(aligned_face):
                mouth_box = aligned_face[MOUTH_BY0:MOUTH_BY1+1, MOUTH_BX0:MOUTH_BX1+1]
                eye0_box = aligned_face[EYE0_BY0:EYE0_BY1+1, EYE0_BX0:EYE0_BX1+1]
                eye1_box = aligned_face[EYE1_BY0:EYE1_BY1+1, EYE1_BX0:EYE1_BX1+1]
                nose_box = aligned_face[NOSE_BY0:NOSE_BY1+1, NOSE_BX0:NOSE_BX1+1]

                mouth_feature = get_lbp_feature(mouth_box)
                nose_feature = get_lbp_feature(nose_box)
                eye0_feature = get_lbp_feature(eye0_box)
                eye1_feature = get_lbp_feature(eye1_box)

                return mouth_feature, nose_feature, eye0_feature, eye1_feature

            @staticmethod
            def _cal_face_similarity(face0_feature, face1_feature):
                face_similarity = np.dot(face0_feature, face1_feature) * 0.5 + 0.5
                face_similarity = postprocess(face_similarity)
                return face_similarity

            @staticmethod
            def _cal_lbp_similarity(lbp0, lbp1):
                lbp_similarity = cal_lbp_simi(lbp0, lbp1)
                lbp_similarity = postprocess(lbp_similarity)
                return lbp_similarity


        models = Models()
    return models


def adjust_input(out_data):
    out_data = out_data.transpose((2,0,1))
    out_data = np.expand_dims(out_data, 0)
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
                 model_folder):
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
        models = ['det1', 'det2-0001', 'det3-0001','det4-0001']
        models = [ os.path.join(model_folder, f) for f in models]

        # self.PNets = []
        # workner_net = mx.model.FeedForward.load(models[0], 1, ctx=ctx)
        # self.PNets.append(workner_net)

        self.RNet = ie.read_network(model='{}.xml'.format(models[1]), weights='{}.bin'.format(models[1]))
        self.eRNet = ie.load_network(network=self.RNet, device_name='CPU')
        self.ONet = ie.read_network(model='{}.xml'.format(models[2]), weights='{}.bin'.format(models[2]))
        self.eONet = ie.load_network(network=self.ONet, device_name='CPU')
        self.LNet = ie.read_network(model='{}.xml'.format(models[3]), weights='{}.bin'.format(models[3]))
        self.eLNet = ie.load_network(network=self.LNet, device_name='CPU')

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
        input_buf = np.zeros((num_box, 3, 24, 24), dtype=np.uint8)

        for i in range(num_box):
            # tmp = np.zeros((tmph[i], tmpw[i], 3), dtype=np.uint8)            # tmp[dy[i]:edy[i]+1, dx[i]:edx[i]+1, :] = img[y[i]:ey[i]+1, x[i]:ex[i]+1, :]
            b = total_boxes[i].astype(np.int32)
            tmp = img[b[1]:b[3]+1, b[0]:b[2]+1, :]
            if tmp.size == 0:
                return None
            input_buf[i, :, :, :] = adjust_input(cv2.resize(tmp, (24, 24)))

        if input_buf.shape[0] != self.RNet.batch_size:
            assert ValueError('RNet batch size:{}'.format(input_buf.shape[0]))
            self.RNet.batch_size = input_buf.shape[0]
            self.eRNet = ie.load_network(network=self.RNet, device_name='CPU')
        output = self.eRNet.infer({'data':input_buf})

        # filter the total_boxes with threshold
        passed = np.where(output['prob1'][:, 1] > rnet_threshold)
        total_boxes = total_boxes[passed]

        if total_boxes.size == 0:
            return None

        total_boxes[:, 4] = output['prob1'][passed, 1].reshape((-1,))
        reg = output['conv5_2'][passed]

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
        input_buf = np.zeros((num_box, 3, 48, 48), dtype=np.uint8)

        for i in range(num_box):
            # tmp = np.zeros((tmph[i], tmpw[i], 3), dtype=np.float32)
            # tmp[dy[i]:edy[i]+1, dx[i]:edx[i]+1, :] = img[y[i]:ey[i]+1, x[i]:ex[i]+1, :]
            b = total_boxes[i].astype(np.int32)
            tmp = img[b[1]:b[3]+1, b[0]:b[2]+1, :]
            if tmp.size == 0:
                return None
            input_buf[i, :, :, :] = adjust_input(cv2.resize(tmp, (48, 48)))

        if input_buf.shape[0] != self.ONet.batch_size:
            self.ONet.batch_size = input_buf.shape[0]
            print('\033[33m reload ONet batch size:{}  \033[0m'.format(self.ONet.batch_size))
            self.eONet = ie.load_network(network=self.ONet, device_name='CPU')
        output = self.eONet.infer({'data':input_buf})

        # filter the total_boxes with threshold
        passed = np.where(output['prob1'][:, 1] > onet_threshold)
        total_boxes = total_boxes[passed]

        if total_boxes.size == 0:
            return None

        total_boxes[:, 4] = output['prob1'][passed, 1].reshape((-1,))
        reg = output['conv6_2'][passed]
        points = output['conv6_3'][passed]

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

        input_buf = np.zeros((num_box, 15, 24, 24), dtype=np.uint8)
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
                input_buf[j, i*3:i*3+3, :, :] = adjust_input(cv2.resize(tmpim, (24, 24)))

        if input_buf.shape[0] != self.LNet.batch_size:
            self.LNet.batch_size = input_buf.shape[0]
            print('\033[33m reload LNet batch size:{}  \033[0m'.format(self.LNet.batch_size))
            self.eLNet = ie.load_network(network=self.LNet, device_name='CPU')
        output = self.eLNet.infer({'data':input_buf})

        pointx = np.zeros((num_box, 5))
        pointy = np.zeros((num_box, 5))

        for k in range(5):
            # do not make a large movement
            key = 'fc5_{}'.format(k+1)
            tmp_index = np.where(np.abs(output[key]-0.5) > 0.35)
            output[key][tmp_index[0]] = 0.5

            pointx[:, k] = np.round(points[:, k] - 0.5*patchw) + output[key][:, 0]*patchw
            pointy[:, k] = np.round(points[:, k+5] - 0.5*patchw) + output[key][:, 1]*patchw

        points = np.hstack([pointx, pointy])
        # points = points.astype(np.int32)

        return total_boxes, points

class Detector(object):
    def __init__(self, tf_configfile, tf_modelfile):
        tf_net = cv2.dnn.readNetFromTensorflow(tf_modelfile, tf_configfile)
        self.net = tf_net

    def detect_faces(self, im, w, h, conf_threshold=0.2):
        blob = cv2.dnn.blobFromImage(im, 1.0, (300, 300), [104, 117, 123], False, False)
        self.net.setInput(blob)
        det = self.net.forward()
        idx = np.where( det[:,:,:,2] > conf_threshold)
        det = det[idx]
        a = np.logical_and(det[:,3] < 1.0, det[:,5] > 0.0)
        b = np.logical_and(det[:,4] < 1.0, det[:,6] > 0.0)
        idx = np.where(np.logical_and(a, b))
        det = det[idx]
        det[:,3::2] *= w
        det[:,4::2] *= h
        det_boxes = np.empty(det[:,2:].shape, dtype=np.float32)
        # round coordinates
        det_boxes[:,:4] = np.round(det[:,3:])
        det_boxes[:,4] = det[:,2]
        # make x0, y0 > 0
        det_boxes[:,:2] = np.maximum(det_boxes[:,:2], 0)
        # det_boxes.shape is (n,5)
        return det_boxes

class Aligner(object):
    def __init__(self, mtcnn_path):
        self.detector = MtcnnDetector(model_folder=mtcnn_path)

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
        return bbox, points

class Recognizer(object):
    def __init__(self, model_path):
        print('loading recog model', model_path)
        self.net = ie.read_network(model='{}.xml'.format(model_path), weights='{}.bin'.format(model_path))
        self.enet = ie.load_network(network=self.net, device_name='CPU')

    def get_feature(self, aligned):
        input_blob = np.transpose(aligned, (2,0,1))
        input_blob = np.expand_dims(input_blob, axis=0)
        output = self.enet.infer({'data':input_blob})
        embedding = output['pre_fc1/Fused_Add_']
        embedding = normalize(embedding).flatten()
        return embedding

def recog_preprocess(img, bbox=None, landmark=None):
    is_aligned = False
    M = None
    if landmark is not None:
        # src = np.array([
        #         [38.2946, 51.6963],
        #         [73.5318, 51.5014],
        #         [56.0252, 71.7366],
        #         [41.5493, 92.3655],
        #         [70.7299, 92.2041] ], dtype=np.float32 )
        src = KPOINTS
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


def cal_face_patch(image, box):
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

def cal_aligned_face(aligner, patch, box, padded_too=False):
    aligned_face, padded_face = None, None

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

def cal_feature(recognizer, face_image):
    feature = recognizer.get_feature(face_image)
    return feature




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
    simi = simi ** 0.7
    return simi









