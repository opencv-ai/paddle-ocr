# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import numpy as np
import time
from utils import create_operators, transform
from posptrocessor import DBPostProcess
import onnxruntime as ort


class TextDetector(object):
    def __init__(self, args):
        self.args = args

        if not os.path.exists(args.det_model_dir):
            raise ValueError("not find model file path {}".format(
                args.det_model_dir))
        self.predictor = ort.InferenceSession(args.det_model_dir)
        self.input_tensor = self.predictor.get_inputs()[0]

        img_h, img_w = self.input_tensor.shape[2:]

        pre_process_list = [
            {
                'DetResizeForTest': {
                    'image_shape': [img_h, img_w],
                    'keep_ratio': True
                }
            }, {
                'PadImage': {
                    'image_shape': [img_h, img_w],
                    'pad_value': 0
                }
            }, {
                'NormalizeImage': {
                    'std': [0.229, 0.224, 0.225],
                    'mean': [0.485, 0.456, 0.406],
                    'scale': '1./255.',
                    'order': 'hwc'
                }
            }, {
                'ToCHWImage': None
            }, {
                'KeepKeys': {
                    'keep_keys': ['image', 'shape', 'pads']
                }
            }]
        postprocess_params = dict()
        postprocess_params["thresh"] = args.det_db_thresh
        postprocess_params["box_thresh"] = args.det_db_box_thresh
        postprocess_params["max_candidates"] = 1000
        postprocess_params["unclip_ratio"] = args.det_db_unclip_ratio
        postprocess_params["use_dilation"] = args.use_dilation

        self.postprocess_op = DBPostProcess(**postprocess_params)

        self.preprocess_op = create_operators(pre_process_list)

    @staticmethod
    def order_points_clockwise(pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        tmp = np.delete(pts, (np.argmin(s), np.argmax(s)), axis=0)
        diff = np.diff(np.array(tmp), axis=1)
        rect[1] = tmp[np.argmin(diff)]
        rect[3] = tmp[np.argmax(diff)]
        return rect

    @staticmethod
    def clip_det_res(points, img_height, img_width):
        for pno in range(points.shape[0]):
            points[pno, 0] = int(min(max(points[pno, 0], 0), img_width - 1))
            points[pno, 1] = int(min(max(points[pno, 1], 0), img_height - 1))
        return points

    def filter_tag_det_res(self, dt_boxes, image_shape):
        img_height, img_width = image_shape[0:2]
        dt_boxes_new = []
        for box in dt_boxes:
            if type(box) is list:
                box = np.array(box)
            box = self.order_points_clockwise(box)
            box = self.clip_det_res(box, img_height, img_width)
            rect_width = int(np.linalg.norm(box[0] - box[1]))
            rect_height = int(np.linalg.norm(box[0] - box[3]))
            if rect_width <= 3 or rect_height <= 3:
                continue
            dt_boxes_new.append(box)
        dt_boxes = np.array(dt_boxes_new)
        return dt_boxes

    def __call__(self, img):
        ori_im = img.copy()
        data = {'image': img}

        st = time.time()

        data = transform(data, self.preprocess_op)
        img, shape_list, pads = data

        if img is None:
            return None, 0
        img = np.expand_dims(img, axis=0)
        shape_list = np.expand_dims(shape_list, axis=0)
        img = img.copy()

        input_dict = dict()
        input_dict[self.input_tensor.name] = img
        outputs = self.predictor.run(None, input_dict)

        preds = dict()
        result_map = outputs[0]
        _, _, h, w = result_map.shape
        preds['maps'] = result_map[:, :, pads[0]:h-pads[0], pads[1]:w-pads[1]]
        post_result = self.postprocess_op(preds, shape_list)
        dt_boxes = post_result[0]['points']
        dt_boxes = self.filter_tag_det_res(dt_boxes, ori_im.shape)
        et = time.time()
        return dt_boxes, et - st
