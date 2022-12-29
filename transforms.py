"""
# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved
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
"""

import cv2
import numpy as np
import math


class PadImage:
    def __init__(self, pad_value=0, image_shape=(640, 640)):
        self.pad_value = pad_value
        self.image_shape = image_shape

    def pad_img(self, data):
        img = data["image"]
        h, w, _ = img.shape
        pads = [
            math.floor((self.image_shape[0] - h) // 2),
            math.floor((self.image_shape[1] - w) // 2),
        ]
        padded_img = cv2.copyMakeBorder(
            img,
            pads[0],
            int(self.image_shape[0] - h - pads[0]),
            pads[1],
            int(self.image_shape[1] - w - pads[1]),
            cv2.BORDER_CONSTANT,
            value=self.pad_value,
        )
        return padded_img, pads

    def __call__(self, data):
        padded_img, pads = self.pad_img(data)
        data["image"] = padded_img
        data["pads"] = pads
        return data

class NormalizeImage:
    """ normalize image such as substract mean, divide std
    """

    def __init__(self, scale=None, mean=None, std=None, order='chw'):
        if isinstance(scale, str):
            scale = eval(scale)
        self.scale = np.float32(scale if scale is not None else 1.0 / 255.0)
        mean = mean if mean is not None else [0.485, 0.456, 0.406]
        std = std if std is not None else [0.229, 0.224, 0.225]

        shape = (3, 1, 1) if order == 'chw' else (1, 1, 3)
        self.mean = np.array(mean).reshape(shape).astype('float32')
        self.std = np.array(std).reshape(shape).astype('float32')

    def __call__(self, data):
        img = data['image']
        from PIL import Image
        if isinstance(img, Image.Image):
            img = np.array(img)
        assert isinstance(img,
                          np.ndarray), "invalid input 'img' in NormalizeImage"
        data['image'] = (
                                img.astype('float32') * self.scale - self.mean) / self.std
        return data


class ToCHWImage(object):
    """ convert hwc image to chw image
    """

    def __init__(self, **kwargs):
        pass

    def __call__(self, data):
        img = data['image']
        from PIL import Image
        if isinstance(img, Image.Image):
            img = np.array(img)
        data['image'] = img.transpose((2, 0, 1))
        return data


class KeepKeys(object):
    def __init__(self, keep_keys):
        self.keep_keys = keep_keys

    def __call__(self, data):
        data_list = []
        for key in self.keep_keys:
            data_list.append(data[key])
        return data_list


class DetResizeForTest(object):
    def __init__(self, image_shape=(640, 640), keep_ratio=True):
        super(DetResizeForTest, self).__init__()
        self.image_shape = image_shape
        self.keep_ratio = keep_ratio

    def __call__(self, data):
        img = data['image']
        src_h, src_w, _ = img.shape
        img, [ratio_h, ratio_w] = self.resize_image(img)
        data['image'] = img
        data['shape'] = np.array([src_h, src_w, ratio_h, ratio_w])
        return data

    def resize_image(self, img):
        resize_h, resize_w = self.image_shape
        ori_h, ori_w = img.shape[:2]  # (h, w, c)
        if self.keep_ratio is True:
            if ori_h > ori_w:
                resize_w = ori_w * resize_h / ori_h
            else:
                resize_h = ori_h * resize_w / ori_w

        ratio_h = float(resize_h) / ori_h
        ratio_w = float(resize_w) / ori_w
        img = cv2.resize(img, (int(resize_w), int(resize_h)))
        return img, [ratio_h, ratio_w]
