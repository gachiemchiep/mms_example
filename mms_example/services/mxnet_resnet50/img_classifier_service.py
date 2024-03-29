#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
mms service file for squeezenet using mxnet+gluoncv
"""

import io
import mxnet as mx
import gluoncv
import numpy as np
from PIL import Image
import cv2 as cv


from mms.context import Context

class ResNet50Classifier():
    def __init__(self):
        self.ctx = None
        self.net = None
        self.initialized = False

    def initialize(self, context):
        """
        Load the model and mapping file to perform infernece.

        :param context: model relevant worker information
        :return:
        """

        properties = context.system_properties
        gpu_id = properties.get("gpu_id")
        self.ctx = mx.cpu() if gpu_id is None else mx.gpu(gpu_id)
        self.net = gluoncv.model_zoo.get_model("ResNet50_v1d", pretrained=True, ctx=self.ctx)
        self.initialized = True

    def preprocess(self, data):
        """
        Scales, crops, and normalizes a PIL image for a mxnet model,

        :param data:
        :return: ndarray
        """

        img_arrs = []
        for idx in range(len(data)):
            img = data[idx].get("data")
            if img is None:
                img = data[idx].get("body")

            if img is None:
                img = data[idx].get("data")

            if img is None or len(img) == 0:
                self.error = "Empty image input"
                return None

            img_arr = mx.image.imdecode(img)
            img_arrs.append(img_arr)

        img_arrs = gluoncv.data.transforms.presets.imagenet.transform_eval(img_arrs)
        return img_arrs

    def inference(self, img, topk=5):
        """

        :param img:
        :param topk:
        :return:
        """
        pred = self.net(img.as_in_context(self.ctx))
        # map predicted values to probability by softmax
        probs = mx.nd.softmax(pred)[0].asnumpy()
        # find the 5 class indices with the highest score
        inds = mx.nd.topk(pred, k=topk)[0].astype('int').asnumpy().tolist()

        rets = []
        for i in range(topk):
            ret = dict()
            ret[self.net.classes[inds[i]]] = str(probs[inds[i]])
            rets.append(ret)

        return [rets]

    def postprocess(self, inference_output):
        return inference_output


# Following code is not necessary if your service class contains `handle(self, data, context)` function
_service = ResNet50Classifier()


def handle(data, context):
    if not _service.initialized:
        _service.initialize(context)

    if data is None:
        return None

    data = _service.preprocess(data)
    data = _service.inference(data)
    data = _service.postprocess(data)

    return data
