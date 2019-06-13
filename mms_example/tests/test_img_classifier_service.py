#!/usr/bin/env python
# -*- coding: utf-8 -*-

from unittest import TestCase
from mms_example.services.mxnet_resnet50.img_classifier_service import ResNet50Classifier
import json
import base64


class TestResNet50Classifier(TestCase):

    def test_initialize(self):

        obj = ResNet50Classifier()
        obj.initialize(context=None)

    def test_preprocess(self):

        with open("kitten.json", "r") as fp:
            data = json.load(fp)

        # remove first unused string
        data['data'] = base64.b64decode(data['data'][len('data:application/octet-stream;base64,'):])

        obj = ResNet50Classifier()
        obj.initialize(context=None)
        ret = obj.preprocess([data])
        return ret

    def test_inference(self):
        with open("kitten.json", "r") as fp:
            data = json.load(fp)

        # remove first unused string
        data['data'] = base64.b64decode(data['data'][len('data:application/octet-stream;base64,'):])

        obj = ResNet50Classifier()
        obj.initialize(context=None)
        ret = obj.preprocess([data])
        ret = obj.initialize(ret)
        return ret

    def test_postprocess(self):
        with open("kitten.json", "r") as fp:
            data = json.load(fp)

        # remove first unused string
        data['data'] = base64.b64decode(data['data'][len('data:application/octet-stream;base64,'):])

        obj = ResNet50Classifier()
        obj.initialize(context=None)
        ret = obj.preprocess([data])
        ret = obj.initialize(ret)
        ret = obj.postprocess(ret)
        print(ret)
        return ret
