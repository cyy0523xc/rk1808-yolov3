# -*- coding: utf-8 -*-
#
#
# Author: alex
# Created Time: 2019年11月21日 星期四 17时40分14秒

# darknet版yolov3
yolov3_model_cfg = './yolov3/gf-yolov3-spp.cfg'
yolov3_weights = './yolov3/gf-yolov3-spp_final.weights'

# rknn model file
pre_compile = True
rknn_model = './rknn_models/gf_yolov3_spp.rknn'

# 识别目标的类别
CLASSES = ('idcard', 'idcard_back', 'logo', 'jobcard')

# 输入图片尺寸
input_size = (608, 608)

# # yolov3
masks = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
anchors = [[10, 13], [16, 30], [33, 23], [30, 61], [62, 45],
           [59, 119], [116, 90], [156, 198], [373, 326]]
# yolov3-tiny
# masks = [[3, 4, 5], [0, 1, 2]]
# anchors = [[10, 14], [23, 27], [37, 58], [81, 82], [135, 169], [344, 319]]
