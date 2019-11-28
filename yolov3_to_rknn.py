import sys
import shutil

# 复制配置文件
# 可以选择tiny或者spp
yolo_type = 'tiny' if len(sys.argv) < 2 else sys.argv[1]
shutil.copy("settings_%s.py" % yolo_type, "settings.py")


if __name__ == '__main__':
    from rknn.api import RKNN
    from timeit import default_timer as timer
    from settings import yolov3_weights, yolov3_model_cfg, rknn_model, \
        pre_compile

    # Create RKNN object
    total_timer = timer()
    rknn = RKNN()

    # pre-process config
    print('--> config model')
    rknn.config(channel_mean_value='103.94 116.78 123.68 58.82',
                reorder_channel='0 1 2')
    print('done')

    # Load tensorflow model
    print('--> Loading model')
    ret = rknn.load_darknet(model=yolov3_model_cfg, weight=yolov3_weights)
    if ret != 0:
        raise Exception('Load darknet yolov3 failed!')
    print('done')

    # Build model
    print('--> Building model')
    build_timer = timer()
    # ret = rknn.build(do_quantization=True, dataset='./dataset.txt')
    # do_quantization:是否对模型进行量化,值为 True 或 False。
    ret = rknn.build(do_quantization=False, pre_compile=pre_compile)
    if ret != 0:
        raise Exception('Build yolov3 failed!')
    print('done, time: %.2fs' % (timer()-build_timer))

    # Export rknn model
    print('--> Export RKNN model')
    ret = rknn.export_rknn(rknn_model)
    if ret != 0:
        raise Exception('Export rknn model: %s failed!' % rknn_model)

    print('done: %s, time: %.2fs' % (rknn_model, timer()-total_timer))
