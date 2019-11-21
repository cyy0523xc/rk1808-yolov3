from rknn.api import RKNN
from settings import yolov3_weights, yolov3_model_cfg, rknn_model


if __name__ == '__main__':
    # Create RKNN object
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
        print('Load darknet yolov3 failed!')
        exit(ret)
    print('done')

    # Build model
    print('--> Building model')
    # ret = rknn.build(do_quantization=True, dataset='./dataset.txt')
    # do_quantization:是否对模型进行量化,值为 True 或 False。
    ret = rknn.build(do_quantization=False)
    if ret != 0:
        print('Build yolov3 failed!')
        exit(ret)
    print('done')

    # Export rknn model
    print('--> Export RKNN model')
    ret = rknn.export_rknn(rknn_model)
    if ret != 0:
        print('Export rknn model: %s failed!' % rknn_model)
        exit(ret)

    print('done: %s' % rknn_model)
