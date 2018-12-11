#!/usr/bin/env python3

import os

import time
import datetime
import cv2
import numpy as np
import uuid
import json

import functools
import logging
import collections
import tensorflow as tf

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@functools.lru_cache(maxsize=1)
def get_host_info():
    ret = {}
    with open('/proc/cpuinfo') as f:
        ret['cpuinfo'] = f.read()

    with open('/proc/meminfo') as f:
        ret['meminfo'] = f.read()

    with open('/proc/loadavg') as f:
        ret['loadavg'] = f.read()

    return ret


@functools.lru_cache(maxsize=100)
def get_predictor(checkpoint_path):
    logger.info('loading model')

    import model
    from icdar import restore_rectangle
    import lanms
    from eval import resize_image, sort_poly, detect

    input_images = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_images')
    global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

    f_score, f_geometry = model.model(input_images, is_training=False)

    variable_averages = tf.train.ExponentialMovingAverage(0.997, global_step)
    saver = tf.train.Saver(variable_averages.variables_to_restore())

    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

    # 获得检查点信息
    ckpt_state = tf.train.get_checkpoint_state(checkpoint_path)
    model_path = os.path.join(checkpoint_path, os.path.basename(ckpt_state.model_checkpoint_path))
    logger.info('Restore from {}'.format(model_path))
    # 加载模型
    saver.restore(sess, model_path)

    def predictor(img):
        """
        :return: {
            'text_lines': [
                {
                    'score': ,
                    'x0': ,
                    'y0': ,
                    'x1': ,
                    ...
                    'y3': ,
                }
            ],
            'rtparams': {  # runtime parameters
                'image_size': ,
                'working_size': ,
            },
            'timing': {
                'net': ,
                'restore': ,
                'nms': ,
                'cpuinfo': ,
                'meminfo': ,
                'uptime': ,
            }
        }
        """
        # 模型开始的时间
        start_time = time.time()
        # 有序的字典
        rtparams = collections.OrderedDict()

        # 图片的基础信息
        rtparams['start_time'] = datetime.datetime.now().isoformat()
        # 图片的大小
        rtparams['image_size'] = '{}x{}'.format(img.shape[1], img.shape[0])

        # todo 暂时不清楚这个timer是干啥的
        timer = collections.OrderedDict([
            ('net', 0),
            ('restore', 0),
            ('nms', 0)
        ])

        # 图像缩放，im_resized为调整后的图像尺寸；ratio_h和ratio_w分别是高和宽的缩放比例
        im_resized, (ratio_h, ratio_w) = resize_image(img)
        rtparams['working_size'] = '{}x{}'.format(im_resized.shape[1], im_resized.shape[0])

        # 模型预测的时间
        start = time.time()
        score, geometry = sess.run([f_score, f_geometry], feed_dict={input_images: [im_resized[:, :, ::-1]]})
        timer['net'] = time.time() - start

        # 文本框的检测
        # first_boxes 为最初的文本框；
        # boxes 为nms后的文本框
        # timer 为时间内容
        first_boxes, boxes, timer = detect(score_map=score, geo_map=geometry, timer=timer,
                                           box_thresh=0.2,
                                           nms_thres=0.5)

        logger.info('net {:.0f}ms, restore {:.0f}ms, nms {:.0f}ms'.format(
            timer['net'] * 1000, timer['restore'] * 1000, timer['nms'] * 1000))

        # 最初的文本框的还原
        if first_boxes is not None:
            first_boxes = np.array(first_boxes)[:, :8].reshape((-1, 4, 2))
            # 坐标还原
            first_boxes[:, :, 0] /= ratio_w
            first_boxes[:, :, 1] /= ratio_h

        # 合并后的文本框还原
        if boxes is not None:
            # 文本框的分值
            scores = boxes[:, 8].reshape(-1)
            # 文本框的坐标
            boxes = boxes[:, :8].reshape((-1, 4, 2))

            # 坐标还原
            boxes[:, :, 0] /= ratio_w
            boxes[:, :, 1] /= ratio_h

        duration = time.time() - start_time
        timer['overall'] = duration
        logger.info('[timing] {}'.format(duration))

        text_lines = []
        if boxes is not None:
            text_lines = []
            for box, score in zip(boxes, scores):
                box = sort_poly(box.astype(np.int32))
                if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3] - box[0]) < 5:
                    continue
                tl = collections.OrderedDict(zip(
                    ['x0', 'y0', 'x1', 'y1', 'x2', 'y2', 'x3', 'y3'],
                    map(float, box.flatten())))
                tl['score'] = float(score)
                text_lines.append(tl)
        ret = {
            'first_boxes': first_boxes.tolist(),
            'text_lines': text_lines,
            'rtparams': rtparams,
            'timing': timer,
        }
        # ret.update(get_host_info())
        return ret

    return predictor


### the webserver
from flask import Flask, request, render_template
import argparse


class Config:
    SAVE_DIR = 'static/results'


config = Config()

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html', session_id='dummy_session_id')


def draw_illu(illu, rst):
    for t in rst['text_lines']:
        d = np.array([t['x0'], t['y0'], t['x1'], t['y1'], t['x2'],
                      t['y2'], t['x3'], t['y3']], dtype='int32')
        d = d.reshape(-1, 2)
        cv2.polylines(illu, [d], isClosed=True, color=(255, 255, 0))
    return illu

def draw_debug(illu, rst):
    for t in rst['first_boxes']:
        d = np.array([t[0][0], t[0][1], t[1][0], t[1][1], t[2][0],
                      t[2][1], t[3][0], t[3][1]], dtype='int32')
        d = d.reshape(-1, 2)
        cv2.polylines(illu, [d], isClosed=True, color=(255, 255, 0))
    return illu


def save_result(img, rst):
    session_id = str(uuid.uuid1())
    dirpath = os.path.join(config.SAVE_DIR, session_id)
    os.makedirs(dirpath)

    # save input image
    output_path = os.path.join(dirpath, 'input.png')
    cv2.imwrite(output_path, img)

    # save debug
    output_path = os.path.join(dirpath, 'debug.png')
    cv2.imwrite(output_path, draw_debug(img.copy(), rst))

    # save illustration
    output_path = os.path.join(dirpath, 'output.png')
    cv2.imwrite(output_path, draw_illu(img.copy(), rst))

    # save json data
    output_path = os.path.join(dirpath, 'result.json')
    with open(output_path, 'w') as f:
        json.dump(rst, f)

    rst['session_id'] = session_id
    return rst


checkpoint_path = './east_icdar2015_resnet_v1_50_rbox'


@app.route('/', methods=['POST'])
def index_post():
    global predictor
    import io
    bio = io.BytesIO()
    request.files['image'].save(bio)
    img = cv2.imdecode(np.frombuffer(bio.getvalue(), dtype='uint8'), 1)

    # 加载文本框预测模型
    rst = get_predictor(checkpoint_path)(img)

    save_result(img, rst)
    return render_template('index.html', session_id=rst['session_id'])


def main():
    global checkpoint_path
    checkpoint_path = "/Users/xingoo/Desktop/east_resnet_v1_50_rbox"
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', default=8769, type=int)
    parser.add_argument('--checkpoint-path', default=checkpoint_path)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    checkpoint_path = args.checkpoint_path

    if not os.path.exists(args.checkpoint_path):
        raise RuntimeError(
            'Checkpoint `{}` not found'.format(args.checkpoint_path))

    app.debug = args.debug
    app.run('0.0.0.0', args.port)


if __name__ == '__main__':
    main()
