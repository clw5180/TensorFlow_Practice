#!/usr/bin/env python

# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Train a Fast R-CNN network on a region of interest database."""

import _init_paths
from fast_rcnn.train import get_training_roidb, train_net
from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from datasets.factory import get_imdb
from networks.factory import get_network
import argparse
import pprint
import numpy as np
import sys
import pdb

### clw note：
#   功能：设置参数，dest为目标，可通过args.XXX来访问，比如下面args.device，args.cfg_file等
def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network') # clw note：输入python train_net.py --help
                                                                               #           就会弹出这个信息
    parser.add_argument('--device', dest='device', help='device to use', # 第一个参数会在控制台输入--help时显示出来
                                                                         # 第二个参数dest是py中的变量名
                                                                         # 第三个参数help应该也会在--help时显示出来
                        default='cpu', type=str)
    parser.add_argument('--device_id', dest='device_id', help='device id to use',
                        default=0, type=int)
    parser.add_argument('--solver', dest='solver',
                        help='solver prototxt',
                        default=None, type=str)
    parser.add_argument('--iters', dest='max_iters',
                        help='number of iterations to train',
                        default=70000, type=int)
    parser.add_argument('--weights', dest='pretrained_model',
                        help='initialize with pretrained model weights',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default=None, type=str)
    parser.add_argument('--imdb', dest='imdb_name',
                        help='dataset to train on',
                        default='kitti_train', type=str)
    parser.add_argument('--rand', dest='randomize',
                        help='randomize (do not use a fixed seed)',
                        action='store_true') # clw note：如果运行py时没有带--rand这个参数，则默认是False；而只要运行py时
                                             # 带了这个--rand参数，就会将该变量设为True，无需传参，而且bool量也不支持传参；
                                             # action这个用法相当于“开关”
    parser.add_argument('--network', dest='network_name',
                        help='name of the network',
                        default=None, type=str)
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)  #  nargs=argparse.REMAINDER，所有剩余的参数，均转化为一个列表赋值给此项，
                                                   # 通常用此方法来将剩余的参数传入另一个parser进行解析。如果nargs没有定义，
                                                   # 则可传入参数的数量由action决定，通常情况下为1个，且不会生成长度为1的列表。

    # 如果sys.argv长度为1,则说明没有参数传入，系统会退出。
    # clw note：一般项目都会这么写！
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)

    if args.cfg_file is not None:     # 如果还有其他配置文件，就加载
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:     # 如果还有其他传入参数，继续加载
        cfg_from_list(args.set_cfgs)

    print('Using config:')
    # 已知类型的前提下，可以使用pprint来标准打印
    pprint.pprint(cfg)

    if not args.randomize:
        # fix the random seeds (numpy and caffe) for reproducibility
        np.random.seed(cfg.RNG_SEED)

    # imdb为存在一个字典(easydict)里的pascal_voc类的一个对象，e.g.{voc_2007_train:内容，voc_2007_val:内容，
    # voc_2007_test:内容,voc_2007_test:内容,voc_2012_train:内容...}
    # 内容里有该类里的各种self名称与操作，包括roi信息等等
    imdb = get_imdb(args.imdb_name)
    print('Loaded dataset `{:s}` for training'.format(imdb.name)) # clw modify: for py3

    #get_training_roidb函数返回imdb对象的各种roi与图片信息，用于训练
    #这是一个列表，列表中存的是各个图片的字典，字典中存roi信息，字典引索为图片引索
    roidb = get_training_roidb(imdb)

    # 输出全路径
    output_dir = get_output_dir(imdb, None)
    print('Output will be saved to `{:s}`'.format(output_dir))  # clw modify: for py3

    device_name = '/{}:{:d}'.format(args.device,args.device_id)
    print(device_name)  # clw modify: for py3

    # 得到网络结构，参数（包括rpn）
    network = get_network(args.network_name)
    print('Use network `{:s}` in training'.format(args.network_name))  # clw modify: for py3

    train_net(network, imdb, roidb, output_dir,
              pretrained_model=args.pretrained_model,
              max_iters=args.max_iters)
