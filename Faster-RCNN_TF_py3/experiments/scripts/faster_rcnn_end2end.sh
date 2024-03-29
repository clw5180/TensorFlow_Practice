#!/bin/bash
# Usage:
# ./experiments/scripts/faster_rcnn_end2end.sh GPU NET DATASET [options args to {train,test}_net.py]
# DATASET is either pascal_voc or coco.
#
# Example:
# ./experiments/scripts/faster_rcnn_end2end.sh 0 VGG_CNN_M_1024 pascal_voc \
#   --set EXP_DIR foobar RNG_SEED 42 TRAIN.SCALES "[400, 500, 600, 700]"

# --------------------------------------------------------------------------------------------------------------------------------------------------
# clw note：
# 可以借鉴下 https://github.com/endernewton/tf-faster-rcnn ，把训练和测试分成两个脚本
# GPU_ID is the GPU you want to test on
# NET in {vgg16, res50, res101, res152} is the network arch to use
# DATASET {pascal_voc, pascal_voc_0712, coco} is defined in train_faster_rcnn.sh
# Examples:
#./experiments/scripts/train_faster_rcnn.sh 0 pascal_voc vgg16
#./experiments/scripts/train_faster_rcnn.sh 1 coco res101

# clw note：
# 在你开始构思并写下具体的代码逻辑之前，先插入一行set -e和一行set -x。
#set -x会在执行每一行 shell 脚本时，把执行的内容输出来。它可以让你看到当前执行的情况，里面涉及的变量也会被替换成实际的值。
#set -e会在执行出错时结束程序，就像其他语言中的“抛出异常”一样。（但也不是所有出错的时候都会结束程序）
# --------------------------------------------------------------------------------------------------------------------------------------------------
set -x
set -e

export PYTHONUNBUFFERED="True"

DEV=$1
DEV_ID=$2
NET=$3
DATASET=$4

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:4:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

case $DATASET in
  pascal_voc)
    TRAIN_IMDB="voc_2007_trainval"
    TEST_IMDB="voc_2007_test"
    PT_DIR="pascal_voc"
    ITERS=70000
    ;;
  coco)
    # This is a very long and slow training schedule
    # You can probably use fewer iterations and reduce the
    # time to the LR drop (set in the solver to 350,000 iterations).
    TRAIN_IMDB="coco_2014_train"
    TEST_IMDB="coco_2014_minival"
    PT_DIR="coco"
    ITERS=490000
    ;;
  *)
    echo "No dataset given"
    exit
    ;;
esac

LOG="experiments/logs/faster_rcnn_end2end_${NET}_${EXTRA_ARGS_SLUG}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time python ./tools/train_net.py --device ${DEV} --device_id ${DEV_ID} \
  --weights data/pretrain_model/VGG_imagenet.npy \
  --imdb ${TRAIN_IMDB} \
  --iters ${ITERS} \
  --cfg experiments/cfgs/faster_rcnn_end2end.yml \
  --network VGGnet_train \
  ${EXTRA_ARGS}

set +x
NET_FINAL=`grep -B 1 "done solving" ${LOG} | grep "Wrote snapshot" | awk '{print $4}'`
set -x

time python ./tools/test_net.py --device ${DEV} --device_id ${DEV_ID} \
  --weights ${NET_FINAL} \
  --imdb ${TEST_IMDB} \
  --cfg experiments/cfgs/faster_rcnn_end2end.yml \
  --network VGGnet_test \
  ${EXTRA_ARGS}
