CONFIG_FILE="configs/recognition/amagi/split1_SF152_ts3_clip8_222.py"
GPU_NUM="2"


CUDA_VISIBLE_DEVICES=0,1 ./tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM} --validate
