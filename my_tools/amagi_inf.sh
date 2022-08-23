frame_interval="3"

SPLIT=1
task_type="active_bleeding"
ckpt="split12_amagi_last.pth"
config_file="configs/recognition/amagi/split1_SF152_ts3_clip8_222.py" 
data_list="[txt file that composed of lines of frame folder names]" #split_1_validation.txt"
output_prefix="results/"

data_path="[base path of dataset]"
num_class="2"
kfold=${SPLIT}
batch_size="64"

CUDA_VISIBLE_DEVICES=4 python -m torch.distributed.launch --nproc_per_node=1 inference/total_video_inference.py --data_path ${data_path} --batch_size ${batch_size} --kfold ${kfold} --num_class ${num_class} --task_type ${task_type} --frame_interval ${frame_interval} --ckpt ${ckpt} --config_file ${config_file} --data_list ${data_list} --output_prefix ${output_prefix}
