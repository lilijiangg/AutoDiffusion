MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond True --image_size 128 --learn_sigma True --num_channels 256 --num_heads 4 --num_res_blocks 2 --resblock_updown True --use_fp16 False --use_scale_shift_norm True"

CLASSIFIER_FLAGS="--image_size 128 --classifier_attention_resolutions 32,16,8 --classifier_depth 2 --classifier_width 128 --classifier_pool attention --classifier_resblock_updown True --classifier_use_scale_shift_norm True --classifier_scale 1.0 --classifier_use_fp16 False"

SAMPLE_FLAGS="--batch_size 32 --num_samples 50000 --timestep_respacing ddim25 --use_ddim True"

PYTHONPATH="$(dirname $0)/..:${PYTHONPATH}" \
python3 -m torch.distributed.launch --nproc_per_node=8  scripts/classifier_sample.py \
    --model_path $1 \
    --classifier_path /data/jupyter/models/128x128_classifier.pt \
    $MODEL_FLAGS $CLASSIFIER_FLAGS $SAMPLE_FLAGS --save_dir $2