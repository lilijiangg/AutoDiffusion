MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond True --diffusion_steps 1000 --in_channels 64 --use_feature True --image_size 256 --learn_sigma False --sigma_small True --noise_schedule linear --num_channels 128 --num_heads 2 --num_res_blocks 2 --resblock_updown True --use_fp16 False --use_scale_shift_norm True"
TRAIN_FLAGS="--batch_size 32 --lr 3e-4 --save_interval 10000 --log_interval 100 --weight_decay 0.05 --use_checkpoint True --res True --use_kl False --predict_xstart True"

PYTHONPATH="$(dirname $0)/..:${PYTHONPATH}" \
mpiexec -n $1 python3 scripts/image_train.py $MODEL_FLAGS $TRAIN_FLAGS --data_dir "s3://research-model-hh-b/Dataset/ILSVRC2012/imagenet.train.nori.list" --save_dir $2