MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --large_size 256 --small_size 64 --learn_sigma True --noise_schedule linear --num_channels 192 --num_heads 4 --num_res_blocks 2 --resblock_updown True --use_fp16 False --use_scale_shift_norm True"
TRAIN_FLAGS="--batch_size 32 --microbatch 16 --lr 1e-4 --save_interval 5000 --log_interval 100 --weight_decay 0.05 --use_checkpoint True"

PYTHONPATH="$(dirname $0)/..:${PYTHONPATH}" \
mpiexec -n $1 python3 scripts/sr_t_guided_train.py $MODEL_FLAGS $TRAIN_FLAGS --data_dir /data/jupyter/data/DIV2K/DIV2K_train_HR_sub/ --save_dir $2