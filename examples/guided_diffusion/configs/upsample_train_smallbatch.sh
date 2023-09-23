MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --large_size 256 --small_size 64 --learn_sigma True --noise_schedule linear --num_channels 128 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True"
TRAIN_FLAGS="--batch_size 32 --microbatch 2 --lr 3e-5 --save_interval 5000 --log_interval 10 --weight_decay 0.05 --use_checkpoint False --prompt_flag None --lq_embed False"
# when using prompt, do not use checkpointing

PYTHONPATH="$(dirname $0)/..:${PYTHONPATH}" \
torchrun --nproc_per_node=8 scripts/super_res_train.py $MODEL_FLAGS $TRAIN_FLAGS --data_dir /data/jupyter/data/DF2K/DF2K_train_HR_sub/ --lq_paths /data/jupyter/data/DF2K/DF2K_train_LR_bicubic_sub/X4/ --model_dir $1 --save_dir $2 --mix_loss None 