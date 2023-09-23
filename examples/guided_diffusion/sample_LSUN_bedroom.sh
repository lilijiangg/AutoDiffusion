MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --dropout 0.1 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True" 
SAMPLE_FLAGS="--batch_size 36 --num_samples 50000 --timestep_respacing ddim15 --use_ddim True"

PYTHONPATH="/userhome/llj/guided_diffusion" \
CUDA_VISIBLE_DEVICES=0 \
python scripts/image_sample.py $MODEL_FLAGS \
--model_path './exps/LSUN_bedroom/lsun_bedroom.pt' $SAMPLE_FLAGS \
--save_dir '/userhome/llj/guided_diffusion/exps/LSUN_bedroom/official_sample/ddim15_numsample50000' \

CUDA_VISIBLE_DEVICES=0 \
python ./evaluations/evaluator_lsun_bedroom.py ./exps/LSUN_bedroom/VIRTUAL_lsun_bedroom256.npz \
./exps/LSUN_bedroom/official_sample/ddim15_numsample50000/samples_50000x256x256x3.npz


# MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --dropout 0.1 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True" 
# SAMPLE_FLAGS="--batch_size 36 --num_samples 50000 --timestep_respacing ddim25 --use_ddim True"

# PYTHONPATH="/userhome/llj/guided_diffusion" \
# CUDA_VISIBLE_DEVICES=3 \
# python scripts/image_sample.py $MODEL_FLAGS \
# --model_path './exps/LSUN_bedroom/lsun_bedroom.pt' $SAMPLE_FLAGS \
# --save_dir '/userhome/llj/guided_diffusion/exps/LSUN_bedroom/official_sample/ddim25_numsample50000_useddim' \

# CUDA_VISIBLE_DEVICES=3 \
# python ./evaluations/evaluator_lsun_bedroom.py ./exps/LSUN_bedroom/VIRTUAL_lsun_bedroom256.npz \
# ./exps/LSUN_bedroom/official_sample/ddim25_numsample50000_useddim/samples_50000x256x256x3.npz