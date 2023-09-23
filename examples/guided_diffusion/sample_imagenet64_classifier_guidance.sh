MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond True --diffusion_steps 1000 --dropout 0.1 --image_size 64 --learn_sigma True --noise_schedule cosine --num_channels 192 --num_head_channels 64 --num_res_blocks 3 --resblock_updown True --use_new_attention_order True --use_fp16 True --use_scale_shift_norm True"
SAMPLE_FLAGS="--batch_size 128 --num_samples 50000 --timestep_respacing ddim4 --use_ddim True"

PYTHONPATH="/userhome/llj/guided_diffusion" \
CUDA_VISIBLE_DEVICES=2 \
python ./scripts/classifier_sample.py $MODEL_FLAGS --classifier_scale 1.0 \
--classifier_path ./exps/imagenet64_cond/64x64_classifier.pt --classifier_depth 4 \
--model_path ./exps/imagenet64_cond/64x64_diffusion.pt $SAMPLE_FLAGS \
--save_dir './exps/imagenet64_cond/official_sample_use_ddim/timestep4_numsample50000_cfg7.5' \
--MASTER_PORT '12341' \
--classifier_scale 1.0 \

CUDA_VISIBLE_DEVICES=2 \
python ./evaluations/evaluator.py VIRTUAL_imagenet64_labeled.npz \
/userhome/llj/guided_diffusion/exps/imagenet64_cond/official_sample_use_ddim/timestep4_numsample50000_cfg7.5/samples_50000x64x64x3.npz