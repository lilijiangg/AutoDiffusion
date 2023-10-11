MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond True --diffusion_steps 1000 --dropout 0.1 --image_size 64 --learn_sigma True --noise_schedule cosine --num_channels 192 --num_head_channels 64 --num_res_blocks 3 --resblock_updown True --use_new_attention_order True --use_fp16 True --use_scale_shift_norm True"
SAMPLE_FLAGS="--batch_size 128 --num_samples 50000 --use_ddim True"

# PYTHONPATH="/userhome/llj/guided_diffusion" \
CUDA_VISIBLE_DEVICES=0 \
python ./scripts/classifier_sample_prunedUNET.py $MODEL_FLAGS --classifier_scale 1.0 \
--classifier_path /userhome/llj/guided_diffusion/exps/imagenet64_cond/64x64_classifier.pt --classifier_depth 4 \
--model_path /userhome/llj/guided_diffusion/exps/imagenet64_cond/64x64_diffusion.pt $SAMPLE_FLAGS \
--use_mean False \
--use_dynamic_unet True \
--MASTER_PORT '12347' \
--save_dir './exps/imagenet64_cond/ea_search_rf_ddim/InitTimeteps10_IndexStep580_numsample3000_progressive_mprob0.25_without_rf_newmutation_max0.2_min0.0_newProgressive/best_cand_pruning' \
--use_timestep '[744, 137, 647, 856, 305, 441, 676, 572, 971, 85]' \
--skip_layers '[[], [], [], [], [], [], [30, 10, 39, 4, 15, 46, 49, 54, 8], [], [], []]' \

CUDA_VISIBLE_DEVICES=0 \
python ./evaluations/evaluator.py VIRTUAL_imagenet64_labeled.npz \
./exps/imagenet64_cond/ea_search_rf_ddim/InitTimeteps10_IndexStep580_numsample3000_progressive_mprob0.25_without_rf_newmutation_max0.2_min0.0_newProgressive/best_cand_pruning_r1/samples_500x64x64x3.npz