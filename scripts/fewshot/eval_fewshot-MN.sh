# 2022.08.31

# # ------ Both-E1CL8SL-H6D384-L128-MR4-0
# # ------ Acc: 91.1 +/- 7.189575787207477
# pueue add -g vipformer_fewshot python eval_fewshot.py --proj_name Model_mp_pt --exp_name Both-E1CL8SL-H6D384-L128-MR4-0 \
#         --mp \
#         --modality both \
#         --ft_dataset ModelNet40 --num_pt_points 2048 --group_size 32 \
#         --num_pc_latents 128 --num_img_latents 128 --num_latent_channels 384 \
#         --num_ca_heads 6 --num_ca_layers 1 --mlp_widen_factor 4 --num_sa_heads 6 --num_sa_layers 8 \
#         --max_dpr 0.0 --atten_drop 0.1 --mlp_drop 0.5 --img_height 144 --img_width 144 --patch_size 12 --svm_coff 1.0 \
#         --n_runs 10 --k_way 5 --n_shot 10 --n_query 20

# # ------ Acc: 93.4 +/- 4.5431266766402185
# pueue add -g vipformer_fewshot python eval_fewshot.py --proj_name Model_mp_pt --exp_name Both-E1CL8SL-H6D384-L128-MR4-0 \
#         --mp \
#         --modality both \
#         --ft_dataset ModelNet40 --num_pt_points 2048 --group_size 32 \
#         --num_pc_latents 128 --num_img_latents 128 --num_latent_channels 384 \
#         --num_ca_heads 6 --num_ca_layers 1 --mlp_widen_factor 4 --num_sa_heads 6 --num_sa_layers 8 \
#         --max_dpr 0.0 --atten_drop 0.1 --mlp_drop 0.5 --img_height 144 --img_width 144 --patch_size 12 --svm_coff 1.0 \
#         --n_runs 10 --k_way 5 --n_shot 20 --n_query 20

# # ------ Acc: 80.75 +/- 4.172828776741265
# pueue add -g vipformer_fewshot python eval_fewshot.py --proj_name Model_mp_pt --exp_name Both-E1CL8SL-H6D384-L128-MR4-0 \
#         --mp \
#         --modality both \
#         --ft_dataset ModelNet40 --num_pt_points 2048 --group_size 32 \
#         --num_pc_latents 128 --num_img_latents 128 --num_latent_channels 384 \
#         --num_ca_heads 6 --num_ca_layers 1 --mlp_widen_factor 4 --num_sa_heads 6 --num_sa_layers 8 \
#         --max_dpr 0.0 --atten_drop 0.1 --mlp_drop 0.5 --img_height 144 --img_width 144 --patch_size 12 --svm_coff 1.0 \
#         --n_runs 10 --k_way 10 --n_shot 10 --n_query 20

# # ------ Acc: 87.05 +/- 5.811411188343156
# pueue add -g vipformer_fewshot python eval_fewshot.py --proj_name Model_mp_pt --exp_name Both-E1CL8SL-H6D384-L128-MR4-0 \
#         --mp \
#         --modality both \
#         --ft_dataset ModelNet40 --num_pt_points 2048 --group_size 32 \
#         --num_pc_latents 128 --num_img_latents 128 --num_latent_channels 384 \
#         --num_ca_heads 6 --num_ca_layers 1 --mlp_widen_factor 4 --num_sa_heads 6 --num_sa_layers 8 \
#         --max_dpr 0.0 --atten_drop 0.1 --mlp_drop 0.5 --img_height 144 --img_width 144 --patch_size 12 --svm_coff 1.0 \
#         --n_runs 10 --k_way 10 --n_shot 20 --n_query 20


# ------ Both-E1CL8SL-H6D384-L96-MR4-0
# ------ Acc: 88.1 +/- 7.327346040688949
pueue add -g vipformer_fewshot python eval_fewshot.py --proj_name Model_mp_pt --exp_name Both-E1CL8SL-H6D384-L96-MR4-0 \
        --mp \
        --modality both \
        --ft_dataset ModelNet40 --num_pt_points 2048 --group_size 32 \
        --num_pc_latents 96 --num_img_latents 96 --num_latent_channels 384 \
        --num_ca_heads 6 --num_ca_layers 1 --mlp_widen_factor 4 --num_sa_heads 6 --num_sa_layers 8 \
        --max_dpr 0.0 --atten_drop 0.1 --mlp_drop 0.5 --img_height 144 --img_width 144 --patch_size 12 --svm_coff 1.0 \
        --n_runs 10 --k_way 5 --n_shot 10 --n_query 20

# ------ Acc: 93.7 +/- 5.139066063011839
pueue add -g vipformer_fewshot python eval_fewshot.py --proj_name Model_mp_pt --exp_name Both-E1CL8SL-H6D384-L96-MR4-0 \
        --mp \
        --modality both \
        --ft_dataset ModelNet40 --num_pt_points 2048 --group_size 32 \
        --num_pc_latents 96 --num_img_latents 96 --num_latent_channels 384 \
        --num_ca_heads 6 --num_ca_layers 1 --mlp_widen_factor 4 --num_sa_heads 6 --num_sa_layers 8 \
        --max_dpr 0.0 --atten_drop 0.1 --mlp_drop 0.5 --img_height 144 --img_width 144 --patch_size 12 --svm_coff 1.0 \
        --n_runs 10 --k_way 5 --n_shot 20 --n_query 20

# ------ Acc: 81.4 +/- 5.59821400091136
pueue add -g vipformer_fewshot python eval_fewshot.py --proj_name Model_mp_pt --exp_name Both-E1CL8SL-H6D384-L96-MR4-0 \
        --mp \
        --modality both \
        --ft_dataset ModelNet40 --num_pt_points 2048 --group_size 32 \
        --num_pc_latents 96 --num_img_latents 96 --num_latent_channels 384 \
        --num_ca_heads 6 --num_ca_layers 1 --mlp_widen_factor 4 --num_sa_heads 6 --num_sa_layers 8 \
        --max_dpr 0.0 --atten_drop 0.1 --mlp_drop 0.5 --img_height 144 --img_width 144 --patch_size 12 --svm_coff 1.0 \
        --n_runs 10 --k_way 10 --n_shot 10 --n_query 20

# ------ Acc: 88.15 +/- 4.105179655021202
pueue add -g vipformer_fewshot python eval_fewshot.py --proj_name Model_mp_pt --exp_name Both-E1CL8SL-H6D384-L96-MR4-0 \
        --mp \
        --modality both \
        --ft_dataset ModelNet40 --num_pt_points 2048 --group_size 32 \
        --num_pc_latents 96 --num_img_latents 96 --num_latent_channels 384 \
        --num_ca_heads 6 --num_ca_layers 1 --mlp_widen_factor 4 --num_sa_heads 6 --num_sa_layers 8 \
        --max_dpr 0.0 --atten_drop 0.1 --mlp_drop 0.5 --img_height 144 --img_width 144 --patch_size 12 --svm_coff 1.0 \
        --n_runs 10 --k_way 10 --n_shot 20 --n_query 20
