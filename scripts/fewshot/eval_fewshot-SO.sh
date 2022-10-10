# 2022.09.09


# # ------ MP-Both-E1CL8SL-H4D256-L128-MR2-SO-0
# ------ Acc: 96.9 +/- 1.445683229480096
pueue add -g vipformer_fewshot python eval_fewshot.py --proj_name Model_mp_pt_ft --exp_name MP-Both-E1CL8SL-H4D256-L128-MR2-SO-0 \
        --mp \
        --modality both \
        --ft_dataset ScanObjectNN --num_obj_classes 15 --num_pt_points 2048 --group_size 32 \
        --num_pc_latents 128 --num_img_latents 128 --num_latent_channels 256 \
        --num_ca_heads 4 --num_ca_layers 1 --mlp_widen_factor 2 --num_sa_heads 4 --num_sa_layers 8 \
        --max_dpr 0.0 --atten_drop 0.1 --mlp_drop 0.5 --img_height 144 --img_width 144 --patch_size 12 --svm_coff 1.0 \
        --n_runs 10 --k_way 5 --n_shot 10 --n_query 20

# ------ Acc: 97.2 +/- 2.0396078054371136
pueue add -g vipformer_fewshot python eval_fewshot.py --proj_name Model_mp_pt_ft --exp_name MP-Both-E1CL8SL-H4D256-L128-MR2-SO-0 \
        --mp \
        --modality both \
        --ft_dataset ScanObjectNN --num_obj_classes 15 --num_pt_points 2048 --group_size 32 \
        --num_pc_latents 128 --num_img_latents 128 --num_latent_channels 256 \
        --num_ca_heads 4 --num_ca_layers 1 --mlp_widen_factor 2 --num_sa_heads 4 --num_sa_layers 8 \
        --max_dpr 0.0 --atten_drop 0.1 --mlp_drop 0.5 --img_height 144 --img_width 144 --patch_size 12 --svm_coff 1.0 \
        --n_runs 10 --k_way 5 --n_shot 20 --n_query 20

# ------ Acc: 97.65 +/- 1.36106575888162
pueue add -g vipformer_fewshot python eval_fewshot.py --proj_name Model_mp_pt_ft --exp_name MP-Both-E1CL8SL-H4D256-L128-MR2-SO-0 \
        --mp \
        --modality both \
        --ft_dataset ScanObjectNN --num_obj_classes 15 --num_pt_points 2048 --group_size 32 \
        --num_pc_latents 128 --num_img_latents 128 --num_latent_channels 256 \
        --num_ca_heads 4 --num_ca_layers 1 --mlp_widen_factor 2 --num_sa_heads 4 --num_sa_layers 8 \
        --max_dpr 0.0 --atten_drop 0.1 --mlp_drop 0.5 --img_height 144 --img_width 144 --patch_size 12 --svm_coff 1.0 \
        --n_runs 10 --k_way 10 --n_shot 10 --n_query 20

# ------ Acc: 97.45 +/- 1.4908051515875573
pueue add -g vipformer_fewshot python eval_fewshot.py --proj_name Model_mp_pt_ft --exp_name MP-Both-E1CL8SL-H4D256-L128-MR2-SO-0 \
        --mp \
        --modality both \
        --ft_dataset ScanObjectNN --num_obj_classes 15 --num_pt_points 2048 --group_size 32 \
        --num_pc_latents 128 --num_img_latents 128 --num_latent_channels 256 \
        --num_ca_heads 4 --num_ca_layers 1 --mlp_widen_factor 2 --num_sa_heads 4 --num_sa_layers 8 \
        --max_dpr 0.0 --atten_drop 0.1 --mlp_drop 0.5 --img_height 144 --img_width 144 --patch_size 12 --svm_coff 1.0 \
        --n_runs 10 --k_way 10 --n_shot 20 --n_query 20
