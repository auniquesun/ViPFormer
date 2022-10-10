# ------ 2022.08.29

# ModelNet40, classification accuracy: 
pueue add -g vipformer_ft python ft_cls.py --proj_name Model_mp_pt_ft \
                    --mp \
                    --modality both \
                    --exp_name MP-Both-E1CL8SL-H4D256-L128-MR2-MN-0 --main_program ft_cls.py \
                    --model_name partseg.py --resume --pc_model_file runs/Model_mp_pt/Both-E1CL8SL-H4D256-L128-MR2-0/models/pc_model_best.pth \
                    --ft_dataset ModelNet40 --num_obj_classes 40 --batch_size 1080 --test_batch_size 1080 --num_workers 0 --epochs 300 \
                    --optim adamw --lr 0.001 --scheduler coswarm --step_size 100 --max_lr 0.001 --min_lr 0.0 --warm_epochs 5 --gamma 0.6 \
                    --num_ft_points 1024 --num_pc_latents 128 --num_latent_channels 256 \
                    --group_size 32 \
                    --num_ca_heads 4 --num_ca_layers 1 --mlp_widen_factor 2 --num_sa_heads 4 --num_sa_layers 8 \
                    --max_dpr 0.0 --atten_drop 0.1 --mlp_drop 0.5 --print_freq 4

# ScanObjectNN, classification accuracy: 
pueue add -g vipformer_ft python ft_cls.py --proj_name Model_mp_pt_ft \
                    --mp \
                    --modality both \
                    --exp_name MP-Both-E1CL8SL-H4D256-L128-MR2-SO-0 --main_program ft_cls.py \
                    --model_name partseg.py --resume --pc_model_file runs/Model_mp_pt/Both-E1CL8SL-H4D256-L128-MR2-0/models/pc_model_best.pth \
                    --ft_dataset ScanObjectNN --num_obj_classes 15 --batch_size 1080 --test_batch_size 1080 --num_workers 0 --epochs 300 \
                    --optim adamw --lr 0.001 --scheduler coswarm --step_size 100 --max_lr 0.001 --min_lr 0.0 --warm_epochs 5 --gamma 0.6 \
                    --num_ft_points 1024 --num_pc_latents 128 --num_latent_channels 256 \
                    --group_size 32 \
                    --num_ca_heads 4 --num_ca_layers 1 --mlp_widen_factor 2 --num_sa_heads 4 --num_sa_layers 8 \
                    --max_dpr 0.0 --atten_drop 0.1 --mlp_drop 0.5 --print_freq 4 
