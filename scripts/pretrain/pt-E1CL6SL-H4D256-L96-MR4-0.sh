# ------ 2022.08.23
#   NOTE: --num_workers is important to training speed, to be finetuned

pueue add -g vipformer_pt python pretrain.py --proj_name Model_mp_pt \
                    --mp \
                    --modality both \
                    --exp_name Both-E1CL6SL-H4D256-L96-MR4-0 --main_program pretrain.py \
                    --model_name partseg.py --batch_size 360 --test_batch_size 360 \
                    --num_workers 18 --epochs 300 --pt_dataset ModelNet40 \
                    --optim adamw --lr 0.001 --scheduler coswarm --step_size 100 --max_lr 0.001 --min_lr 0.0 --warm_epochs 5 --gamma 0.6 \
                    --num_pt_points 2048 --num_test_points 1024 --num_pc_latents 96 --num_img_latents 96 --num_latent_channels 256 \
                    --group_size 32 \
                    --num_ca_heads 4 --num_ca_layers 1 --mlp_widen_factor 4 --num_sa_heads 4 --num_sa_layers 6 \
                    --max_dpr 0.0 --atten_drop 0.1 --mlp_drop 0.5 --print_freq 100 --img_height 144 --img_width 144 --svm_coff 1.0
