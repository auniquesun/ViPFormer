# 2022.08.16
# --num_sa_layers 12 --max_dpr 0.2 --layer_idx 3 7 11 
python show_balls.py --ballradius 13 \
                --class_choice Pistol --pc_model_file ../runs/CrossFormer_partseg_scratch/E_1CL12SL-D_5L-2/models/model_best.pth \
                --ft_dataset ShapeNetPart --num_ft_points 2500 --num_obj_classes 16 --test_batch_size 100 \
                --num_pc_latents 128 --num_latent_channels 384 --group_size 32 \
                --num_ca_layers 1 --num_ca_heads 6 --num_sa_layers 12 --num_sa_heads 6 \
                --mlp_widen_factor 4 --max_dpr 0.2 --atten_drop 0.0 --mlp_drop 0.0 --num_part_classes 50 \
                --layer_idx 3 7 11 

# --num_sa_layers 12 --max_dpr 0.2 --layer_idx 4 8 12
python show_balls.py --ballradius 13 \
                --class_choice Pistol --pc_model_file ../runs/CrossFormer_partseg_scratch/E_1CL12SL-D_5L-2/models/model_best.pth \
                --ft_dataset ShapeNetPart --num_ft_points 2500 --num_obj_classes 16 --test_batch_size 100 \
                --num_pc_latents 128 --num_latent_channels 384 --group_size 32 \
                --num_ca_layers 1 --num_ca_heads 6 --num_sa_layers 12 --num_sa_heads 6 \
                --mlp_widen_factor 4 --max_dpr 0.2 --atten_drop 0.0 --mlp_drop 0.0 --num_part_classes 50 \
                --layer_idx 3 7 11
