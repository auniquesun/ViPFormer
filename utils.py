import os
import shutil
import logging
from unittest.mock import patch
import numpy as np

import torch
import torch.nn.functional as F

from parser import args

from vipformer.model.core import PerceiverEncoder, PerceiverEncoder_feats_head
from vipformer.model.core import PerceiverDecoder, PerceiverIO, ClassificationOutputAdapter
from vipformer.model.image import ImageInputAdapter
from vipformer.model.pointcloud import PointCloudInputAdapter, CrossFormer_partseg, CrossFormer_semseg
from vipformer.model.pointcloud import CrossFormer_pc_mp, CrossFormer_img_mp, CrossFormer_pc_mp_ft

import torchvision.transforms as transforms


transform = transforms.Compose([transforms.Resize((args.img_height, args.img_width)),
                                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(), 
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
# shapenetpart: 16 object classes, 50 parts
# NOTE do not use these weights, otherwise the model performance will degrade significantly
shapenetpart_part_weights = [0.0756, 0.0547, 0.0214, 0.0160, 0.0003, 0.0041, 0.0023, 0.0008, 
                0.0028, 0.0038, 0.0085, 0.0378, 0.0742, 0.0900, 0.0466, 0.0073, 0.0024, 0.0010, 
                0.0005, 0.0039, 0.0087, 0.0323, 0.0113, 0.0109, 0.0148, 0.0537, 0.0011, 0.0204, 
                0.0140, 0.0122, 0.0005, 0.0004, 0.0025, 0.0002, 7.6761e-05, 0.0071, 0.0006, 0.0098, 0.0112, 
                0.0049, 0.0009, 0.0027, 0.0007, 0.0004, 0.0010, 0.0070, 0.0006, 0.2342, 0.0727, 0.0089]
category2part = {'Airplane': [0, 1, 2, 3], 'Bag': [4, 5], 'Cap': [6, 7], 'Car': [8, 9, 10, 11], 'Chair': [12, 13, 14, 15], 
                    'Earphone': [16, 17, 18], 'Guitar': [19, 20, 21], 'Knife': [22, 23], 'Lamp': [24, 25, 26, 27], 'Laptop': [28, 29], 
                    'Motorbike': [30, 31, 32, 33, 34, 35], 'Mug': [36, 37], 'Pistol': [38, 39, 40], 'Rocket': [41, 42, 43], 
                    'Skateboard': [44, 45, 46], 'Table': [47, 48, 49]}
part2category = { 0:'Airplane', 1:'Airplane', 2:'Airplane', 3:'Airplane', 4:'Bag', 5:'Bag', 6:'Cap', 7:'Cap', 
                8:'Car', 9:'Car', 10:'Car', 11:'Car', 12:'Chair', 13:'Chair', 14:'Chair', 15:'Chair', 
                16:'Earphone', 17:'Earphone', 18:'Earphone', 19:'Guitar', 20:'Guitar', 21:'Guitar', 22:'Knife', 23:'Knife',
                24:'Lamp', 25:'Lamp', 26:'Lamp', 27:'Lamp', 28:'Laptop', 29:'Laptop', 30:'Motorbike', 31:'Motorbike',
                32:'Motorbike', 33:'Motorbike', 34:'Motorbike', 35:'Motorbike', 36:'Mug', 37:'Mug', 38:'Pistol', 39:'Pistol',
                40:'Pistol', 41:'Rocket', 42:'Rocket', 43:'Rocket', 44:'Skateboard', 45:'Skateboard', 46:'Skateboard',
                47:'Table', 48:'Table', 49:'Table'}

# s3dis: 13 object classes
s3dis_obj_weights = [0.2525, 0.2322, 0.1732, 0.0242, 0.0156, 0.0106, 0.0460, 0.0340, 0.0533, 0.0049, 0.0329, 0.0069, 0.1138]
categories = ['ceiling', 'floor', 'wall', 'beam', 'column', 'window', 'door', 'table', 'chair', 'sofa', 'bookcase', 'board', 'clutter']
category2label = {cls: i for i, cls in enumerate(categories)}
label2category = {}
for i, cat in enumerate(category2label.keys()):
    label2category[i] = cat


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class AccuracyMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.num_pos = 0
        self.num_neg = 0
        self.total = 0

    def update(self, num_pos, num_neg, n=1):
        self.num_pos += num_pos
        self.num_neg += num_neg
        self.total += n

    def pos_count(self, pred, label):
        # torch.eq(a,b): Computes element-wise equality
        results = torch.eq(pred, label)
        
        return results.sum()


class Logger(object):
    def __init__(self, logger_name='Test', log_level=logging.INFO, log_path='runs', log_file='test.log'):
        logger = logging.getLogger(logger_name)
        logger.setLevel(log_level)

        formatter = logging.Formatter('%(asctime)s [%(levelname)s] [%(name)s] %(message)s')
        file_handler = logging.FileHandler(os.path.join(log_path, log_file))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        self.logger = logger

    def write(self, msg, rank=-1):
        if rank == 0:
            self.logger.info(msg)


def build_model(rank=None):
    ''' construct a point cloud and an image model, 
            which will pretrain on a selected dataset
    '''
    pc_input_adapter = PointCloudInputAdapter(
        pointcloud_shape=(args.num_pt_points, args.point_channels),
        num_input_channels=args.num_latent_channels).to(rank)

    if args.mp:
        pc_model = CrossFormer_pc_mp(input_adapter=pc_input_adapter,
        num_latents=args.num_pc_latents,
        num_latent_channels=args.num_latent_channels,
        group_size=args.group_size,
        num_cross_attention_layers=args.num_ca_layers,
        num_cross_attention_heads=args.num_ca_heads,
        num_self_attention_layers=args.num_sa_layers,
        num_self_attention_heads=args.num_sa_heads,
        mlp_widen_factor=args.mlp_widen_factor,
        max_dpr=args.max_dpr,
        atten_drop=args.atten_drop,
        mlp_drop=args.mlp_drop,
        modal_prior=True).to(rank)

        if args.modality != 'imc-only':
            img_model = CrossFormer_img_mp(
                img_height=args.img_height,
                img_width=args.img_width,
                patch_size=args.patch_size,
                num_latent_channels=args.num_latent_channels,
                num_cross_attention_layers=args.num_ca_layers,
                num_cross_attention_heads=args.num_ca_heads,
                num_self_attention_layers=args.num_sa_layers,
                num_self_attention_heads=args.num_sa_heads,
                mlp_widen_factor=args.mlp_widen_factor,
                max_dpr=args.max_dpr,
                atten_drop=args.atten_drop,
                mlp_drop=args.mlp_drop,
                modal_prior=True).to(rank)
            return pc_model, img_model
    else:
        # Generic Perceiver encoder
        pc_model = PerceiverEncoder_feats_head(
            input_adapter=pc_input_adapter,
            num_latents=args.num_pc_latents,  # N
            num_latent_channels=args.num_latent_channels,  # D
            num_cross_attention_heads=args.num_ca_heads,
            num_cross_attention_qk_channels=pc_input_adapter.num_input_channels,  # C
            num_cross_attention_v_channels=None,
            num_cross_attention_layers=args.num_ca_layers,
            first_cross_attention_layer_shared=False,
            cross_attention_widening_factor=args.mlp_widen_factor,
            num_self_attention_heads=args.num_sa_heads,
            num_self_attention_qk_channels=None,
            num_self_attention_v_channels=None,
            num_self_attention_layers_per_block=args.num_sa_layers_per_block,
            num_self_attention_blocks=args.num_sa_blocks,
            first_self_attention_block_shared=True,
            self_attention_widening_factor=args.mlp_widen_factor,
            max_dpr=args.max_dpr,
            atten_drop=args.atten_drop,
            mlp_drop=args.mlp_drop).to(rank)

        if args.modality != 'imc-only':
            img_input_adapter = ImageInputAdapter(
                image_shape=(args.img_height, args.img_width, 3),
                num_frequency_bands=64).to(rank)
            # Generic Perceiver encoder
            img_model = PerceiverEncoder_feats_head(
                input_adapter=img_input_adapter,
                num_latents=args.num_img_latents,  # N
                num_latent_channels=args.num_latent_channels,  # D
                num_cross_attention_heads=args.num_ca_heads,
                num_cross_attention_qk_channels=args.num_latent_channels,  # C
                num_cross_attention_v_channels=None,
                num_cross_attention_layers=args.num_ca_layers,
                first_cross_attention_layer_shared=False,
                cross_attention_widening_factor=args.mlp_widen_factor,
                num_self_attention_heads=args.num_sa_heads,
                num_self_attention_qk_channels=None,
                num_self_attention_v_channels=None,
                num_self_attention_layers_per_block=args.num_sa_layers_per_block,
                num_self_attention_blocks=args.num_sa_blocks,
                first_self_attention_block_shared=True,
                self_attention_widening_factor=args.mlp_widen_factor,
                max_dpr=args.max_dpr,
                atten_drop=args.atten_drop,
                mlp_drop=args.mlp_drop).to(rank)
            return pc_model, img_model

    return pc_model


def build_ft_cls(rank=None):
    ''' construct a point cloud model, which will finetune on downstream datasets
    '''
    input_adapter = PointCloudInputAdapter(
        pointcloud_shape=(args.num_pt_points, args.point_channels),
        num_input_channels=args.num_latent_channels).to(rank)
    if args.mp:
        model = CrossFormer_pc_mp_ft(
            input_adapter=input_adapter,
            num_latents=args.num_pc_latents,
            num_latent_channels=args.num_latent_channels,
            group_size=args.group_size,
            num_cross_attention_layers=args.num_ca_layers,
            num_cross_attention_heads=args.num_ca_heads,
            num_self_attention_layers=args.num_sa_layers,
            num_self_attention_heads=args.num_sa_heads,
            mlp_widen_factor=args.mlp_widen_factor,
            max_dpr=args.max_dpr,
            atten_drop=args.atten_drop,
            mlp_drop=args.mlp_drop,
            modal_prior=True,
            num_obj_classes=args.num_obj_classes).to(rank)
    else:
        encoder = PerceiverEncoder(
            input_adapter=input_adapter,
            num_latents=args.num_pc_latents,  # N
            num_latent_channels=args.num_latent_channels,  # D
            num_cross_attention_heads=args.num_ca_heads,
            num_cross_attention_qk_channels=input_adapter.num_input_channels,  # C
            num_cross_attention_v_channels=None,
            num_cross_attention_layers=args.num_ca_layers,
            first_cross_attention_layer_shared=False,
            cross_attention_widening_factor=args.mlp_widen_factor,
            num_self_attention_heads=args.num_sa_heads,
            num_self_attention_qk_channels=None,
            num_self_attention_v_channels=None,
            num_self_attention_layers_per_block=args.num_sa_layers_per_block,
            num_self_attention_blocks=args.num_sa_blocks,
            first_self_attention_block_shared=True,
            self_attention_widening_factor=args.mlp_widen_factor,
            max_dpr=args.max_dpr,
            atten_drop=args.atten_drop,
            mlp_drop=args.mlp_drop).to(rank)

        output_adapter = ClassificationOutputAdapter(
            num_classes=args.num_obj_classes,
            num_output_queries=args.output_seq_length,
            num_output_query_channels=args.num_latent_channels).to(rank)
        decoder = PerceiverDecoder(
            output_adapter=output_adapter,
            num_latent_channels=args.num_latent_channels,  # D
            num_cross_attention_heads=args.num_ca_heads,
            num_cross_attention_qk_channels=args.num_latent_channels,
            num_cross_attention_v_channels=None,
            cross_attention_widening_factor=args.mlp_widen_factor,
            num_self_attention_heads=args.num_sa_heads,
            num_self_attention_qk_channels=None,
            num_self_attention_v_channels=None,
            num_self_attention_layers_per_block=2,  # In decoder, set `num_sa_layers=2`
            self_attention_widening_factor=args.mlp_widen_factor,
            atten_drop=args.atten_drop,
            mlp_drop=args.mlp_drop).to(rank)
        # PerceiverDecoder_var doesn't show better performances than PerceiverDecoder
        # decoder = PerceiverDecoder_var(
        #     num_latent_channels=args.num_latent_channels,
        #     num_classes=args.num_obj_classes,
        #     mlp_drop=args.mlp_drop
        # )

        model = PerceiverIO(encoder, decoder).to(rank)

    return model


def build_ft_partseg(rank=None):
    input_adapter = PointCloudInputAdapter(
        pointcloud_shape=(args.num_ft_points, args.point_channels),
        num_input_channels=args.num_latent_channels).to(rank)

    model = CrossFormer_partseg(
        input_adapter=input_adapter,
        num_latents=args.num_pc_latents,
        num_latent_channels=args.num_latent_channels,
        group_size=args.group_size,
        num_cross_attention_layers=args.num_ca_layers,
        num_cross_attention_heads=args.num_ca_heads,
        num_self_attention_layers=args.num_sa_layers,
        num_self_attention_heads=args.num_sa_heads,
        mlp_widen_factor=args.mlp_widen_factor,
        max_dpr=args.max_dpr,
        atten_drop=args.atten_drop,
        mlp_drop=args.mlp_drop,
        layer_idx=args.layer_idx,
        num_part_classes=args.num_part_classes).to(rank)

    return model


def build_ft_semseg(rank=None):
    input_adapter = PointCloudInputAdapter(
        pointcloud_shape=(args.num_ft_points, args.point_channels),
        num_input_channels=args.num_latent_channels).to(rank)

    model = CrossFormer_semseg(
        input_adapter=input_adapter,
        point_channels=args.point_channels,
        num_latents=args.num_pc_latents,
        num_latent_channels=args.num_latent_channels,
        group_size=args.group_size,
        num_cross_attention_layers=args.num_ca_layers,
        num_cross_attention_heads=args.num_ca_heads,
        num_self_attention_layers=args.num_sa_layers,
        num_self_attention_heads=args.num_sa_heads,
        mlp_widen_factor=args.mlp_widen_factor,
        max_dpr=args.max_dpr,
        atten_drop=args.atten_drop,
        mlp_drop=args.mlp_drop,
        layer_idx=args.layer_idx,
        num_obj_classes=args.num_obj_classes).to(rank)

    return model


def init(proj_name, exp_name, main_program, model_name):
    if not os.path.exists('runs'):
        os.makedirs('runs')
    if not os.path.exists(os.path.join('runs', proj_name)):
        os.makedirs(os.path.join('runs', proj_name))
    if not os.path.exists(os.path.join('runs', proj_name, exp_name)):
        os.makedirs(os.path.join('runs', proj_name, exp_name))
    if not os.path.exists(os.path.join('runs', proj_name, exp_name, 'files')):
        os.makedirs(os.path.join('runs',proj_name,  exp_name, 'files'))
    if not os.path.exists(os.path.join('runs', proj_name, exp_name, 'models')):
        os.makedirs(os.path.join('runs', proj_name, exp_name, 'models'))

    shutil.copy(main_program, os.path.join('runs', proj_name, exp_name, 'files'))
    if 'seg.py' in main_program:    # ft_partseg.py, ft_semseg.py
        shutil.copy(f'perceiver/model/pointcloud/{model_name}', os.path.join('runs', proj_name, exp_name, 'files'))
    else:
        if args.mp:
            shutil.copy(f'perceiver/model/pointcloud/partseg.py', os.path.join('runs', proj_name, exp_name, 'files'))
        else:
            shutil.copy(f'perceiver/model/core/{model_name}', os.path.join('runs', proj_name, exp_name, 'files'))
    shutil.copy('utils.py', os.path.join('runs', proj_name, exp_name, 'files'))
    
    # to fix BlockingIOError: [Errno 11]
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


def calculate_shape_IoU(pred_np, seg_np, label, class_choice, visual=False):
    seg_num = [4, 2, 2, 4, 4, 3, 3, 2, 4, 2, 6, 2, 3, 3, 3, 3]
    index_start = [0, 4, 6, 8, 12, 16, 19, 22, 24, 28, 30, 36, 38, 41, 44, 47]

    if not visual:
        label = label.squeeze()
    shape_ious = []
    for shape_idx in range(seg_np.shape[0]):
        if not class_choice:
            start_index = index_start[label[shape_idx]]
            num = seg_num[label[shape_idx]]
            parts = range(start_index, start_index + num)
        else:
            parts = range(seg_num[label[0]])
        part_ious = []
        for part in parts:
            I = np.sum(np.logical_and(pred_np[shape_idx] == part, seg_np[shape_idx] == part))
            U = np.sum(np.logical_or(pred_np[shape_idx] == part, seg_np[shape_idx] == part))
            if U == 0:
                iou = 1  # If the union of groundtruth and prediction points is empty, then count part IoU as 1
            else:
                iou = I / float(U)
            part_ious.append(iou)
        shape_ious.append(np.mean(part_ious))
    return shape_ious


def partseg_loss(pred, gold, smoothing=True):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.2
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        loss = -(one_hot * log_prb).sum(dim=1).mean()
    else:
        loss = F.cross_entropy(pred, gold, reduction='mean')

    return loss