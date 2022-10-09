import os
import wandb
from datetime import datetime

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts, StepLR, ReduceLROnPlateau
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.nn import CrossEntropyLoss

from datasets.shapenet_part import ShapeNetPart
from torch.utils.data import DataLoader

from utils import AccuracyMeter, init, Logger, AverageMeter
from utils import build_ft_partseg, category2part, part2category
from parser import args


def setup(rank):
    # initialization for distibuted training on multiple GPUs
    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = args.master_port

    dist.init_process_group(args.backend, rank=rank, world_size=args.world_size)

    
def cleanup():
    dist.destroy_process_group()


def main(rank, logger_name, log_path, log_file):
    if rank == 0:
        os.environ["WANDB_BASE_URL"] = args.wb_url
        wandb.login(key=args.wb_key)
        wandb.init(project=args.proj_name, name=args.exp_name)

    logger = Logger(logger_name=logger_name, log_path=log_path, log_file=log_file)

    setup(rank)

    train_dataset = ShapeNetPart(partition='trainval', num_points=args.num_ft_points)
    train_sampler = DistributedSampler(train_dataset, num_replicas=args.world_size, rank=rank)
    
    samples_per_gpu = args.batch_size // args.world_size
    train_loader = DataLoader(
                    train_dataset, 
                    sampler=train_sampler,
                    batch_size=samples_per_gpu, 
                    shuffle=False, 
                    num_workers=args.num_workers,
                    pin_memory=True,
                    drop_last=False)

    test_dataset = ShapeNetPart(partition='test', num_points=args.num_ft_points)
    test_sampler = DistributedSampler(test_dataset, num_replicas=args.world_size, rank=rank)

    test_samples_per_gpu = args.test_batch_size // args.world_size
    test_loader = DataLoader(test_dataset, 
                            sampler=test_sampler,
                            batch_size=test_samples_per_gpu, 
                            shuffle=False, 
                            num_workers=0, 
                            pin_memory=True, 
                            drop_last=False)
    
    num_part_classes = train_loader.dataset.seg_num_all
    seg_start_index = train_loader.dataset.seg_start_index

    model = build_ft_partseg(rank=rank)
    model_ddp = DDP(model, device_ids=[rank], find_unused_parameters=True)

    # ----- load pretrained model
    assert args.resume, 'Finetuning ViPFormer_partseg requires pretrained model weights'
    map_location = torch.device('cuda:%d' % rank)
    pretrained = torch.load(args.pc_model_file, map_location=map_location)
    # append `module.` at the beginning of key
    pretrained = {"module."+key: value for key, value in pretrained.items()}
    model_ddp.load_state_dict(pretrained, strict=False)

    if args.optim == 'sgd':
        optimizer = optim.SGD(
            model_ddp.parameters(),
            lr=args.lr,
            momentum=args.momentum)
    elif args.optim == 'adam':
        optimizer = optim.Adam(
            model_ddp.parameters(),
            lr=args.lr,
            weight_decay=1e-6)
    elif args.optim == 'adamw':
        optimizer = optim.AdamW(
            model_ddp.parameters(),
            lr=args.lr)

    logger.write(f'Using {args.optim} optimizer ...', rank=rank)

    if args.scheduler == 'cos':
        lr_scheduler = CosineAnnealingLR(
            optimizer, 
            T_max=args.epochs)
    elif args.scheduler == 'coswarm':
        # lr_scheduler = CosineAnnealingWarmRestarts(
        #     optimizer, 
        #     T_0=args.warm_epochs)
        lr_scheduler = CosineAnnealingWarmupRestarts(
            optimizer,
            first_cycle_steps=args.step_size,
            max_lr=args.max_lr,
            min_lr=args.min_lr,
            warmup_steps=args.warm_epochs,
            gamma=args.gamma)
    elif args.scheduler == 'plateau':
        lr_scheduler = ReduceLROnPlateau(
            optimizer, 
            mode='min',
            factor=args.factor,
            patience=args.patience)
    elif args.scheduler == 'step':
        lr_scheduler = StepLR(
            optimizer, 
            step_size=args.step_size)

    criterion = CrossEntropyLoss(label_smoothing=0.2)

    test_best_point_level_acc = .0
    test_best_mean_part_acc = .0
    test_best_mean_part_iou = .0
    test_best_mean_category_iou = .0
    best_epoch = 0
    for epoch in range(args.epochs):
        # ------ Train
        model_ddp.train()

        train_sampler.set_epoch(epoch)

        train_loss = AverageMeter()
        points_seg_acc = AccuracyMeter()

        start_train = datetime.now()
        for batch_idx, (points, obj_label, partseg_label) in enumerate(train_loader):
            # points: [batch, num_points, 3]
            # obj_label: [batch, 1], label of object categories
            # partseg_label: [batch, num_points], label of object parts
            optimizer.zero_grad(set_to_none=True)
            
            batch_size, num_points, _ = points.size()
            label_onehot = torch.zeros(batch_size, args.num_obj_classes, device=f'cuda:{rank}')
            for i in range(batch_size):
                label_onehot[i, obj_label[i]] = 1
            partseg_label = partseg_label - seg_start_index
            points, partseg_label = points.to(rank), partseg_label.to(rank)
            # partseg_pred: [batch, num_points, num_part_classes]
            partseg_pred = model_ddp(points, label_onehot)
            # partseg_pred = model_ddp(points)
            loss = criterion(partseg_pred.reshape(-1, num_part_classes), partseg_label.reshape(-1))
            train_loss.update(loss, batch_size)

            refined_pred = torch.zeros(batch_size, num_points, dtype=torch.int32, device=f'cuda:{rank}')
            for i in range(batch_size):
                # partseg_label[i, 0] is a tensor, it should be converted to an integer to serve as an index
                idx = partseg_label[i, 0].item()
                cat = part2category[idx]
                logits = partseg_pred[i, :, :]
                refined_pred[i, :] = torch.argmax(logits[:, category2part[cat]], dim=1) + category2part[cat][0]
            pos = points_seg_acc.pos_count(refined_pred, partseg_label)
            points_seg_acc.update(pos, batch_size*num_points-pos, batch_size*num_points)

            loss.backward()
            # clip grad to prevent exploding, following Point-MAE and Point-Bert
            torch.nn.utils.clip_grad_norm_(model_ddp.parameters(), 10, norm_type=2)
            optimizer.step()

            # if batch_idx % args.print_freq == 0:
            #     msg = 'Batch (%d/%d), train_loss: %.6f, point_level_acc: %.6f' % \
            #         (batch_idx, len(train_loader), train_loss.avg.item(), points_seg_acc.num_pos.item()/points_seg_acc.total)
            #     logger.write(msg, rank=rank)
        train_duration = datetime.now() - start_train
        train_loss, train_acc = train_loss.avg.item(), points_seg_acc.num_pos.item()/points_seg_acc.total
        outstr = 'Train (%d/%d), train_loss: %.6f, point_level_acc: %.6f' % (epoch, args.epochs, train_loss, train_acc)
        logger.write(outstr, rank=rank)

        # ------ Test
        with torch.no_grad():
            test_start = datetime.now()
            test_mean_part_iou, test_mean_category_iou, test_mean_part_acc, test_point_level_acc, test_loss = \
                test(rank, model_ddp, test_loader, criterion)
            test_duration = datetime.now() - test_start
            outstr = 'Test (%d/%d), mean_part_iou: %.6f, mean_category_iou: %.6f, mean_part_acc: %.6f, point_level_acc: %.6f, test_loss: %.6f' % \
            (epoch, args.epochs, test_mean_part_iou, test_mean_category_iou, test_mean_part_acc, test_point_level_acc, test_loss)
            logger.write(outstr, rank=rank)

            if rank == 0:
                if test_point_level_acc > test_best_point_level_acc:
                    test_best_point_level_acc = test_point_level_acc
                if test_mean_part_acc > test_best_mean_part_acc:
                    test_best_mean_part_acc = test_mean_part_acc
                if test_mean_part_iou > test_best_mean_part_iou:
                    test_best_mean_part_iou = test_mean_part_iou

                if test_mean_category_iou > test_best_mean_category_iou:
                    test_best_mean_category_iou = test_mean_category_iou
                    best_epoch = epoch
                    logger.write(f'Find new highest Mean Category IoU: {test_best_mean_category_iou} at epoch {best_epoch}!', rank=rank)
                    logger.write('Saving best model ...', rank=rank)
                    save_state = {'epoch': epoch, # start from 0
                        'test_loss': test_loss, 
                        'test_mean_part_iou': test_mean_part_iou,
                        'test_mean_category_iou': test_mean_category_iou, 
                        'test_mean_part_acc': test_mean_part_acc,
                        'test_point_level_acc': test_point_level_acc,
                        'model_state_dict': model_ddp.module.state_dict(),
                        'optim_state_dict': optimizer.state_dict()}
                    save_path = os.path.join('runs', args.proj_name, args.exp_name, 'models', 'model_best.pth')
                    torch.save(save_state, save_path)
                
                wandb_log = dict()
                if args.scheduler == 'coswarm':
                    wandb_log['learning_rate'] = lr_scheduler.get_lr()[0]
                else:
                    wandb_log['learning_rate'] = lr_scheduler.get_last_lr()[0]
                wandb_log["train_loss"] = train_loss
                wandb_log["train_point_level_acc"] = train_acc
                wandb_log["test_point_level_acc"] = test_point_level_acc
                wandb_log["test_best_point_level_acc"] = test_best_point_level_acc
                wandb_log["test_mean_part_acc"] = test_mean_part_acc
                wandb_log["test_best_mean_part_acc"] = test_best_mean_part_acc
                wandb_log["test_mean_part_iou"] = test_mean_part_iou
                wandb_log["test_best_mean_part_iou"] = test_best_mean_part_iou
                wandb_log["test_mean_category_iou"]= test_mean_category_iou
                wandb_log["test_best_mean_category_iou"]= test_best_mean_category_iou
                wandb_log["test_loss"] = test_loss
                wandb_log['test_time_per_epoch'] = test_duration.total_seconds()
                wandb_log['train_time_per_epoch'] = train_duration.total_seconds()
                wandb.log(wandb_log)

            lr_scheduler.step()

    if rank == 0:
        logger.write(f'Final highest Mean Category IoU: {test_best_mean_category_iou} at epoch {best_epoch}!', rank=rank)
        logger.write('End of DDP finetuning on %s ...' % args.ft_dataset, rank=rank)
        wandb.finish()
    cleanup()


def test(rank, model, test_loader, criterion):
    model.eval()

    num_part_classes = test_loader.dataset.seg_num_all
    seg_start_index = test_loader.dataset.seg_start_index

    test_loss = AverageMeter()
    points_seg_acc = AccuracyMeter()
   
    part2correct, part2total = torch.zeros(num_part_classes, device=f'cuda:{rank}'), torch.zeros(num_part_classes, device=f'cuda:{rank}')
    shape_ious = {obj:[] for obj in category2part.keys()}

    for points, obj_label, partseg_label in test_loader:
        # obj_label: [batch, 1]
        batch_size, num_points, _ = points.size()
        label_onehot = torch.zeros(batch_size, args.num_obj_classes, device=f'cuda:{rank}')
        for i in range(batch_size):
            label_onehot[i, obj_label[i]] = 1
        # partseg_label: [batch, num_points]
        partseg_label = partseg_label - seg_start_index

        points, partseg_label = points.to(rank), partseg_label.to(rank)
        # partseg_pred: [batch_size, num_points, num_part_classes]
        partseg_pred = model(points, label_onehot)
        # partseg_pred = model(points)
        loss = criterion(partseg_pred.reshape(-1, num_part_classes), partseg_label.reshape(-1))
        test_loss.update(loss, batch_size)

        # NOTE: the following loop ensures the predictions happen on the target object category
        refined_pred = torch.zeros(batch_size, num_points, dtype=torch.int32, device=f'cuda:{rank}')
        for i in range(batch_size):
            # partseg_label[i, 0] is a tensor, it should be converted to an integer to serve as an index
            idx = partseg_label[i, 0].item()
            cat = part2category[idx]
            logits = partseg_pred[i, :, :]
            refined_pred[i, :] = torch.argmax(logits[:, category2part[cat]], dim=1) + category2part[cat][0]

        pos = points_seg_acc.pos_count(refined_pred, partseg_label)
        points_seg_acc.update(pos, batch_size*num_points-pos, batch_size*num_points)

        for i in range(num_part_classes):
            # torch.eq expects one of 
            #   * (Tensor input, Tensor other, *, Tensor out)
            #   * (Tensor input, Number other, *, Tensor out)
            part2correct[i] += torch.eq(refined_pred, i).sum()
            part2total[i] += torch.eq(partseg_label, i).sum()

        for i in range(batch_size):
            pred, gt = refined_pred[i, :], partseg_label[i, :]
            part = gt[0].item()    # `0th` point in the point cloud belongs to one part, gt[0].item() converts to a number to serve as an index
            cat = part2category[part]     # find the object category of this part
            part_ious = [.0 for _ in range(len(category2part[cat]))]

            for j, part in enumerate(category2part[cat]):
                if j == 0:
                    start_id = category2part[cat][0]
                if torch.logical_or(torch.eq(gt, part), torch.eq(pred, part)).sum() == 0:
                    part_ious[part - start_id] = 1
                else:
                    intersection = torch.logical_and(torch.eq(gt, part), torch.eq(pred, part)).sum()
                    union = torch.logical_or(torch.eq(gt, part), torch.eq(pred, part)).sum()
                    part_ious[part - start_id] = intersection / union
            shape_ious[cat].append(torch.mean(torch.tensor(part_ious)))

    all_part_ious = []
    for cat in shape_ious.keys():
        for iou in shape_ious[cat]:
            all_part_ious.append(iou)
        shape_ious[cat] = torch.mean(torch.tensor(shape_ious[cat]))
    mean_part_iou = torch.mean(torch.tensor(all_part_ious))
    
    category_ious = [cat_iou for cat_iou in shape_ious.values()]
    mean_category_iou = torch.mean(torch.tensor(category_ious))

    mean_part_acc = torch.mean(part2correct/part2total)
    point_level_acc = points_seg_acc.num_pos/points_seg_acc.total

    return mean_part_iou.item(), mean_category_iou.item(), mean_part_acc.item(), point_level_acc.item(), test_loss.avg.item()


if __name__ == "__main__":
    init(args.proj_name, args.exp_name, args.main_program, args.model_name)

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    
    logger_name = args.proj_name
    log_path = os.path.join('runs', args.proj_name, args.exp_name)
    log_file = f'{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.log'

    logger = Logger(logger_name=logger_name, log_path=log_path, log_file=log_file)

    if args.cuda:
        num_devices = torch.cuda.device_count()
        if num_devices > 1:
            logger.write('%d GPUs are available and %d of them are used. Ready for DDP finetuning' % (num_devices, args.world_size), rank=0)
            logger.write(str(args), rank=0)
            # Set seed for generating random numbers for all GPUs, and 
            # torch.cuda.manual_seed() is insufficient to get determinism for all GPUs
            mp.spawn(main, args=(logger_name, log_path, log_file), nprocs=args.world_size)
        else:
            logger.write('Only one GPU is available, the process will be much slower! Exit', rank=0)
    else:
        logger.write('CUDA is unavailable! Exit', rank=0)
