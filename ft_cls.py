import os
import wandb
from datetime import datetime

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from torch.utils.data import DataLoader
from datasets.data import ModelNet40SVM, ScanObjectNNSVM

from utils import build_ft_cls

import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts, StepLR, ReduceLROnPlateau
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
from torch.nn import CrossEntropyLoss

from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler

from utils import init, Logger, AverageMeter, AccuracyMeter
from parser import args


def setup(rank):
    # initialization for distributed training on multiple GPUs
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

    # NOTE: only write logs of the results obtained by
    # the first gpu device whose rank=0, otherwise produce duplicate logs
    logger = Logger(logger_name=logger_name, log_path=log_path, log_file=log_file)

    setup(rank)

    if 'ModelNet40' in args.ft_dataset:
        train_set = ModelNet40SVM(partition='train', num_points=args.num_ft_points)
        test_set = ModelNet40SVM(partition='test', num_points=args.num_ft_points)
    elif 'ScanObjectNN' in args.ft_dataset:
        train_set = ScanObjectNNSVM(partition='train', num_points=args.num_ft_points)
        test_set = ScanObjectNNSVM(partition='test', num_points=args.num_ft_points)
    else:
        raise NotImplementedError('Please choose dataset among [ModelNet40, ScanObjectNN]')

    train_sampler = DistributedSampler(train_set, num_replicas=args.world_size, rank=rank)
    test_sampler = DistributedSampler(test_set, num_replicas=args.world_size, rank=rank)

    assert args.batch_size % args.world_size == 0 and args.test_batch_size % args.world_size == 0, \
        'Argument `batch_size` should be divisible by `world_size`'
    samples_per_gpu = args.batch_size // args.world_size
    test_samples_per_gpu = args.test_batch_size // args.world_size

    train_loader = DataLoader(
        train_set,
        sampler=train_sampler,
        batch_size=samples_per_gpu,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False)
    test_loader = DataLoader(
        test_set,
        sampler=test_sampler,
        batch_size=test_samples_per_gpu,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False)

    model = build_ft_cls(rank=rank)
    model_ddp = DDP(model, device_ids=[rank], find_unused_parameters=True)

    # ----- load pretrained model
    assert args.resume, 'Finetuning ViPFormer requires pretrained model weights'
    map_location = torch.device('cuda:%d' % rank)
    pretrained = torch.load(args.pc_model_file, map_location=map_location)
    # append `module.` before key
    pretrained = {"module."+key: value for key, value in pretrained.items()}

    model_ddp.load_state_dict(
        pretrained,
        strict=False)    # it is necessary to set `strict`=False
    
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

    # PyTorch 1.11.0 -> `label_smoothing` in CrossEntropyLoss
    #   https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
    criterion = CrossEntropyLoss(label_smoothing=0.2)
    scaler = GradScaler()
    logger.write('Start DDP finetuning on %s ...' % args.ft_dataset, rank=rank)

    ft_test_best_acc = .0
    best_epoch = 0
    for epoch in range(args.epochs):
        # ------ Train
        model_ddp.train()
        # required by DistributedSampler
        train_sampler.set_epoch(epoch)
        test_sampler.set_epoch(epoch)

        # average losses across all scanned batches within an epoch
        train_loss = AverageMeter()
        acc_meter = AccuracyMeter()

        start_train = datetime.now()
        for i, (points,label) in enumerate(train_loader):
            optimizer.zero_grad(set_to_none=True)

            with autocast():
                # points.shape: [B, N, C]
                batch_size = points.shape[0]
                points = points.to(rank)
                label = label.to(rank)

                pred_classes = model_ddp(points)
                # NOTE: here `loss` has already been averaged by `batch_size`
                # ------ ref: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
                #           The `pred_classes` is expected to contain `raw`, `unnormalized` scores for each class
                #           label.squeeze() is a `batch_size`-Dimension class index tensor
                loss = criterion(pred_classes, label.squeeze())

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss.update(loss, n=batch_size)
            # x.argmax: low bound begins with 0
            pred_idx = pred_classes.argmax(dim=1)
            pos = acc_meter.pos_count(pred_idx, label.squeeze())
            acc_meter.update(pos, batch_size-pos, n=batch_size)

            if i % args.print_freq == 0:
                logger.write(f'Epoch: {epoch}/{args.epochs}, Batch: {i}/{len(train_loader)}, '
                             f'Loss: {loss.item()}, Acc: {pos.item()/batch_size}', rank=rank)
        train_duration = datetime.now() - start_train

        # ------ Test
        with torch.no_grad():
            ft_train_acc = acc_meter.num_pos.item() / acc_meter.total

            logger.write('Start testing on the %s test set ...' % args.ft_dataset, rank=rank)
            
            test_start = datetime.now()
            ft_test_loss, ft_test_acc = test(rank, test_loader, model_ddp, criterion)
            test_duration = datetime.now() - test_start

            logger.write(f'Test on {args.ft_dataset}, Epoch: {epoch}/{args.epochs}, Acc: {ft_test_acc}, Loss: {ft_test_loss}', rank=rank)

            if rank == 0:
                if ft_test_acc > ft_test_best_acc:
                    ft_test_best_acc = ft_test_acc
                    best_epoch = epoch
                    logger.write(f'Finding new best test score: {ft_test_best_acc} at epoch {best_epoch}!', rank=rank)
                    logger.write('Saving best model ...', rank=rank)
                    save_path = os.path.join('runs', args.proj_name, args.exp_name, 'models', 'model_best.pth')
                    torch.save(model_ddp.module.state_dict(), save_path)

                wandb_log = dict()
                # NOTE get_lr() vs. get_last_lr(), which should be used? The answer is get_last_lr()
                # https://discuss.pytorch.org/t/whats-the-difference-between-get-lr-and-get-last-lr/121681
                if args.scheduler == 'coswarm':
                    wandb_log['learning_rate'] = lr_scheduler.get_lr()[0]
                else:
                    wandb_log['learning_rate'] = lr_scheduler.get_last_lr()[0]
                wandb_log['ft_train_loss'] = train_loss.avg.item()
                wandb_log['ft_train_acc'] = ft_train_acc
                wandb_log['ft_test_loss'] = ft_test_loss
                wandb_log['ft_test_acc'] = ft_test_acc
                wandb_log['ft_test_best_acc'] = ft_test_best_acc
                wandb_log['test_time_per_epoch'] = test_duration.total_seconds()
                wandb_log['train_time_per_epoch'] = train_duration.total_seconds()
                wandb.log(wandb_log)

            # adjust learning rate before a new epoch
            lr_scheduler.step()

    if rank == 0:
        logger.write(f'Final best finetuning score: {ft_test_best_acc} at epoch {best_epoch}!', rank=rank)
        logger.write('End of DDP finetuning on %s ...' % args.ft_dataset, rank=rank)
        wandb.finish()

    cleanup()


def test(rank, test_loader, model_ddp, criterion):
    model_ddp.eval()

    test_loss = AverageMeter()
    acc_meter = AccuracyMeter()  
    for (points, label) in test_loader:
        batch_size = points.shape[0]

        points = points.to(rank)
        label = label.to(rank)

        # pred_classes: [batch, num_classes]
        pred_classes = model_ddp(points)
        loss = criterion(pred_classes, label.squeeze())
        test_loss.update(loss, n=batch_size)
        # pred_idx: a batch-Dimension tensor
        pred_idx = pred_classes.argmax(dim=1)
        pos = acc_meter.pos_count(pred_idx, label.squeeze())
        acc_meter.update(pos, batch_size-pos, n=batch_size)
    
    ft_test_loss = test_loss.avg.item()
    ft_test_acc = acc_meter.num_pos.item() / acc_meter.total

    return ft_test_loss, ft_test_acc


if '__main__' == __name__:
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
        