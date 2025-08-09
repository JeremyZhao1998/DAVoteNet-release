import os
import numpy as np
import argparse

import torch
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from pc_datasets import PCDetectionDataset
from models.votenet import VoteNet
from models.votenet_criterion import VoteNetCriterion
from models.evaluator import Evaluator
from engine import train_one_epoch, teach_one_epoch, evaluate
from utils.distributed_utils import init_distributed_mode, get_rank


def parse_args():
    parser = argparse.ArgumentParser()
    # Mode
    parser.add_argument('--mode', default='source_only')
    # Dataset configuration
    parser.add_argument('--data_root', default='<data_root>', help='Data root')
    parser.add_argument('--src_dataset', default='scannet', help='Dataset name')
    parser.add_argument('--tgt_dataset', default='sunrgbd', help='Dataset name')
    parser.add_argument('--num_points_preload', type=int, default=100000, help='Preload point Number')
    parser.add_argument('--num_points', type=int, default=40000, help='Point Number')
    parser.add_argument('--categories', nargs='+', type=str,
                        default=['bed', 'bookshelf', 'cabinet', 'chair', 'desk', 'garbage_can', 'lamp', 'night_stand',
                                 'shelf', 'sink', 'sofa', 'table', 'toilet', 'tv', 'others'], help='Categories')
    parser.add_argument('--ignored_categories', nargs='+', type=str, default=['others'], help='Ignored categories')
    parser.add_argument('--max_num_obj', type=int, default=64, help='Max number of objects')
    parser.add_argument('--axis_aligned', type=int, default=1, help='Use axis aligned bounding boxes')
    parser.add_argument('--few_shot', type=int, default=-1, help='Few shot fine-tuning')
    # Model configuration
    parser.add_argument('--model', default='votenet', help='Model name')
    parser.add_argument('--backbone', default='pointnet2', help='Backbone name')
    # # VoteNet configuration
    parser.add_argument('--num_proposals', type=int, default=64, help='Proposal number')
    parser.add_argument('--vote_factor', type=int, default=1, help='Vote factor')
    parser.add_argument('--gt_vote_factor', type=int, default=3, help='GT vote factor')
    parser.add_argument('--sampling', default='vote_fps', help='Vote cluster sampling: vote_fps, seed_fps, random')
    parser.add_argument('--use_color', type=int, default=1, help='Use RGB color in input')
    parser.add_argument('--use_height', type=int, default=1, help='Use height signal in input')
    # Training configuration
    parser.add_argument('--start_epoch', type=int, default=0, help='Start epoch')
    parser.add_argument('--epoch_num', type=int, default=90, help='Epoch number')
    parser.add_argument('--epoch_eval', type=int, default=1, help='Evaluate every [epoch_eval] epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch Size')
    parser.add_argument('--eval_batch_size', type=int, default=64, help='Evaluation Batch Size')
    # # VoteNet training configuration
    parser.add_argument('--lr', type=float, default=0.008, help='Initial learning rate')
    parser.add_argument('--lr_decay_steps', nargs='+', type=int, default=[65, 80], help='lr decay epochs')
    parser.add_argument('--lr_decay_rates', nargs='+', type=float, default=[0.1, 0.1], help='lr decay rates')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Optimization L2 weight decay')
    # Teaching configuration
    parser.add_argument('--obj_threshold', type=float, default=0.95, help='Objectness threshold for pseudo labels')
    parser.add_argument('--sem_threshold', type=float, default=0.95, help='Semantic threshold for pseudo labels')
    parser.add_argument('--alpha_ema', type=float, default=0.9996, help='Exponential moving average weight')
    parser.add_argument('--epoch_switch', type=int, default=1000, help='Epoch to switch teacher/student')
    parser.add_argument('--oracle', type=int, default=0, help='Use oracle bounding boxes for training')
    # Loss coefficients
    parser.add_argument('--use_focal_loss', type=int, default=0, help='Whether to use focal loss')
    parser.add_argument('--coef_src', type=float, default=1.0, help='Domain loss coefficient')
    parser.add_argument('--coef_tgt', type=float, default=1.0, help='Domain loss coefficient')
    parser.add_argument('--coef_vote_loss', type=float, default=10.0, help='Vote loss coefficient')
    parser.add_argument('--coef_objectness_loss', type=float, default=5.0, help='Objectness loss coefficient')
    parser.add_argument('--coef_center_loss', type=float, default=10.0, help='Center loss coefficient')
    parser.add_argument('--coef_heading_cls_loss', type=float, default=1.0, help='Heading cls loss coefficient')
    parser.add_argument('--coef_heading_reg_loss', type=float, default=10.0, help='Heading reg loss coefficient')
    parser.add_argument('--coef_size_cls_loss', type=float, default=1.0, help='Size cls loss coefficient')
    parser.add_argument('--coef_size_reg_loss', type=float, default=10.0 / 3.0, help='Size reg loss coefficient')
    parser.add_argument('--coef_cls_loss', type=float, default=1.0, help='Semantic cls loss coefficient')
    # Output configuration
    parser.add_argument('--output_dir', default='<output_dir>')
    parser.add_argument('--ap_iou_threshold', type=float, default=0.25, help='AP IoU threshold')
    # Other configurations
    parser.add_argument('--ckpt_detector', default=None, help='Resume detector checkpoint path')
    parser.add_argument('--save_optimizer', type=int, default=0, help='Save optimizer checkpoint')
    parser.add_argument('--ckpt_optimizer', default=None, help='Resume optimizer checkpoint path')
    parser.add_argument('--device', default='cuda', help='Device to use')
    parser.add_argument('--seed', type=int, default=1618, help='Random seed')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')
    parser.add_argument('--flush', type=int, default=1, help='Flush the output log')
    args_parsed = parser.parse_args()
    args_parsed.axis_aligned = bool(args_parsed.axis_aligned)
    args_parsed.use_color = bool(args_parsed.use_color)
    args_parsed.use_height = bool(args_parsed.use_height)
    args_parsed.use_focal_loss = bool(args_parsed.use_focal_loss)
    args_parsed.oracle = bool(args_parsed.oracle)
    args_parsed.flush = bool(args_parsed.flush)
    return args_parsed


def set_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_param_num(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total Parameters': total_num, 'Trainable': trainable_num}


def build_datasets(dataset_name, few_shot=-1):
    train_dataset = PCDetectionDataset(
        data_root=args.data_root,
        dataset_name=dataset_name,
        categories=args.categories,
        split_set='train',
        num_points=args.num_points_preload,
        max_num_obj=args.max_num_obj,
        axis_aligned=args.axis_aligned,
        augment=False,
        use_color=args.use_color,
        use_height=args.use_height,
        few_shot=few_shot,
        return_votes=(args.model == 'votenet')
    )
    val_dataset = PCDetectionDataset(
        data_root=args.data_root,
        dataset_name=dataset_name,
        categories=args.categories,
        split_set='val',
        num_points=args.num_points,
        max_num_obj=args.max_num_obj,
        axis_aligned=args.axis_aligned,
        augment=False,
        use_color=args.use_color,
        use_height=args.use_height,
        return_votes=(args.model == 'votenet')
    )
    return train_dataset, val_dataset


def build_dataloader_from_dataset(dataset, split_set):
    def worker_init_fn(worker_id):
        np.random.seed(np.random.get_state()[1][0] + worker_id)

    if args.distributed:
        sampler = DistributedSampler(dataset, shuffle=(split_set == 'train'), drop_last=(split_set == 'train'))
    else:
        sampler = None
    loader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size if split_set == 'train' else args.eval_batch_size,
        shuffle=(split_set == 'train' if sampler is None else None),
        drop_last=(split_set == 'train'),
        num_workers=args.num_workers,
        pin_memory=True,
        sampler=sampler,
        worker_init_fn=worker_init_fn
    )
    return loader


def build_model(num_classes, num_heading_areas, mean_sizes):
    if args.model == 'votenet':
        detector = VoteNet(
            num_classes=num_classes,
            num_heading_areas=num_heading_areas,
            mean_sizes=mean_sizes,
            backbone=args.backbone,
            axis_aligned=args.axis_aligned,
            num_proposals=args.num_proposals,
            input_feature_dim=int(args.use_color) * 3 + int(args.use_height) * 1,
            vote_factor=args.vote_factor,
            gt_vote_factor=args.gt_vote_factor,
            sampling=args.sampling
        ).to(device)
        criterion = VoteNetCriterion(
            num_classes=num_classes,
            num_heading_areas=num_heading_areas,
            mean_sizes=mean_sizes,
            use_focal_loss=args.use_focal_loss,
            gt_vote_factor=args.gt_vote_factor,
            coef_vote_loss=args.coef_vote_loss,
            coef_objectness_loss=args.coef_objectness_loss,
            coef_center_loss=args.coef_center_loss,
            coef_heading_cls_loss=args.coef_heading_cls_loss,
            coef_heading_reg_loss=args.coef_heading_reg_loss,
            coef_size_cls_loss=args.coef_size_cls_loss,
            coef_size_reg_loss=args.coef_size_reg_loss,
            coef_cls_loss=args.coef_cls_loss,
            device=device
        ).to(device)
    else:
        raise NotImplementedError('Model not implemented: {}'.format(args.model))
    if args.ckpt_detector is not None:
        detector.load_state_dict(torch.load(args.ckpt_detector, weights_only=True, map_location=device))
    if args.distributed:
        detector = DistributedDataParallel(detector, device_ids=[args.gpu])
        detector = torch.nn.SyncBatchNorm.convert_sync_batchnorm(detector)
    return detector, criterion


def build_optimizer(detector):
    if args.model == 'votenet':
        param_dicts = [{'params': detector.parameters(), 'lr': args.lr}]
        optimizer = optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
        if args.ckpt_optimizer is not None:
            ckpt = torch.load(args.ckpt_optimizer, map_location=device)
            args.start_epoch = ckpt['epoch'] + 1
            optimizer.load_state_dict(ckpt['opt'])
    else:
        raise NotImplementedError('Model not implemented: {}'.format(args.model))
    return optimizer


def save_model(detector, optimizer, epoch, m_ap, best_ap, name='model'):
    detector_state_dict = detector.module.state_dict() if args.distributed else detector.state_dict()
    torch.save(detector_state_dict, os.path.join(args.output_dir, name + '_last.pth'))
    if optimizer is not None:
        torch.save({'epoch': epoch, 'opt': optimizer.state_dict()}, os.path.join(args.output_dir, 'opt_last.pth'))
    if m_ap > best_ap:
        best_ap = m_ap
        print('Saving best model with mAP: ', m_ap, flush=args.flush)
        torch.save(detector_state_dict, os.path.join(args.output_dir, 'model_best.pth'))
    return best_ap


def source_only():
    # Prepare datasets
    src_train_dataset, _ = build_datasets(args.src_dataset)
    if args.few_shot > 0:
        mean_sizes = src_train_dataset.mean_sizes
        src_train_dataset, _ = build_datasets(args.tgt_dataset, few_shot=args.few_shot)
        src_train_dataset.mean_sizes = mean_sizes
    _, tgt_val_dataset = build_datasets(args.tgt_dataset)
    tgt_val_dataset.mean_sizes = src_train_dataset.mean_sizes
    # Build dataloaders
    src_train_loader = build_dataloader_from_dataset(src_train_dataset, 'train')
    tgt_val_loader = build_dataloader_from_dataset(tgt_val_dataset, 'val')
    # Build model, criterion and evaluator
    detector, criterion = build_model(
        num_classes=src_train_dataset.num_classes,
        num_heading_areas=src_train_dataset.num_heading_areas,
        mean_sizes=src_train_dataset.mean_sizes
    )
    print('Detector params: ', get_param_num(detector), flush=args.flush)
    evaluator = Evaluator(
        num_classes=src_train_dataset.num_classes,
        categories=src_train_dataset.category_dict,
        ignored_categories=args.ignored_categories,
        iou_threshold=args.ap_iou_threshold
    )
    # Build optimizer
    optimizer = build_optimizer(detector)
    # Train the model
    best_ap = 0.0
    for epoch in range(args.start_epoch, args.epoch_num):
        # Set the epoch for the sampler
        if args.distributed and hasattr(src_train_loader.sampler, 'set_epoch'):
            src_train_loader.sampler.set_epoch(epoch)
        detector = train_one_epoch(
            train_loader=src_train_loader,
            num_points=args.num_points,
            detector=detector,
            criterion=criterion,
            optimizer=optimizer,
            epoch=epoch,
            base_lr=args.lr,
            lr_decay_steps=args.lr_decay_steps,
            lr_decay_rates=args.lr_decay_rates,
            device=device,
            flush=args.flush
        )
        if (epoch + 1) % args.epoch_eval == 0:
            m_ap = evaluate(
                val_loader=tgt_val_loader,
                detector=detector,
                criterion=criterion,
                evaluator=evaluator,
                device=device,
                flush=args.flush
            )
            best_ap = save_model(detector, optimizer, epoch, m_ap, best_ap)
    print('Source only training finished. Best mAP: {}'.format(best_ap), flush=args.flush)


def teaching():
    # Prepare datasets
    src_train_dataset, _ = build_datasets(args.src_dataset)
    tgt_train_dataset, tgt_val_dataset = build_datasets(args.tgt_dataset)
    mean_sizes, categories = src_train_dataset.mean_sizes, src_train_dataset.category_dict
    tgt_train_dataset.mean_sizes = tgt_val_dataset.mean_sizes = mean_sizes
    num_classes, num_heading_areas = src_train_dataset.num_classes, src_train_dataset.num_heading_areas
    # Build dataloaders
    src_train_loader = build_dataloader_from_dataset(src_train_dataset, 'train')
    tgt_train_loader = build_dataloader_from_dataset(tgt_train_dataset, 'train')
    tgt_val_loader = build_dataloader_from_dataset(tgt_val_dataset, 'val')
    # Build model, criterion and evaluator
    student, criterion_src = build_model(num_classes, num_heading_areas, mean_sizes)
    print('Detector parameters: ', get_param_num(student), flush=args.flush)
    teacher, criterion_tgt = build_model(num_classes, num_heading_areas, mean_sizes)
    criterion_tgt.coef_vote_loss = criterion_tgt.coef_objectness_loss = 0.0
    evaluator = Evaluator(num_classes, categories, args.ignored_categories, args.ap_iou_threshold)
    # Build optimizer
    optimizer = build_optimizer(student)
    # Train the model
    best_ap = 0.0
    for epoch in range(args.start_epoch, args.epoch_num):
        student, teacher = teach_one_epoch(
            src_loader=src_train_loader,
            tgt_loader=tgt_train_loader,
            num_points=args.num_points,
            student=student,
            teacher=teacher,
            criterion_src=criterion_src,
            criterion_tgt=criterion_tgt,
            obj_threshold=args.obj_threshold,
            sem_threshold=args.sem_threshold,
            alpha_ema=args.alpha_ema,
            coef_src=args.coef_src,
            coef_tgt=args.coef_tgt,
            optimizer=optimizer,
            epoch=epoch,
            base_lr=args.lr,
            lr_decay_steps=args.lr_decay_steps,
            lr_decay_rates=args.lr_decay_rates,
            oracle=args.oracle,
            device=device,
            flush=args.flush
        )
        if (epoch + 1) % args.epoch_eval == 0:
            m_ap_stu = evaluate(
                val_loader=tgt_val_loader,
                detector=student,
                criterion=criterion_tgt,
                evaluator=evaluator,
                device=device,
                flush=args.flush
            )
            best_ap = save_model(student, optimizer, epoch, m_ap_stu, best_ap, name='stu_model')
            if not args.oracle:
                m_ap_tch = evaluate(
                    val_loader=tgt_val_loader,
                    detector=teacher,
                    criterion=criterion_tgt,
                    evaluator=evaluator,
                    device=device,
                    flush=args.flush
                )
                best_ap = save_model(teacher, None, epoch, m_ap_tch, best_ap, name='tch_model')
        if (epoch + 1) % args.epoch_switch == 0 and not args.oracle:
            student, teacher = teacher, student
    print('Teaching finished. Best mAP: {}'.format(best_ap), flush=args.flush)


def evaluation():
    _, tgt_val_dataset = build_datasets(args.tgt_dataset)
    src_train_dataset, _ = build_datasets(args.src_dataset)
    tgt_val_dataset.mean_sizes = src_train_dataset.mean_sizes
    tgt_val_loader = build_dataloader_from_dataset(tgt_val_dataset, 'val')
    detector, criterion = build_model(
        num_classes=tgt_val_dataset.num_classes,
        num_heading_areas=tgt_val_dataset.num_heading_areas,
        mean_sizes=tgt_val_dataset.mean_sizes
    )
    evaluator = Evaluator(tgt_val_dataset.num_classes, tgt_val_dataset.category_dict,
                          args.ignored_categories, args.ap_iou_threshold)
    evaluate(
        val_loader=tgt_val_loader,
        detector=detector,
        criterion=criterion,
        evaluator=evaluator,
        device=device,
        flush=args.flush
    )


if __name__ == '__main__':
    args = parse_args()
    if not os.path.exists(os.path.abspath(args.output_dir)):
        os.makedirs(os.path.abspath(args.output_dir))
    device = torch.device(args.device)
    set_random_seed(args.seed + get_rank())
    init_distributed_mode(args)
    print('-------------------------Args---------------------------', flush=args.flush)
    for arg_key, arg_value in vars(args).items():
        print('{}: {}'.format(arg_key, arg_value), flush=args.flush)
    print('--------------------------------------------------------', flush=args.flush)
    if args.mode == 'source_only':
        source_only()
    elif args.mode == 'teaching':
        teaching()
    elif args.mode == 'eval':
        evaluation()
