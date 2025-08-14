import time
import datetime
from itertools import cycle
from copy import deepcopy
import torch
from torch.utils.data import DataLoader

from models import VoteNet, VoteNetCriterion, Evaluator


def adjust_learning_rate(optimizer, epoch, base_lr, lr_decay_steps, lr_decay_rates):
    lr = base_lr
    for i, lr_decay_epoch in enumerate(lr_decay_steps):
        if epoch >= lr_decay_epoch:
            lr *= lr_decay_rates[i]
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def box_num_cnt(cnt_dict, predictions):
    batch_ids, box_ids = torch.where(predictions['label_masks'])
    categories = predictions['categories'][batch_ids, box_ids].detach().cpu().numpy()
    for i, c in enumerate(categories):
        cnt_dict[c] += 1
    return cnt_dict


def batch_data_to_device(batch_data_label, device):
    for key, value in batch_data_label.items():
        if torch.is_tensor(value):
            batch_data_label[key] = value.to(device)
    return batch_data_label


def apply_subsample(batch_data_label, num_points, choices=None):
    batch_data_label_new = deepcopy(batch_data_label)
    point_clouds = batch_data_label['point_clouds']
    if 'point_votes' not in batch_data_label and 'point_votes_mask' not in batch_data_label:
        point_votes, point_votes_mask = None, None
    else:
        point_votes, point_votes_mask = batch_data_label['point_votes'], batch_data_label['point_votes_mask']
    batch_size, num_points_orig = point_clouds.shape[0], point_clouds.shape[1]
    if num_points_orig >= num_points:
        point_clouds_new = torch.zeros_like(point_clouds)[:, :num_points, :]
        if point_votes is not None and point_votes_mask is not None:
            point_votes_new = torch.zeros_like(point_votes)[:, :num_points, :]
            point_votes_mask_new = torch.zeros_like(point_votes_mask)[:, :num_points]
        else:
            point_votes_new, point_votes_mask_new = None, None
        for i in range(batch_size):
            if choices is None:
                choice = torch.randperm(num_points_orig)[:num_points]
            else:
                choice = choices[i]
            point_clouds_new[i] = point_clouds[i, choice, :]
            if point_votes_new is not None and point_votes_mask_new is not None:
                point_votes_new[i] = point_votes[i, choice, :]
                point_votes_mask_new[i] = point_votes_mask[i, choice]
        batch_data_label_new['point_clouds'] = point_clouds_new
        if point_votes_new is not None and point_votes_mask_new is not None:
            batch_data_label_new['point_votes'] = point_votes_new
            batch_data_label_new['point_votes_mask'] = point_votes_mask_new
    return batch_data_label_new


def train_one_epoch(train_loader: DataLoader,
                    num_points: int,
                    detector: VoteNet,
                    criterion: VoteNetCriterion,
                    optimizer: torch.optim.Optimizer,
                    epoch: int,
                    base_lr: float,
                    lr_decay_steps: list,
                    lr_decay_rates: list,
                    device: torch.device,
                    print_freq: int = 10,
                    flush: bool = False):
    start_time = time.time()
    adjust_learning_rate(optimizer, epoch, base_lr, lr_decay_steps, lr_decay_rates)
    detector.train()
    epoch_loss, epoch_obj_acc = 0, 0
    for batch_idx, (batch_data_label) in enumerate(train_loader):
        batch_data_label = apply_subsample(batch_data_label, num_points)
        batch_data_label = batch_data_to_device(batch_data_label, device)
        # Forward pass
        optimizer.zero_grad()
        outputs = detector(batch_data_label['point_clouds'])
        # Compute loss and gradients, update parameters.
        loss, loss_dict = criterion(outputs, batch_data_label)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        if (batch_idx + 1) % print_freq == 0:
            print('Epoch %d batch: %03d / %03d, loss: %f'
                  % (epoch, batch_idx + 1, len(train_loader), loss.item()), flush=flush)
    end_time = time.time()
    time_str = str(datetime.timedelta(seconds=int(end_time - start_time)))
    print('Training epoch %d done. Epoch loss: %f ' % (epoch, epoch_loss / len(train_loader)), flush=flush)
    print('Time cost: {}'.format(time_str), flush=flush)
    return detector


@torch.no_grad()
def calculate_rv_info(train_loader: DataLoader,
                      num_points: int,
                      detector: VoteNet,
                      device: torch.device,
                      print_freq: int = 10,
                      flush: bool = False):
    start_time = time.time()
    detector.eval()
    proposal_features_all, pred_categories_all = [], []
    for batch_idx, (batch_data_label) in enumerate(train_loader):
        batch_data_label = apply_subsample(batch_data_label, num_points)
        batch_data_label = batch_data_to_device(batch_data_label, device)
        outputs = detector(batch_data_label['point_clouds'])
        model = detector if hasattr(detector, 'post_process') else detector.module
        predictions = model.post_process(
            batch_data_label['point_clouds'],
            outputs,
            obj_threshold=0.5,
            sem_threshold=0.5,
        )
        label_masks = predictions['label_masks']
        label_masks = label_masks.view(-1)
        proposal_features = outputs['proposal_features']
        proposal_features = proposal_features.view(-1, proposal_features.shape[-1])[label_masks]
        pred_categories = predictions['categories']
        pred_categories = pred_categories.view(-1)[label_masks]
        proposal_features_all.append(proposal_features)
        pred_categories_all.append(pred_categories)
        if (batch_idx + 1) % print_freq == 0:
            print('Calculating rv info of batch: %03d / %03d'
                  % (batch_idx + 1, len(train_loader)), flush=flush)
    proposal_features_all = torch.cat(proposal_features_all, dim=0)
    pred_categories_all = torch.cat(pred_categories_all, dim=0)
    end_time = time.time()
    time_str = str(datetime.timedelta(seconds=int(end_time - start_time)))
    print(f'Calculating rv info done, select {proposal_features_all.shape[0]} proposals', flush=flush)
    print('Time cost: {}'.format(time_str), flush=flush)
    return proposal_features_all, pred_categories_all


def teach_one_epoch(src_loader: DataLoader,
                    tgt_loader: DataLoader,
                    num_points: int,
                    student: VoteNet,
                    teacher: VoteNet,
                    criterion_src: VoteNetCriterion,
                    criterion_tgt: VoteNetCriterion,
                    optimizer: torch.optim.Optimizer,
                    obj_threshold: float,
                    sem_threshold: float,
                    reliable_voting: bool,
                    alpha_ema: float,
                    coef_src: float,
                    coef_tgt: float,
                    epoch: int,
                    base_lr: float,
                    lr_decay_steps: list,
                    lr_decay_rates: list,
                    oracle: bool,
                    device: torch.device,
                    print_freq: int = 20,
                    flush: bool = False):
    start_time = time.time()
    adjust_learning_rate(optimizer, epoch, base_lr, lr_decay_steps, lr_decay_rates)
    student.train()
    teacher.eval()
    epoch_loss, epoch_obj_acc = 0, 0
    gt_box_num_cnt = {cls: 0 for cls in range(criterion_src.num_classes)}
    pseudo_box_num_cnt = {cls: 0 for cls in range(criterion_src.num_classes)}
    for batch_idx, (data_label_src, data_label_tgt) in enumerate(zip(cycle(src_loader), tgt_loader)):
        # Prepare data
        data_label_src = apply_subsample(data_label_src, num_points)
        data_label_src = batch_data_to_device(data_label_src, device)
        data_label_tgt = apply_subsample(data_label_tgt, num_points)
        data_label_tgt = batch_data_to_device(data_label_tgt, device)
        # Data forward pass to student model
        optimizer.zero_grad()
        outputs_src = student(data_label_src['point_clouds'])
        outputs_tgt = student(data_label_tgt['point_clouds'])
        # Generate pseudo labels for target data
        if oracle:
            supervision = data_label_tgt
        else:
            with torch.no_grad():
                outputs_tch = teacher(data_label_tgt['point_clouds'])
                model = teacher if hasattr(teacher, 'post_process') else teacher.module
                supervision = model.post_process(
                    data_label_tgt['point_clouds'],
                    outputs_tch,
                    obj_threshold=obj_threshold,
                    sem_threshold=sem_threshold,
                    proposal_feature=outputs_tch['proposal_features'] if reliable_voting else None,
                    get_votes=True
                )
        # Compute loss and gradients, update parameters.
        loss_src, loss_dict_src = criterion_src(outputs_src, data_label_src)
        loss_tgt, loss_dict_tgt = criterion_tgt(outputs_tgt, supervision)
        loss = coef_src * loss_src + coef_tgt * loss_tgt
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        # Count box number
        gt_box_num_cnt = box_num_cnt(gt_box_num_cnt, data_label_tgt)
        pseudo_box_num_cnt = box_num_cnt(pseudo_box_num_cnt, supervision)
        # EMA update teacher
        if not oracle:
            with torch.no_grad():
                state_dict, student_state_dict = teacher.state_dict(), student.state_dict()
                for key, value in state_dict.items():
                    state_dict[key] = alpha_ema * value + (1 - alpha_ema) * student_state_dict[key].detach()
                teacher.load_state_dict(state_dict)
        if (batch_idx + 1) % print_freq == 0:
            print('Epoch %d batch: %03d / %03d, loss_src: %f, loss_tgt: %f'
                  % (epoch, batch_idx + 1, len(tgt_loader), loss_src.item(), loss_tgt.item()), flush=flush)
    end_time = time.time()
    time_str = str(datetime.timedelta(seconds=int(end_time - start_time)))
    print('Mean teacher training epoch %d done. Epoch loss: %f ' % (epoch, epoch_loss / len(tgt_loader)), flush=flush)
    print('GT box number: {}'.format(gt_box_num_cnt), flush=flush)
    if not oracle:
        print('Pseudo box number: {}'.format(pseudo_box_num_cnt), flush=flush)
    print('Time cost: {}'.format(time_str), flush=flush)
    return student, teacher


@torch.no_grad()
def evaluate(val_loader: DataLoader,
             detector: VoteNet,
             criterion: VoteNetCriterion,
             evaluator: Evaluator,
             device: torch.device,
             print_freq: int = 20,
             flush: bool = False):
    start_time = time.time()
    detector.eval()  # set model to eval mode (for bn and dp)
    epoch_loss = 0.0
    for batch_idx, (batch_data_label) in enumerate(val_loader):
        batch_data_label = batch_data_to_device(batch_data_label, device)
        # Forward pass
        point_clouds = batch_data_label['point_clouds']
        outputs = detector(point_clouds)
        # Compute loss
        loss, loss_dict = criterion(outputs, batch_data_label)
        epoch_loss += loss.item()
        # Accumulate statistics for evaluator
        model = detector if hasattr(detector, 'post_process') else detector.module
        predictions = model.post_process(point_clouds, outputs)
        evaluator.accumulate(predictions, batch_data_label)
        # Print
        if (batch_idx + 1) % print_freq == 0:
            print('Evaluation batch: %03d / %03d, loss: %f'
                  % (batch_idx + 1, len(val_loader), loss.item()), flush=flush)
    # Evaluate average precision
    evaluator.reduce_between_processes()
    metrics_dict, m_ap = evaluator.compute_metrics()
    evaluator.reset()
    for key, value in metrics_dict.items():
        print(key.ljust(15) + str(value).replace("'", ""), flush=flush)
    end_time = time.time()
    time_str = str(datetime.timedelta(seconds=int(end_time - start_time)))
    print('Evaluation done. Epoch loss: %f, mAP: %f' % (epoch_loss / len(val_loader), m_ap), flush=flush)
    print('Time cost: {}'.format(time_str), flush=flush)
    return m_ap
