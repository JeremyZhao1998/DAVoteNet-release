# Created by Zijing Zhao (Peking University)
# Date: Oct 2023

import numpy as np

import torch
from pytorch3d.ops import box3d_overlap

import utils.bbox_utils as bbox_utils
import utils.distributed_utils as dist_utils


def voc_ap(recall, precision):
    """
    Compute VOC AP given precision and recall.
    """
    m_recall = np.concatenate(([0.], recall, [1.]))
    m_precision = np.concatenate(([0.], precision, [0.]))
    for i in range(m_precision.size - 1, 0, -1):
        m_precision[i - 1] = np.maximum(m_precision[i - 1], m_precision[i])
    i = np.where(m_recall[1:] != m_recall[:-1])[0]
    ap = np.sum((m_recall[i + 1] - m_recall[i]) * m_precision[i + 1])
    return ap


class Evaluator:
    """
    Calculating Average Precision and Recall for detection task.
    """
    def __init__(self,
                 num_classes,
                 categories=None,
                 ignored_categories=None,
                 iou_threshold=0.25,
                 per_class_proposal=True):
        self.num_classes = num_classes
        self.iou_threshold = iou_threshold
        self.per_class_proposal = per_class_proposal
        if categories is not None:
            assert len(categories) == num_classes
            self.label2name = {value: key for key, value in categories.items()}
            self.ignored_category_ids = [] if ignored_categories is None \
                else [categories[cat] for cat in ignored_categories if cat in categories]
        else:
            self.label2name = {cls: 'cls_%d' % cls for cls in range(self.num_classes)}
            self.ignored_category_ids = []
        self.tps_dict, self.scores_dict = None, None
        self.gt_box_cnt, self.box_match_cnt, self.cls_match_cnt = None, None, None
        self.reset()

    def reset(self):
        self.tps_dict = {cls: [] for cls in range(self.num_classes)}
        self.scores_dict = {cls: [] for cls in range(self.num_classes)}
        self.gt_box_cnt = {cls: 0 for cls in range(self.num_classes)}
        self.box_match_cnt = {cls: 0 for cls in range(self.num_classes)}
        self.cls_match_cnt = {cls: 0 for cls in range(self.num_classes)}

    def _update_match_dict(self, ious, cls_gt, sem_probs_i):
        ious_match = torch.gt(ious, self.iou_threshold)
        ious_match_any = ious_match.any(dim=0)
        for idx_c, (c, ious_m) in enumerate(zip(cls_gt, ious_match_any)):
            if ious_m:
                self.box_match_cnt[c] += 1
                box_match_id = ious_match[:, idx_c]
                match_probs = sem_probs_i[box_match_id]
                match_cls_pred = torch.argmax(match_probs, dim=-1)
                if torch.eq(match_cls_pred, c).any():
                    self.cls_match_cnt[c] += 1

    def accumulate(self, predictions, gt):
        batch_ids, box_ids = torch.where(predictions['label_masks'])
        batch_ids_gt, box_ids_gt = torch.where(gt['label_masks'])
        corners = predictions['corners']
        corners_gt = bbox_utils.boxes_to_corners_3d(gt['centers'], gt['heading_angles'], gt['box_sizes'])
        categories, categories_gt = predictions['categories'], gt['categories']
        obj_probs, sem_probs = predictions['obj_probs'], predictions['sem_cls_probs']
        for i, (boxes, boxes_gt) in enumerate(zip(corners, corners_gt)):
            valid_box_id, valid_box_id_gt = box_ids[batch_ids == i], box_ids_gt[batch_ids_gt == i]
            cls_gt = categories_gt[i, valid_box_id_gt].detach().cpu().numpy()
            for c in cls_gt:
                self.gt_box_cnt[c] += 1
            if len(valid_box_id) == 0 or len(valid_box_id_gt) == 0:
                continue
            obj_probs_i, sem_probs_i = obj_probs[i, valid_box_id], sem_probs[i, valid_box_id]
            if self.per_class_proposal:
                valid_boxes, valid_boxes_gt = boxes[valid_box_id, :, :], boxes_gt[valid_box_id_gt, :, :]
                _, ious = box3d_overlap(valid_boxes, valid_boxes_gt, eps=1e-6)
                self._update_match_dict(ious, cls_gt, sem_probs_i)
                scores = obj_probs_i.unsqueeze(-1) * sem_probs_i
                sorted_ids = torch.argsort(scores.view(-1), descending=True)
                sorted_box_ids, sorted_cls = sorted_ids // self.num_classes, sorted_ids % self.num_classes
                max_ious, gt_ids = torch.max(ious, dim=-1)
                max_ious, gt_ids = max_ious.detach().cpu().numpy(), gt_ids.detach().cpu().numpy()
                sorted_box_ids, sorted_cls = sorted_box_ids.detach().cpu().numpy(), sorted_cls.detach().cpu().numpy()
                scores = scores.detach().cpu().numpy()
                for box_id, cls in zip(sorted_box_ids, sorted_cls):
                    tp = max_ious[box_id] > self.iou_threshold and cls_gt[gt_ids[box_id]] == cls
                    self.tps_dict[cls].append(tp)
                    self.scores_dict[cls].append(scores[box_id, cls])
                    if tp:
                        cls_gt[gt_ids[box_id]] = -1
            else:
                sorted_ids = torch.argsort(obj_probs_i, descending=True)
                valid_box_id = valid_box_id[sorted_ids]
                valid_boxes, valid_boxes_gt = boxes[valid_box_id, :, :], boxes_gt[valid_box_id_gt, :, :]
                cls = categories[i, valid_box_id].detach().cpu().numpy()
                scores = obj_probs_i.detach().cpu().numpy()
                _, ious = box3d_overlap(valid_boxes, valid_boxes_gt)
                self._update_match_dict(ious, cls_gt, sem_probs_i)
                max_ious, gt_ids = torch.max(ious, dim=-1)
                max_ious, gt_ids = max_ious.detach().cpu().numpy(), gt_ids.detach().cpu().numpy()
                for box_id in range(len(valid_boxes)):
                    box_cls = cls[box_id]
                    tp = max_ious[box_id] > self.iou_threshold and cls_gt[gt_ids[box_id]] == box_cls
                    self.tps_dict[box_cls].append(tp)
                    self.scores_dict[box_cls].append(scores[box_id])
                    if tp:
                        cls_gt[gt_ids[box_id]] = -1

    def reduce_between_processes(self):
        if dist_utils.is_dist_avail_and_initialized():
            self.tps_dict = dist_utils.reduce_dict(self.tps_dict)
            self.scores_dict = dist_utils.reduce_dict(self.scores_dict)
            self.gt_box_cnt = dist_utils.reduce_dict(self.gt_box_cnt)
            self.box_match_cnt = dist_utils.reduce_dict(self.box_match_cnt)
            self.cls_match_cnt = dist_utils.reduce_dict(self.cls_match_cnt)

    def compute_metrics(self):
        metric_dict = {}
        ap_list = []
        for cls in range(self.num_classes):
            if cls in self.ignored_category_ids:
                continue
            if self.gt_box_cnt[cls] == 0:
                metric_dict['[%s]' % self.label2name[cls]] = {
                    'AP%.2f' % self.iou_threshold: 'N/A',
                    'GT box number': '0', 'Box match rate': 'N/A', 'Class match rate': 'N/A'
                }
            else:
                tps = np.array(self.tps_dict[cls])
                fps = np.bitwise_not(tps) if len(tps) > 0 else np.array([])
                scores = np.array(self.scores_dict[cls])
                sorted_ids = np.argsort(-scores)
                tps, fps, scores = tps[sorted_ids], fps[sorted_ids], scores[sorted_ids]
                tps, fps = np.cumsum(tps.astype(np.int64)), np.cumsum(fps.astype(np.int64))
                recall = tps / float(self.gt_box_cnt[cls])
                precision = tps / np.maximum(tps + fps, np.finfo(np.float64).eps)
                ap = voc_ap(recall, precision)
                metric_dict['[%s]' % self.label2name[cls]] = {
                    'AP%.2f' % self.iou_threshold: '%6.2f' % (ap * 100),
                    'GT box num': '%5d' % self.gt_box_cnt[cls],
                    'Box match': '%6.2f%%' % (self.box_match_cnt[cls] / float(self.gt_box_cnt[cls]) * 100)
                }
                if self.box_match_cnt[cls] > 0:
                    metric_dict['[%s]' % self.label2name[cls]]['Class match'] = (
                            '%6.2f%%' % (self.cls_match_cnt[cls] / float(self.box_match_cnt[cls]) * 100))
                ap_list.append(ap)
        return metric_dict, np.mean(ap_list)
