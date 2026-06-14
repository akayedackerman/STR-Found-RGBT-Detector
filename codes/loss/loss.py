# Copyright (c) OpenMMLab. All rights reserved.
import math
import warnings

import mmcv
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import *
from mmdet.registry import MODELS


def iou_loss(pred, target, linear=False, mode='log', eps=1e-6,
             reduction=None, avg_factor=None):
	"""IoU loss.

    Computing the IoU loss between a set of predicted bboxes and target bboxes.
    The loss is calculated as negative log of IoU.

    Args:
        pred (torch.Tensor): Predicted bboxes of format (x1, y1, x2, y2),
            shape (n, 4).
        target (torch.Tensor): Corresponding gt bboxes, shape (n, 4).
        linear (bool, optional): If True, use linear scale of loss instead of
            log scale. Default: False.
        mode (str): Loss scaling mode, including "linear", "square", and "log".
            Default: 'log'
        eps (float): Eps to avoid log(0).

    Return:
        torch.Tensor: Loss tensor.
    """
	assert mode in ['linear', 'square', 'log']
	if linear:
		mode = 'linear'
		warnings.warn('DeprecationWarning: Setting "linear=True" in '
		              'iou_loss is deprecated, please use "mode=`linear`" '
		              'instead.')
	ious = bbox_overlaps(pred, target, is_aligned=True).clamp(min=eps)
	if mode == 'linear':
		loss = 1 - ious
	elif mode == 'square':
		loss = 1 - ious ** 2
	elif mode == 'log':
		loss = -ious.log()
	else:
		raise NotImplementedError
	return loss


def bounded_iou_loss(pred, target, beta=0.2, eps=1e-3,
                     reduction=None, avg_factor=None):
	"""BIoULoss.

    This is an implementation of paper
    `Improving Object Localization with Fitness NMS and Bounded IoU Loss.
    <https://arxiv.org/abs/1711.00164>`_.

    Args:
        pred (torch.Tensor): Predicted bboxes.
        target (torch.Tensor): Target bboxes.
        beta (float): beta parameter in smoothl1.
        eps (float): eps to avoid NaN.
    """
	pred_ctrx = (pred[:, 0] + pred[:, 2]) * 0.5
	pred_ctry = (pred[:, 1] + pred[:, 3]) * 0.5
	pred_w = pred[:, 2] - pred[:, 0]
	pred_h = pred[:, 3] - pred[:, 1]
	with torch.no_grad():
		target_ctrx = (target[:, 0] + target[:, 2]) * 0.5
		target_ctry = (target[:, 1] + target[:, 3]) * 0.5
		target_w = target[:, 2] - target[:, 0]
		target_h = target[:, 3] - target[:, 1]

	dx = target_ctrx - pred_ctrx
	dy = target_ctry - pred_ctry

	loss_dx = 1 - torch.max(
		(target_w - 2 * dx.abs()) /
		(target_w + 2 * dx.abs() + eps), torch.zeros_like(dx))
	loss_dy = 1 - torch.max(
		(target_h - 2 * dy.abs()) /
		(target_h + 2 * dy.abs() + eps), torch.zeros_like(dy))
	loss_dw = 1 - torch.min(target_w / (pred_w + eps), pred_w /
	                        (target_w + eps))
	loss_dh = 1 - torch.min(target_h / (pred_h + eps), pred_h /
	                        (target_h + eps))
	# view(..., -1) does not work for empty tensor
	loss_comb = torch.stack([loss_dx, loss_dy, loss_dw, loss_dh],
	                        dim=-1).flatten(1)

	loss = torch.where(loss_comb < beta, 0.5 * loss_comb * loss_comb / beta,
	                   loss_comb - 0.5 * beta)
	return loss


def giou_loss(pred, target, eps=1e-7,
              reduction=None, avg_factor=None):
	r"""`Generalized Intersection over Union: A Metric and A Loss for Bounding
    Box Regression <https://arxiv.org/abs/1902.09630>`_.

    Args:
        pred (torch.Tensor): Predicted bboxes of format (x1, y1, x2, y2),
            shape (n, 4).
        target (torch.Tensor): Corresponding gt bboxes, shape (n, 4).
        eps (float): Eps to avoid log(0).

    Return:
        Tensor: Loss tensor.
    """
	gious = bbox_overlaps(pred, target, mode='giou', is_aligned=True, eps=eps)
	loss = 1 - gious
	return loss


def diou_loss(pred, target, eps=1e-7,
              reduction=None, avg_factor=None):
	r"""`Implementation of Distance-IoU Loss: Faster and Better
    Learning for Bounding Box Regression, https://arxiv.org/abs/1911.08287`_.

    Code is modified from https://github.com/Zzh-tju/DIoU.

    Args:
        pred (Tensor): Predicted bboxes of format (x1, y1, x2, y2),
            shape (n, 4).
        target (Tensor): Corresponding gt bboxes, shape (n, 4).
        eps (float): Eps to avoid log(0).
    Return:
        Tensor: Loss tensor.
    """
	# overlap
	lt = torch.max(pred[:, :2], target[:, :2])
	rb = torch.min(pred[:, 2:], target[:, 2:])
	wh = (rb - lt).clamp(min=0)
	overlap = wh[:, 0] * wh[:, 1]

	# union
	ap = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
	ag = (target[:, 2] - target[:, 0]) * (target[:, 3] - target[:, 1])
	union = ap + ag - overlap + eps

	# IoU
	ious = overlap / union

	# enclose area
	enclose_x1y1 = torch.min(pred[:, :2], target[:, :2])
	enclose_x2y2 = torch.max(pred[:, 2:], target[:, 2:])
	enclose_wh = (enclose_x2y2 - enclose_x1y1).clamp(min=0)

	cw = enclose_wh[:, 0]
	ch = enclose_wh[:, 1]

	c2 = cw ** 2 + ch ** 2 + eps

	b1_x1, b1_y1 = pred[:, 0], pred[:, 1]
	b1_x2, b1_y2 = pred[:, 2], pred[:, 3]
	b2_x1, b2_y1 = target[:, 0], target[:, 1]
	b2_x2, b2_y2 = target[:, 2], target[:, 3]

	left = ((b2_x1 + b2_x2) - (b1_x1 + b1_x2)) ** 2 / 4
	right = ((b2_y1 + b2_y2) - (b1_y1 + b1_y2)) ** 2 / 4
	rho2 = left + right

	# DIoU
	dious = ious - rho2 / c2
	loss = 1 - dious
	return loss


def safit_loss(pred, target, eps=1e-7, C=32,
               reduction=None, avg_factor=None):
	# -----------------------------------------------------
	# Prediction stabilization
	# -----------------------------------------------------

	pred = torch.clamp(
		pred,
		min=-50,
		max=700
	)

	# -----------------------------------------------------
	# IOU
	# -----------------------------------------------------

	ious = bbox_overlaps(
		pred,
		target,
		is_aligned=True
	).clamp(min=eps)

	# -----------------------------------------------------
	# NWD
	# -----------------------------------------------------

	pred_xyhw = xyxy2xyw2h2(pred)
	target_xyhw = xyxy2xyw2h2(target)

	diff = (
			(pred_xyhw - target_xyhw) *
			(pred_xyhw - target_xyhw)
	)

	sum_diff = torch.sum(diff, dim=1)

	sum_diff = torch.clamp(
		sum_diff,
		min=1e-12
	)

	sqrt_term = torch.sqrt(sum_diff)

	exp_input = -sqrt_term / C

	exp_input = torch.clamp(
		exp_input,
		min=-50,
		max=50
	)

	nwd = torch.exp(exp_input)

	# -----------------------------------------------------
	# SAFit
	# -----------------------------------------------------

	ag = (
			(target[:, 2] - target[:, 0]) *
			(target[:, 3] - target[:, 1])
	)

	area = torch.clamp(
		ag,
		min=1e-12
	)

	sqrt_area = torch.sqrt(area)

	sigmoid_term = sigmoid(
		sqrt_area - C,
		C
	)

	safit = (
			sigmoid_term * ious +
			(1 - sigmoid_term) * nwd
	)

	loss = 1 - safit

	return loss


def nwd_loss(pred, target, eps=1e-7, C=32,
             reduction=None, avg_factor=None):
	r"""`Implementation of Distance-IoU Loss: Faster and Better
    Learning for Bounding Box Regression, https://arxiv.org/abs/1911.08287`_.

    Code is modified from https://github.com/Zzh-tju/DIoU.

    Args:
        pred (Tensor): Predicted bboxes of format (x1, y1, x2, y2),
            shape (n, 4).
        target (Tensor): Corresponding gt bboxes, shape (n, 4).
        eps (float): Eps to avoid log(0).
    Return:
        Tensor: Loss tensor.
    """
	pred, target = xyxy2xyw2h2(pred), xyxy2xyw2h2(target)
	nwd = torch.exp(-torch.norm((pred - target), p=2, dim=1) / C)
	loss = 1 - nwd.clamp(min=-1.0, max=1.0)
	return loss


def ciou_loss(pred, target, eps=1e-7,
              reduction=None, avg_factor=None):
	r"""`Implementation of paper `Enhancing Geometric Factors into
    Model Learning and Inference for Object Detection and Instance
    Segmentation <https://arxiv.org/abs/2005.03572>`_.

    Code is modified from https://github.com/Zzh-tju/CIoU.

    Args:
        pred (Tensor): Predicted bboxes of format (x1, y1, x2, y2),
            shape (n, 4).
        target (Tensor): Corresponding gt bboxes, shape (n, 4).
        eps (float): Eps to avoid log(0).
    Return:
        Tensor: Loss tensor.
    """
	# overlap
	lt = torch.max(pred[:, :2], target[:, :2])
	rb = torch.min(pred[:, 2:], target[:, 2:])
	wh = (rb - lt).clamp(min=0)
	overlap = wh[:, 0] * wh[:, 1]

	# union
	ap = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
	ag = (target[:, 2] - target[:, 0]) * (target[:, 3] - target[:, 1])
	union = ap + ag - overlap + eps

	# IoU
	ious = overlap / union

	# enclose area
	enclose_x1y1 = torch.min(pred[:, :2], target[:, :2])
	enclose_x2y2 = torch.max(pred[:, 2:], target[:, 2:])
	enclose_wh = (enclose_x2y2 - enclose_x1y1).clamp(min=0)

	cw = enclose_wh[:, 0]
	ch = enclose_wh[:, 1]

	c2 = cw ** 2 + ch ** 2 + eps

	b1_x1, b1_y1 = pred[:, 0], pred[:, 1]
	b1_x2, b1_y2 = pred[:, 2], pred[:, 3]
	b2_x1, b2_y1 = target[:, 0], target[:, 1]
	b2_x2, b2_y2 = target[:, 2], target[:, 3]

	w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
	w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

	left = ((b2_x1 + b2_x2) - (b1_x1 + b1_x2)) ** 2 / 4
	right = ((b2_y1 + b2_y2) - (b1_y1 + b1_y2)) ** 2 / 4
	rho2 = left + right

	factor = 4 / math.pi ** 2
	v = factor * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)

	with torch.no_grad():
		alpha = (ious > 0.5).float() * v / (1 - ious + v)

	# CIoU
	cious = ious - (rho2 / c2 + alpha * v)
	loss = 1 - cious.clamp(min=-1.0, max=1.0)
	return loss


class IoULoss(nn.Module):
	"""IoULoss.

    Computing the IoU loss between a set of predicted bboxes and target bboxes.

    Args:
        linear (bool): If True, use linear scale of loss else determined
            by mode. Default: False.
        eps (float): Eps to avoid log(0).
        reduction (str): Options are "none", "mean" and "sum".
        loss_weight (float): Weight of loss.
        mode (str): Loss scaling mode, including "linear", "square", and "log".
            Default: 'log'
    """

	def __init__(self,
	             linear=False,
	             eps=1e-6,
	             reduction='mean',
	             loss_weight=1.0,
	             mode='log'):
		super(IoULoss, self).__init__()
		assert mode in ['linear', 'square', 'log']
		if linear:
			mode = 'linear'
			warnings.warn('DeprecationWarning: Setting "linear=True" in '
			              'IOULoss is deprecated, please use "mode=`linear`" '
			              'instead.')
		self.mode = mode
		self.linear = linear
		self.eps = eps
		self.reduction = reduction
		self.loss_weight = loss_weight

	def forward(self,
	            pred,
	            target,
	            weight=None,
	            avg_factor=None,
	            reduction_override=None,
	            **kwargs):
		"""Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning target of the prediction.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None. Options are "none", "mean" and "sum".
        """
		assert reduction_override in (None, 'none', 'mean', 'sum')
		reduction = (
			reduction_override if reduction_override else self.reduction)
		if (weight is not None) and (not torch.any(weight > 0)) and (
				reduction != 'none'):
			if pred.dim() == weight.dim() + 1:
				weight = weight.unsqueeze(1)
			return (pred * weight).sum()  # 0
		if weight is not None and weight.dim() > 1:
			# TODO: remove this in the future
			# reduce the weight of shape (n, 4) to (n,) to match the
			# iou_loss of shape (n,)
			assert weight.shape == pred.shape
			weight = weight.mean(-1)
		loss = self.loss_weight * iou_loss(
			pred,
			target,
			weight,
			mode=self.mode,
			eps=self.eps,
			reduction=reduction,
			avg_factor=avg_factor,
			**kwargs)
		return loss


class BoundedIoULoss(nn.Module):

	def __init__(self, beta=0.2, eps=1e-3, reduction='mean', loss_weight=1.0):
		super(BoundedIoULoss, self).__init__()
		self.beta = beta
		self.eps = eps
		self.reduction = reduction
		self.loss_weight = loss_weight

	def forward(self,
	            pred,
	            target,
	            weight=None,
	            avg_factor=None,
	            reduction_override=None,
	            **kwargs):
		if weight is not None and not torch.any(weight > 0):
			if pred.dim() == weight.dim() + 1:
				weight = weight.unsqueeze(1)
			return (pred * weight).sum()  # 0
		assert reduction_override in (None, 'none', 'mean', 'sum')
		reduction = (
			reduction_override if reduction_override else self.reduction)
		loss = self.loss_weight * bounded_iou_loss(
			pred,
			target,
			beta=self.beta,
			eps=self.eps,
			reduction=reduction,
			avg_factor=avg_factor,
			**kwargs)
		return loss


class GIoULoss(nn.Module):

	def __init__(self, eps=1e-6, reduction='mean', loss_weight=1.0):
		super(GIoULoss, self).__init__()
		self.eps = eps
		self.reduction = reduction
		self.loss_weight = loss_weight

	def forward(self,
	            pred,
	            target,
	            weight=None,
	            avg_factor=None,
	            reduction_override=None,
	            **kwargs):
		if weight is not None and not torch.any(weight > 0):
			if pred.dim() == weight.dim() + 1:
				weight = weight.unsqueeze(1)
			return (pred * weight).sum()  # 0
		assert reduction_override in (None, 'none', 'mean', 'sum')
		reduction = (
			reduction_override if reduction_override else self.reduction)
		if weight is not None and weight.dim() > 1:
			# TODO: remove this in the future
			# reduce the weight of shape (n, 4) to (n,) to match the
			# iou_loss of shape (n,)
			assert weight.shape == pred.shape
			weight = weight.mean(-1)
		loss = self.loss_weight * giou_loss(
			pred,
			target,
			eps=self.eps,
			reduction=reduction,
			avg_factor=avg_factor,
			**kwargs)
		return loss


class DIoULoss(nn.Module):

	def __init__(self, eps=1e-6, reduction='mean', loss_weight=1.0):
		super(DIoULoss, self).__init__()
		self.eps = eps
		self.reduction = reduction
		self.loss_weight = loss_weight

	def forward(self,
	            pred,
	            target,
	            weight=None,
	            avg_factor=None,
	            reduction_override=None,
	            **kwargs):
		if weight is not None and not torch.any(weight > 0):
			if pred.dim() == weight.dim() + 1:
				weight = weight.unsqueeze(1)
			return (pred * weight).sum()  # 0
		assert reduction_override in (None, 'none', 'mean', 'sum')
		reduction = (
			reduction_override if reduction_override else self.reduction)
		if weight is not None and weight.dim() > 1:
			# TODO: remove this in the future
			# reduce the weight of shape (n, 4) to (n,) to match the
			# iou_loss of shape (n,)
			assert weight.shape == pred.shape
			weight = weight.mean(-1)
		loss = self.loss_weight * diou_loss(
			pred,
			target,
			eps=self.eps,
			reduction=reduction,
			avg_factor=avg_factor,
			**kwargs)
		return loss


class CIoULoss(nn.Module):

	def __init__(self, eps=1e-6, reduction='mean', loss_weight=1.0):
		super(CIoULoss, self).__init__()
		self.eps = eps
		self.reduction = reduction
		self.loss_weight = loss_weight

	def forward(self,
	            pred,
	            target,
	            weight=None,
	            avg_factor=None,
	            reduction_override=None,
	            **kwargs):
		if weight is not None and not torch.any(weight > 0):
			if pred.dim() == weight.dim() + 1:
				weight = weight.unsqueeze(1)
			return (pred * weight).sum()  # 0
		assert reduction_override in (None, 'none', 'mean', 'sum')
		reduction = (
			reduction_override if reduction_override else self.reduction)
		if weight is not None and weight.dim() > 1:
			# TODO: remove this in the future
			# reduce the weight of shape (n, 4) to (n,) to match the
			# iou_loss of shape (n,)
			assert weight.shape == pred.shape
			weight = weight.mean(-1)
		loss = self.loss_weight * ciou_loss(
			pred,
			target,
			eps=self.eps,
			reduction=reduction,
			avg_factor=avg_factor,
			**kwargs)
		return loss


@MODELS.register_module()
class SAFitLoss(nn.Module):

	def __init__(self, eps=1e-6, reduction='mean', loss_weight=1.0):
		super(SAFitLoss, self).__init__()
		self.eps = eps
		self.reduction = reduction
		self.loss_weight = loss_weight

	def forward(self,
	            pred,
	            target,
	            weight=None,
	            avg_factor=None,
	            reduction_override=None,
	            **kwargs):
		if weight is not None and not torch.any(weight > 0):
			if pred.dim() == weight.dim() + 1:
				weight = weight.unsqueeze(1)
			return (pred * weight).sum()  # 0
		assert reduction_override in (None, 'none', 'mean', 'sum')
		reduction = (
			reduction_override if reduction_override else self.reduction)
		if weight is not None and weight.dim() > 1:
			# TODO: remove this in the future
			# reduce the weight of shape (n, 4) to (n,) to match the
			# iou_loss of shape (n,)
			assert weight.shape == pred.shape
			weight = weight.mean(-1)
		loss = self.loss_weight * safit_loss(
			pred,
			target,
			eps=self.eps,
			reduction=reduction,
			avg_factor=avg_factor,
			**kwargs)
		return loss


class NWDLoss(nn.Module):

	def __init__(self, eps=1e-6, reduction='mean', loss_weight=1.0, C=32):
		super(NWDLoss, self).__init__()
		self.eps = eps
		self.reduction = reduction
		self.loss_weight = loss_weight
		self.C = C

	def forward(self,
	            pred,
	            target,
	            weight=None,
	            avg_factor=None,
	            reduction_override=None,
	            **kwargs):
		if weight is not None and not torch.any(weight > 0):
			if pred.dim() == weight.dim() + 1:
				weight = weight.unsqueeze(1)
			return (pred * weight).sum()  # 0
		assert reduction_override in (None, 'none', 'mean', 'sum')
		reduction = (
			reduction_override if reduction_override else self.reduction)
		if weight is not None and weight.dim() > 1:
			# TODO: remove this in the future
			# reduce the weight of shape (n, 4) to (n,) to match the
			# iou_loss of shape (n,)
			assert weight.shape == pred.shape
			weight = weight.mean(-1)
		loss = self.loss_weight * nwd_loss(
			pred,
			target,
			eps=self.eps,
			C=self.C,
			reduction=reduction,
			avg_factor=avg_factor,
			**kwargs)
		return loss


# ==============================================================================
# UPGRADE 2: ANTI-SHORTCUT BATCH-SHUFFLING TEMPORAL CONTRASTIVE LOSS
# ==============================================================================
@MODELS.register_module()
class TemporalContrastiveLoss(nn.Module):
	"""Temporal Contrastive Learning (TCL) Loss via Fortified Batch-Shuffling.

	Prevents background shortcut learning by explicitly ensuring negative samples
	are pulled from distinct video clips across the active training batch.
	"""

	def __init__(self, temperature=0.07, loss_weight=0.1):
		super(TemporalContrastiveLoss, self).__init__()
		self.temperature = temperature
		self.loss_weight = loss_weight

	def forward(self, curr_vectors, past_vectors, **kwargs):
		total_tcl_loss = 0.0
		num_scales = len(curr_vectors)
		batch_size = curr_vectors[0].size(0)

		# Safeguard: if batch_size is 1, shuffling cross-batch negatives is impossible
		if batch_size < 2:
			return curr_vectors[0].sum() * 0.0

		for scale_idx in range(num_scales):
			# Normalize feature vectors onto a unit hypersphere
			z_curr = F.normalize(curr_vectors[scale_idx], dim=1)
			z_past = F.normalize(past_vectors[scale_idx], dim=1)

			# Compute raw cosine similarity score matrix: [B, B]
			similarity_matrix = torch.matmul(z_curr, z_past.T) / self.temperature

			# Create an identity diagonal mask tracking matching video slots
			diagonal_mask = torch.eye(batch_size, device=similarity_matrix.device, dtype=torch.bool)

			# Extract true cross-frame moving targets
			positives = similarity_matrix[diagonal_mask].view(batch_size, 1)

			# Isolate negative samples belonging strictly to alternative video clips
			negatives = similarity_matrix[~diagonal_mask].view(batch_size, -1)

			# Reassemble logits where index 0 is always the true positive tracking target
			logits = torch.cat([positives, negatives], dim=1)
			labels = torch.zeros(batch_size, dtype=torch.long, device=similarity_matrix.device)

			# Apply Cross-Entropy to optimize InfoNCE tracking without scene shortcuts
			scale_loss = F.cross_entropy(logits, labels)
			total_tcl_loss += scale_loss

		return (total_tcl_loss / num_scales) * self.loss_weight


if __name__ == "__main__":
	pred, target = torch.rand(1, 4) * torch.tensor([512, 512, 200, 200]), torch.rand(1, 4) * torch.tensor(
		[512, 512, 200, 200])
	print(pred)
	print(target)
	pred, target = xywh2xyxy(pred), xywh2xyxy(target)
	print('BoundedIoULoss:\t%.2f' % (weighted_loss(BoundedIoULoss(), pred, target)))
	print('GIoULoss:\t%.2f' % (weighted_loss(GIoULoss(), pred, target)))
	print('DIoULoss:\t%.2f' % (weighted_loss(DIoULoss(), pred, target)))
	print('CIoULoss:\t%.2f' % (weighted_loss(CIoULoss(), pred, target)))
	print('IoULoss:\t%.2f' % (weighted_loss(IoULoss(mode='linear'), pred, target)))
	print('NWDLoss:\t%.2f' % (weighted_loss(NWDLoss(), pred, target)))
	print('SAfitLoss:\t%.2f' % (weighted_loss(SAFitLoss(), pred, target)))