import torch
from nnunetv2.training.loss.dice import SoftDiceLoss, MemoryEfficientSoftDiceLoss
from nnunetv2.training.loss.dice import  *

from nnunetv2.training.loss.robust_ce_loss import RobustCrossEntropyLoss, TopKLoss
from nnunetv2.utilities.helpers import softmax_helper_dim1
from torch import nn


class PartMemoryEfficientSoftDiceLoss(nn.Module):
    def __init__(self, apply_nonlin: Callable = None, batch_dice: bool = False, do_bg: bool = True, smooth: float = 1.,
                 ddp: bool = True):
        """
        saves 1.6 GB on Dataset017 3d_lowres
        """
        super(PartMemoryEfficientSoftDiceLoss, self).__init__()

        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
        self.ddp = ddp

    def forward(self, x, y, loss_mask=None):
        shp_x, shp_y = x.shape, y.shape

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        if not self.do_bg:
            x = x[:, 1:]

        # make everything shape (b, c)
        axes = list(range(2, len(shp_x)))

        with torch.no_grad():
            if len(shp_x) != len(shp_y):
                y = y.view((shp_y[0], 1, *shp_y[1:]))

            if all([i == j for i, j in zip(shp_x, shp_y)]):
                # if this is the case then gt is probably already a one hot encoding
                y_onehot = y
            else:
                gt = y.long()
                y_onehot = torch.zeros(shp_x, device=x.device, dtype=torch.bool)
                y_onehot.scatter_(1, gt, 1)

            if not self.do_bg:
                y_onehot = y_onehot[:, 1:]

            sum_gt = y_onehot.sum(axes) if loss_mask is None else (y_onehot ).sum(axes) * loss_mask

        intersect = (x * y_onehot).sum(axes) if loss_mask is None else (x * y_onehot ).sum(axes) * loss_mask

        sum_pred = x.sum(axes) if loss_mask is None else (x ).sum(axes) * loss_mask

        if self.ddp and self.batch_dice:
            intersect = AllGatherGrad.apply(intersect).sum(0)
            sum_pred = AllGatherGrad.apply(sum_pred).sum(0)
            sum_gt = AllGatherGrad.apply(sum_gt).sum(0)

        if self.batch_dice:
            intersect = intersect.sum(0)
            sum_pred = sum_pred.sum(0)
            sum_gt = sum_gt.sum(0)

        dc = (2 * intersect + self.smooth) / (torch.clip(sum_gt + sum_pred + self.smooth, 1e-8))

        dc = dc.mean()
        return -dc
    


# class DC_and_BCE_loss(nn.Module):
#     def __init__(self, bce_kwargs, soft_dice_kwargs, weight_ce=1, weight_dice=1, use_ignore_label: bool = False,
#                  dice_class=MemoryEfficientSoftDiceLoss):
#         """
#         DO NOT APPLY NONLINEARITY IN YOUR NETWORK!

#         target mut be one hot encoded
#         IMPORTANT: We assume use_ignore_label is located in target[:, -1]!!!

#         :param soft_dice_kwargs:
#         :param bce_kwargs:
#         :param aggregate:
#         """
#         super(DC_and_BCE_loss, self).__init__()
#         if use_ignore_label:
#             bce_kwargs['reduction'] = 'none'

#         self.weight_dice = weight_dice
#         self.weight_ce = weight_ce
#         self.use_ignore_label = use_ignore_label

#         self.ce = nn.BCEWithLogitsLoss(**bce_kwargs)
#         self.dc = dice_class(apply_nonlin=torch.sigmoid, **soft_dice_kwargs)

#     def forward(self, net_output: torch.Tensor, target: torch.Tensor):
#         if self.use_ignore_label:
#             # target is one hot encoded here. invert it so that it is True wherever we can compute the loss
#             mask = (1 - target[:, -1:]).bool()
#             # remove ignore channel now that we have the mask
#             target_regions = torch.clone(target[:, :-1])
#         else:
#             target_regions = target
#             mask = None

#         dc_loss = self.dc(net_output, target_regions, loss_mask=mask)
#         if mask is not None:
#             ce_loss = (self.ce(net_output, target_regions) * mask).sum() / torch.clip(mask.sum(), min=1e-8)
#         else:
#             ce_loss = self.ce(net_output, target_regions)
#         result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
#         return result

class Part_DC_and_BCE_loss(nn.Module):
    def __init__(self, bce_kwargs, soft_dice_kwargs, weight_ce=1, weight_dice=1, use_ignore_label: bool = False,
                 dice_class=PartMemoryEfficientSoftDiceLoss, do_bg=True, part_label=[14]):
        """
        
        DO NOT APPLY NONLINEARITY IN YOUR NETWORK!

        target mut be one hot encoded
        IMPORTANT: We assume use_ignore_label is located in target[:, -1]!!!

        :param soft_dice_kwargs:
        :param bce_kwargs:
        :param aggregate:
        """
        super(Part_DC_and_BCE_loss, self).__init__()
        # if use_ignore_label:
        #     bce_kwargs['reduction'] = 'none'

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.use_ignore_label = use_ignore_label

        self.ce = nn.BCEWithLogitsLoss(reduction='none',**bce_kwargs) # This loss combines a Sigmoid layer and the BCELoss in one single class.
        self.dc = dice_class(apply_nonlin=torch.sigmoid, **soft_dice_kwargs)
        self.part_label = part_label
        # self.channel_mask = torch.tensor([1]*14)
        self.do_bg = do_bg
        

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        # if self.use_ignore_label:
        #     # target is one hot encoded here. invert it so that it is True wherever we can compute the loss
        #     mask = (1 - target[:, -1:]).bool()
        #     # remove ignore channel now that we have the mask
        #     target_regions = torch.clone(target[:, :-1])
        # else:
        #     target_regions = target
        #     mask = None

        x, y = net_output, target
        shp_x, shp_y = x.shape, y.shape

        # if self.apply_nonlin is not None:
        #     x = self.apply_nonlin(x)

        if not self.do_bg:
            x = x[:, 1:]

        # make everything shape (b, c)
        axes = list(range(2, len(shp_x)))

        with torch.no_grad():
            if len(shp_x) != len(shp_y):
                y = y.view((shp_y[0], 1, *shp_y[1:]))

            if all([i == j for i, j in zip(shp_x, shp_y)]):
                # if this is the case then gt is probably already a one hot encoding
                y_onehot = y
            else:
                gt = y.long()
                y_onehot = torch.zeros(shp_x, device=x.device, dtype=torch.bool)
                y_onehot.scatter_(1, gt, 1)

            if not self.do_bg:
                y_onehot = y_onehot[:, 1:]
        

        n = shp_x[0]
        c = shp_x[1]
        # length = len(y.size)

        batch_channel_mask = torch.ones([n,c]).to(x.device) 
        for i in range(n):
            for j in self.part_label:
                if not torch.any(y_onehot[i, j]):
                    batch_channel_mask[i, j] = 0

        dc_loss = self.dc(x, y_onehot, loss_mask=batch_channel_mask)

        
        ce_loss = (self.ce(x, y_onehot.float()).mean(dim=axes) * batch_channel_mask).sum() / torch.clip(batch_channel_mask.sum(), min=1e-8)

        # if mask is not None:
        #     ce_loss = (self.ce(net_output, target_regions) * mask).sum() / torch.clip(mask.sum(), min=1e-8)
        # else:
        #     ce_loss = self.ce(net_output, target_regions)
        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        return result
    

