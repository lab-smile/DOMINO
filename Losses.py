import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from monai.losses import DiceCELoss
from torch.nn._reduction import legacy_get_string

N_classes = 11
matrix_vals = np.ones(N_classes) * 3.0
matrix_penalty = torch.from_numpy(matrix_vals)  # .to_numpy())
matrix_penalty = matrix_penalty.float().cuda()


class _Loss(nn.Module):
    reduction: str

    def __init__(self, size_average=None, reduce=None, reduction: str = "mean") -> None:
        super().__init__()
        if size_average is not None or reduce is not None:
            self.reduction: str = legacy_get_string(size_average, reduce)
        else:
            self.reduction = reduction


class DOMINO(_Loss):

    def __init__(self):
        super().__init__()

    def ce(
            self,
            input: torch.Tensor, 
            target: torch.Tensor
        ):
        ce_compute = DiceCELoss(to_onehot_y=True, softmax=True)
        return ce_compute(input, target)

    def penalty(
        self, 
        input: torch.Tensor, 
        target: torch.Tensor, 
        matrix_penalty: torch.Tensor
    ):

        n, c, h, w, z = input.size()

        target_new = torch.flatten(target)  # b * 1 * 64 * 64 * 64 -> b*1*64*64*64
        target_new = F.one_hot(target_new.to(torch.int64), c).cuda()  # NHWZ * C
        target_new = target_new.unsqueeze(1)  # NHWZ * 1 * C

        outputs = torch.swapaxes(input, 0, 1)  # C * N * HWZ
        outputs = torch.reshape(outputs, (c, n * h * w * z)).cuda()  #
        outputs = torch.swapaxes(outputs, 0, 1)  # nhwz , c
        outputs = outputs.unsqueeze(2)  # nhwz , c, 1

        m = nn.Softmax(dim=1)
        outputs_soft = m(outputs).float()

        matrix_penalty_rep = matrix_penalty.unsqueeze(0).repeat(n * h * w * z, 1, 1)
        penalty = torch.bmm(
            target_new.float(), matrix_penalty_rep
        ).cuda()  # (1 x N) * (N x N) = 1 x N
        penalty_term = torch.bmm(penalty.float(), outputs_soft)

        beta = 3.0
        penalty_sum = beta * (torch.mean(penalty_term).cuda())

        return penalty_sum

    def forward(
        self, 
        input: torch.Tensor, 
        target: torch.Tensor, 
        matrix_penalty: torch.Tensor
    ):
        return self.ce(input, target) + self.penalty(input, target, matrix_penalty)
