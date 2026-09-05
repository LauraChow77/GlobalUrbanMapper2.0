import torch
import torch.nn as nn


def build_criterions(*args, ignore_index=255):
    criterions = []
    device = torch.device("cuda:0")
    for arg in args:
        if arg == "ce":
            criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)
        elif arg == "kl":
            criterion = nn.KLDivLoss(reduction="none")
        else:
            raise ValueError(f"Loss '{arg}' is not implemented.")
        criterions.append(criterion.cuda(device))
    return criterions


