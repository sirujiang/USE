import torch
import numpy as np


def cal_ent(zs, ens, alpha):
    ens_pred = alpha * zs.softmax(dim=0) + (1.0 - alpha) * ens.softmax(dim=1).mean(dim=0)
    pred_ = ens_pred.argmax()

    output_ = torch.clamp(ens_pred, min=1e-7)
    ent_ = -(output_ * output_.log()).sum() / np.log(len(output_))

    return pred_.item(), ent_.item()


def select_and_cal_ent(outputs, selection_p, alpha=None, scale=0.4):
   
    batch_entropy = -(outputs[1:].softmax(1) * outputs[1:].log_softmax(1)).sum(1)
    zs_entropy = -(outputs[0].softmax(0) * outputs[0].log_softmax(0)).sum()
    
    # 选择最低熵的样本
    selected_num = max(1, int(batch_entropy.size(0) * selection_p))
    idx = torch.argsort(batch_entropy, descending=False)[:selected_num]
    
    # 计算alpha
    if alpha is not None:
        final_alpha = alpha
    else:
        # print(alpha)
        rank_ = (zs_entropy < batch_entropy).sum().item() / len(batch_entropy)
        final_alpha = (rank_ - 0.5) * scale + 0.5
    
    # 计算cal_ent
    pred_mean, ent = cal_ent(outputs[0], outputs[1:][idx], final_alpha)
    
    return pred_mean, ent, idx, final_alpha


def rce_loss(output, outputs, alpha):
    prob_q = (1 - alpha) * outputs.softmax(dim=1).mean(dim=0) + alpha * output.softmax(dim=0)
    prob_p = torch.cat((output.unsqueeze(0), outputs), dim=0).softmax(dim=1).mean(dim=0)
    loss = -(prob_p * prob_q.detach().log()).sum()
    return loss, prob_q.argmax().item()
