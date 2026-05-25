import torch
from copy import deepcopy
import numpy as np
import pdb
# from collections import Counter
import math

def cal_ent(zs, ens, alpha):
    ens_pred = alpha * zs.softmax(dim=0) + (1.0 - alpha) * ens.softmax(dim=1).mean(dim=0)
    pred_ = ens_pred.argmax()

    output_ = torch.clamp(ens_pred, min=1e-7)
    ent_ = -(output_ * output_.log()).sum() / np.log(len(output_))

    return pred_.item(), ent_.item()

def select_confident_samples(logits, top, order=False):
    batch_entropy = -(logits.softmax(1) * logits.log_softmax(1)).sum(1)
    selected_num = max(1, int(batch_entropy.size()[0] * top))
    idx = torch.argsort(batch_entropy, descending=order)[:selected_num]
    return logits[idx], idx

def my_loss(output, outputs, alpha):
    prob_q = (1 - alpha) * outputs.softmax(dim=1).mean(dim=0) + alpha * output.softmax(dim=0)
    prob_p = torch.cat((output.unsqueeze(0), outputs), dim=0).softmax(dim=1).mean(dim=0)
    loss = -(prob_p * prob_q.detach().log()).sum()
    return loss, prob_q.argmax().item()

class UNU():
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def pre_adaptation(self, args):
        if args.load is not None:
            self.model.prompt_learner.ctx_init_state = args.pretrained_ctx
        with torch.no_grad():
            self.model.reset()
        self.optimizer = torch.optim.AdamW(self.model.prompt_learner.parameters(), args.lr)
        self.scaler = torch.amp.GradScaler('cuda', init_scale=1000)

    def obtain_logits(self, cached_feats):
        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                text_features = self.model.get_text_features()
                logit_scale = self.model.logit_scale.exp()

                feats = cached_feats.to(self.device, non_blocking=True).squeeze(0)
                outputs = logit_scale * feats @ text_features.t()
        return outputs

    def adaptation_process(self, image, images, args):
        assert args.tta_steps > 0
        tta_steps = args.tta_steps

        self.pre_adaptation(args)
        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                outputs = self.model(images)

        pred_zs = outputs[0].argmax() # zero-shot prediction
        # pred_dict = [pred_zs.item()]
        pred_dict = []

        pred_augs = outputs[1:].argmax(dim=1)
        return_dict = {
            'aug': pred_augs,
            'opt': False,
        }

        # ---------------------------O---------------------------O---------------------------
        batch_entropy = -(outputs[1:].softmax(1) * outputs[1:].log_softmax(1)).sum(1)
        zs_entropy = -(outputs[0].softmax(0) * outputs[0].log_softmax(0)).sum()

        selected_num = max(1, int(batch_entropy.size()[0] * args.selection_p))
        idx = torch.argsort(batch_entropy, descending=False)[:selected_num]

        # ---------------------------O---------------------------O---------------------------

        if args.skip and not (pred_augs[idx] != pred_zs).sum().item() == 0: 
        # if pred_zs == pred_mean and pred_zs == pred_feat:         
            return_dict['opt'] = True

            if args.alpha is not None:
                # 使用传入的alpha值
                alpha = args.alpha
            else:
                # 使用原本的计算方式
                rank_ = (zs_entropy < batch_entropy).sum().item() / len(batch_entropy)
                alpha = (rank_ - 0.5) * args.scale + 0.5
                # scale = 1 ==> rank_ ; scale = 0 ==> 0.5 
            return_dict['alpha'] = alpha

            with torch.amp.autocast('cuda'):
                outputs = self.model(images)
                loss, pred_ = my_loss(outputs[0], outputs[1:][idx], alpha)

                pred_dict.append(pred_)

                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

            with torch.no_grad():
                with torch.amp.autocast('cuda'):
                    tta_outputs = self.model(images)

            # pred_mean, _ = cal_ent(tta_outputs[0], tta_outputs[1:][idx], alpha)
            # pred_dict.append(pred_mean)
            pred_dict.append(tta_outputs[0].argmax())

        else:
            for i in range(2):
                pred_dict.append(pred_zs.item())

        return_dict['pred'] = torch.tensor(pred_dict)

        return return_dict

    def feature_adaptation_process(self, cached_feats, args):
        assert args.tta_steps > 0
        tta_steps = args.tta_steps

        self.pre_adaptation(args)

        outputs = self.obtain_logits(cached_feats)

        pred_zs = outputs[0].argmax() # zero-shot prediction
        pred_dict = [pred_zs.item()]
        # pred_dict = []

        pred_augs = outputs[1:].argmax(dim=1)
        return_dict = {
            'aug': pred_augs,
            'opt': False,
        }       

        # ---------------------------O---------------------------O---------------------------

        batch_entropy = -(outputs[1:].softmax(1) * outputs[1:].log_softmax(1)).sum(1)
        zs_entropy = -(outputs[0].softmax(0) * outputs[0].log_softmax(0)).sum()

        selected_num = max(1, int(batch_entropy.size()[0] * args.selection_p))
        idx = torch.argsort(batch_entropy, descending=False)[:selected_num]

        # ---------------------------O---------------------------O---------------------------

        if not (pred_augs[idx] != pred_zs).sum().item() == 0:     
        # if 1:   
            return_dict['opt'] = True

            if args.alpha is not None:
                # 使用传入的alpha值
                alpha = args.alpha
            else:
                # 使用原本的计算方式
                rank_ = (zs_entropy < batch_entropy).sum().item() / len(batch_entropy)
                alpha = (rank_ - 0.5) * args.scale + 0.5
                # scale = 1 ==> rank_ ; scale = 0 ==> 0.5 
            return_dict['alpha'] = alpha

            with torch.amp.autocast('cuda'):
                text_features = self.model.get_text_features()
                logit_scale = self.model.logit_scale.exp()

                feats = cached_feats.to(self.device, non_blocking=True).squeeze(0)
                outputs = logit_scale * feats @ text_features.t()

                loss, pred_ = my_loss(outputs[0], outputs[1:][idx], alpha)

            # pred_dict.append(pred_)

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()


            tta_outputs = self.obtain_logits(cached_feats)
            tmp=1/7
            pred_mean, _ = cal_ent(tta_outputs[0], tta_outputs[1:][idx],tmp)
            pred_dict.append(pred_mean)

        else:
            K = 2 - len(pred_dict)
            for _ in range(K):
                pred_dict.append(pred_zs.item())

        return_dict['pred'] = torch.tensor(pred_dict)

        return return_dict