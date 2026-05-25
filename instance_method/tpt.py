import torch
import numpy as np

def avg_entropy(outputs):
    logits = outputs - outputs.logsumexp(dim=-1, keepdim=True)
    avg_logits = logits.logsumexp(dim=0) - np.log(logits.shape[0])
    min_real = torch.finfo(avg_logits.dtype).min
    avg_logits = torch.clamp(avg_logits, min=min_real)
    return -(avg_logits * torch.exp(avg_logits)).sum(dim=-1)

def select_confident_samples(logits, top):
    batch_entropy = -(logits.softmax(1) * logits.log_softmax(1)).sum(1)
    idx = torch.argsort(batch_entropy, descending=False)[:int(batch_entropy.size()[0] * top)]
    return logits[idx], idx

class TPT():
    def __init__(self, model, device, use_sher=False):
        self.model = model
        self.device = device
        self.use_sher = use_sher

    def pre_adaptation(self, args):
        if args.load is not None:
            self.model.prompt_learner.ctx_init_state = args.pretrained_ctx
        with torch.no_grad():
            self.model.reset()
        self.optimizer = torch.optim.AdamW(self.model.prompt_learner.parameters(), args.lr)
        self.scaler = torch.amp.GradScaler('cuda', init_scale=1000)

    def adaptation_process(self, image, images, args):
        assert args.tta_steps > 0

        self.pre_adaptation(args)
        selected_idx = None
        pred_dict = []
        for j in range(args.tta_steps):
            with torch.amp.autocast('cuda'):
                output = self.model(images) 

                if selected_idx is None:
                    _, selected_idx = select_confident_samples(output, args.selection_p)
                
                zs_output = output[0].detach()
                loss = avg_entropy(output[selected_idx])

                self.optimizer.zero_grad()

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                tta_output = self.model(image)

        return tta_output.argmax().item()

