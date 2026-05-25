import torch
import numpy as np


class CLIP():
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

    def adaptation_process(self, image, images, args):
        assert args.tta_steps > 0

        self.pre_adaptation(args)
        selected_idx = None
        pred_dict = []
        
        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                tta_output = self.model(images)
                # print(tta_output.shape)
        
        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                tta_output = self.model(images)
                # print(tta_output.shape)

        return tta_output[0].argmax().item()

