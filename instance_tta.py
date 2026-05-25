import argparse
import pdb
import time

from PIL import Image
import numpy as np

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

from collections import Counter

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
import torchvision.models as models

from clip.custom_clip import get_coop, get_shift_v2, get_clip_lora_v2
from data.imagnet_prompts import imagenet_classes
from data.datautils import AugMixAugmenter, build_dataset, build_dataset_adv
from utils.tools import Summary, AverageMeter, ProgressMeter, accuracy, load_model_weight, set_random_seed
from data.cls_to_names import *
from data.fewshot_datasets import fewshot_datasets
from data.imagenet_variants import thousand_k_to_200, imagenet_a_mask, imagenet_r_mask, imagenet_v_mask
import os

from instance_method.tpt import TPT
from instance_method.ctpt import CTPT
from instance_method.mta import MTA
from instance_method.rlcf import RLCF
from instance_method.clipzs import CLIPZS
from instance_method.zero import ZERO
from instance_method.tps import TPS
from instance_method.ttl import TTL
from instance_method.rtpt import RTPT
from instance_method.sts import STS
from instance_method.use import USE
from instance_method.se import SE
from instance_method.clip import CLIP
from utils.use import select_and_cal_ent

import time

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s

def concat_dict(dict1, dict2):
    assert len(dict1.keys()) > 0
    for key in dict1.keys():
        if key in ['target']:
            dict1[key] = dict1[key].cpu()
        else:
            dict1[key] = dict1[key].unsqueeze(0).cpu()
    if dict2 is None:
        return dict1
    else:
        assert dict2.keys() == dict1.keys()
        for key in dict1.keys():
            dict2[key] = torch.cat([dict2[key], dict1[key]], dim=0)
        return dict2

def main():
    args = parser.parse_args()
    assert args.gpu is not None

    set_random_seed(args.seed)
    print("Use GPU: {} for training".format(args.gpu))

    cudnn.benchmark = True

    # norm stats from clip.load() # NOTE: normalize are implemented in Model forward()
    normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                     std=[0.26862954, 0.26130258, 0.27577711])

    # iterating through eval datasets
    dset = args.test_sets

    # creat log
    args.output_dir = os.path.join(args.output_dir, args.arch.replace('/', ''), 'seed_' + str(args.seed), args.algorithm)

    if not os.path.exists(args.output_dir):
        os.system('mkdir -p ' + args.output_dir)
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    args.out_file = open(os.path.join(args.output_dir, 'log_' + dset + '.txt'), 'w')
    args.out_file.write(print_args(args)+'\n')
    args.out_file.flush()

    if True:
        if args.algorithm in ['sts', 'use','se', 'tpt', 'mta', 'ctpt', 'rlcf', 'tps', 'ttl', 'rtpt', 'zero', 'clipzs', 'clip']:
            base_transform = transforms.Compose([
                transforms.Resize(args.resolution, interpolation=BICUBIC),
                transforms.CenterCrop(args.resolution)])
            preprocess = transforms.Compose([
                transforms.ToTensor(),
                normalize
                ])
            data_transform = AugMixAugmenter(base_transform, preprocess, n_views=args.batch_size-1, 
                                            augmix=len(dset)>1)
            batchsize = 1
            val_dataset = build_dataset(dset, data_transform, args.data, mode=args.dataset_mode)

            print_log = "number of test samples: {}".format(len(val_dataset))
            args.out_file.write(print_log + '\n')
            args.out_file.flush()
            print(print_log+'\n')

            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batchsize, shuffle=False,
                        num_workers=args.workers, pin_memory=True)
        else:
            raise NotImplementedError

        print("evaluating: {}".format(dset))

        if len(dset) > 1: 
            # fine-grained classification datasets
            classnames = eval("{}_classes".format(dset.lower()))
        else:
            assert dset in ['A', 'R', 'K', 'V', 'I']
            classnames_all = imagenet_classes
            classnames = []
            if dset in ['A', 'R', 'V']:
                label_mask = eval("imagenet_{}_mask".format(dset.lower()))
                if dset == 'R':
                    for i, m in enumerate(label_mask):
                        if m:
                            classnames.append(classnames_all[i])
                else:
                    classnames = [classnames_all[i] for i in label_mask]
            else:
                classnames = classnames_all

        # ##########  Model  ##########
        # ... (前面的代码保持不变)
# 
        if True:
            if args.algorithm in ['sts', 'use','se', 'tpt', 'mta', 'ctpt', 'rlcf', 'clipzs', 'zero', 'rtpt','clip']:
                model = get_coop(args.arch, classnames, args.gpu, args.n_ctx, args.ctx_init)
            elif args.algorithm in ['tps']:
                model = get_shift_v2(args.arch, classnames, args.gpu, args.n_ctx, args.ctx_init)
                model.add_shifter()
            elif args.algorithm in ['ttl']:
                model = get_clip_lora_v2(args.arch, classnames, args.gpu, args.n_ctx, args.ctx_init)
                model.add_lora()

            if args.load is not None:
                print("Use pre-trained soft prompt (CoOp) as initialization")
                load_path = {
                    'RN50': 'rn50_ep50_16shots/nctx4_cscFalse_ctpend/seed2/prompt_learner/model.pth.tar-50',
                    'ViT-B/16': 'vit_b16_ep50_16shots/nctx4_cscFalse_ctpend/seed2/prompt_learner/model.pth.tar-50'
                }[args.arch]
                load_path = os.path.join(args.load, load_path)
                pretrained_ctx = torch.load(load_path, weights_only=False, map_location='cpu')['state_dict']['ctx']
                # print(pretrained_ctx)
                assert pretrained_ctx.size()[0] == args.n_ctx
                args.pretrained_ctx = pretrained_ctx
            
            # if args.load_tecoa:
            #     print('loading tecoa')
            #     args.robust_pretrain_path = {
            #         'RN50': 'your_cache_path/tecoa/rn50_eps1.pth.tar',
            #         'ViT-B/32': 'your_cache_path/tecoa/vitb32_eps4.pth.tar'
            #     }[args.arch]
            #     robust_state_dict = torch.load(args.robust_pretrain_path, map_location='cpu')
            #     model.image_encoder.load_state_dict(robust_state_dict['vision_encoder_state_dict'])

        print("=> Model created: visual backbone {}".format(args.arch))
        
        if not torch.cuda.is_available():
            print('using CPU, this will be slow')
        else:
            assert args.gpu is not None
            torch.cuda.set_device(args.gpu)
            model = model.cuda(args.gpu)

        if args.algorithm == 'tpt':
            tta_trainer = TPT(model, args.gpu)
        elif args.algorithm == 'ctpt':
            tta_trainer = CTPT(model, args.gpu)
        elif args.algorithm == 'mta':
            tta_trainer = MTA(model, args.gpu)
        elif args.algorithm == 'rlcf':
            tta_trainer = RLCF(model, args.gpu)
        elif args.algorithm == 'clipzs':
            tta_trainer = CLIPZS(model, args.gpu)
        elif args.algorithm == 'zero':
            tta_trainer = ZERO(model, args.gpu)
        elif args.algorithm == 'tps':
            tta_trainer = TPS(model, args.gpu)
        elif args.algorithm == 'ttl':
            tta_trainer = TTL(model, args.gpu)
        elif args.algorithm == 'rtpt':
            tta_trainer = RTPT(model, args.gpu)
        elif args.algorithm == 'use':
            tta_trainer = USE(model, args.gpu)
        elif args.algorithm == 'se':
            tta_trainer = SE(model, args.gpu)
        elif args.algorithm == 'clip':
            tta_trainer = CLIP(model, args.gpu)
        elif args.algorithm == 'sts':
            tta_trainer = STS(model, args.gpu)
        else:  
            raise NotImplementedError
        
        zs1 = AverageMeter('TTA_Acc@1', ':6.2f', Summary.AVERAGE)
        tta1 = AverageMeter('TTA_Acc@1', ':6.2f', Summary.AVERAGE)

        tta_trainer.model.eval()
        for i, data in enumerate(val_loader):

            assert args.gpu is not None
            target = data[1]
            old_target = target

            images = data[0]
            if isinstance(images, list):
                for k in range(len(images)):
                    images[k] = images[k].cuda(args.gpu, non_blocking=True)
                image = images[0]
            else:
                if len(images.size()) > 4:
                    # when using ImageNet Sampler as the dataset
                    assert images.size()[0] == 1
                    images = images.squeeze(0)
                images = images.cuda(args.gpu, non_blocking=True)
                image = images

            images = torch.cat(images, dim=0)
            pred_tta = tta_trainer.adaptation_process(image, images, args)

            acc_tta = float(int(pred_tta) == int(target.item()))
            zs1.update(acc_tta * 100, 1)

            if (i + 1) % args.print_freq == 0:
                print_log = 'iter:{}/{}, tta_acc1={:.3f}'.format(i + 1, len(val_loader), zs1.avg)
                args.out_file.write(print_log + '\n')
                args.out_file.flush()
                print(print_log + '\n')

        print_log = 'iter:{}/{}, tta_acc1={:.3f}'.format(i + 1, len(val_loader), zs1.avg)
        args.out_file.write(print_log + '\n')
        args.out_file.flush()
        print(print_log+'\n')

        del val_dataset, val_loader
        print_log = "=> Acc@1 on [{}]: TTA {:.3f}".format(dset, zs1.avg)
      
        args.out_file.write(print_log + '\n')
        args.out_file.flush()
        print(print_log+'\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test-time Prompt Tuning')
    parser.add_argument('--data', type=str, default='/data1/shared/', help='path to dataset root')
    parser.add_argument('--test_sets', type=str, default='A/R/V/K/I', help='test dataset (multiple datasets split by slash)')
    parser.add_argument('--dataset_mode', type=str, default='test', help='which split to use: train/val/test')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='RN50')
    parser.add_argument('--resolution', default=224, type=int, help='CLIP image resolution')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-b', '--batch-size', default=64, type=int, metavar='N')
    parser.add_argument('-p', '--print-freq', default=200, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--gpu', default=0, type=int, help='GPU id to use.')
    parser.add_argument('--n_ctx', default=4, type=int, help='number of tunable tokens')
    parser.add_argument('--ctx_init', default='a_photo_of_a', type=str, help='init tunable prompts')
    parser.add_argument('--load', default=None, type=str, help='path to a pre-trained coop/cocoop') # ./coop_weight/to_gdrive/
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output_dir', type=str, default='temp')
    parser.add_argument('--lr', '--learning-rate', default=5e-3, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--selection_p', default=0.1, type=float, help='confidence selection percentile')
    parser.add_argument('--tta_steps', default=1, type=int, help='test-time-adapt steps')
    parser.add_argument('--algorithm', type=str, default='tpt', choices=['sts', 'use','se', 'tpt', 'mta', 'ctpt', 'rlcf', 'clipzs', 'zero', 'tps', 'ttl', 'rtpt', 'clip'])
    parser.add_argument('--load_tecoa', action='store_true')
    parser.add_argument('--reset', action='store_true')
    parser.add_argument('--scale', default=0.4, type=float)
    parser.add_argument('--beta', default=-1, type=float)
    parser.add_argument('--alpha', default=None, type=float, help='manual alpha value for method (if not provided, will be calculated automatically)')
    parser.add_argument('--skip', action='store_true', help='enable the skip strategy')
    parser.add_argument('--infer', action='store_true', help='only provide the training-free results')

    main()