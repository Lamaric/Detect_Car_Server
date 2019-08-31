import argparse
import os, time
import numpy as np
import sys
from tqdm import tqdm

from mypath import Path
from dataloaders import make_data_loader
from dataloaders import make_test_data_loader
from modeling.sync_batchnorm.replicate import patch_replication_callback
from modeling.deeplab import *
from utils.loss import SegmentationLosses
from utils.calculate_weights import calculate_weigths_labels
from utils.lr_scheduler import LR_Scheduler
from utils.saver import Saver
from utils.summaries import TensorboardSummary
from torchvision.utils import make_grid
from utils.metrics import Evaluator
from PIL import Image
from dataloaders.utils import decode_seg_map_sequence
import cv2
import matplotlib
from torchvision import transforms
from PIL import Image
from dataloaders import custom_transforms_test as tr

class Tester(object):
    def __init__(self, args):
        self.args = args

        # Define network
        model = DeepLab(num_classes=32,
                        backbone=args.backbone,
                        output_stride=args.out_stride,
                        sync_bn=args.sync_bn,
                        freeze_bn=args.freeze_bn)

#         self.criterion = SegmentationLosses(weight=weight, cuda=args.cuda).build_loss(mode=args.loss_type)
        self.model = model
        
        # Define Evaluator
        self.evaluator = Evaluator(32)

        # Using cuda
        if args.cuda:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.args.gpu_ids)
            patch_replication_callback(self.model)
            self.model = self.model.cuda()

        # Resuming checkpoint
        self.best_pred = 0.0
        time_start = time.time()
        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'" .format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            if args.cuda:
                self.model.module.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint['state_dict'])
            self.best_pred = checkpoint['best_pred']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))

        # Clear start epoch if fine-tuning
        if args.ft:
            args.start_epoch = 0


    def validation(self, epoch, imagedir):
        self.model.eval()
        self.evaluator.reset()
        image = Image.open(imagedir)
        composed_transforms = transforms.Compose([
                        tr.FixScaleCrop(crop_size=513),
                        tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                        tr.ToTensor()])
        image = composed_transforms(image)
        image = image.unsqueeze(0)

        if self.args.cuda:
            image = image.cuda()
        with torch.no_grad():
            output = self.model(image)

        pred = output.data.cpu().numpy()
        pred = np.argmax(pred, axis=1)

        grid_image = make_grid(image[:3].clone().cpu().data, 3, normalize=True)
        grid_image = grid_image.numpy()
        grid_image = np.moveaxis(grid_image, 0,2)
        matplotlib.image.imsave('D:/Excise/Deecamp/backend/untitled/detect/test1/image.png', grid_image)

        grid_image = make_grid(decode_seg_map_sequence(torch.max(output[:3], 1)[1].detach().cpu().numpy()), 3, normalize=False, range=(0, 255))
        grid_image = grid_image.numpy()
        grid_image = np.moveaxis(grid_image,0,2)
        matplotlib.image.imsave('D:/Excise/Deecamp/backend/untitled/detect/test1/predict.png', grid_image)

        return grid_image


def detect(imagedir):

    parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Training")
    args = parser.parse_args()
    args.backbone = 'resnet'
    args.out_stride = 16
    args.dataset = 'pascal'
    args.use_sbd = True
    args.workers = 0
    args.base_size = 513
    args.freeze_bn = False
    args.loss_type = 'ce'
    args.start_epoch = 0
    args.test_batch_size = None
    args.momentum = 0.9
    args.weight_decay = 5e-4
    args.nesterov = False
    args.no_cuda = False
    args.gpu_ids = '0'
    args.seed = 1
    args.resume = 'D:/Excise/Deecamp/backend/untitled/detect/run/coco/deeplab-resnet/experiment_hd/checkpoint.pth.tar'
    args.ft = False
    args.eval_interval = 1
    args.no_val = False
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')

    args.sync_bn = False
    args.epochs = 30
    args.batch_size = 1
    args.lr = 0.01
    args.checkname = 'deeplab-'+str(args.backbone)

    torch.manual_seed(args.seed)
    tester = Tester(args)

    epoch=89
    # cv2.imshow('res', tester.validation(epoch, imagedir))
    # cv2.waitKey(0)

    return tester.validation(epoch, imagedir)


if __name__ == "__main__":
    img_dir = sys.argv[1]
    print(img_dir)
    detect(img_dir)
