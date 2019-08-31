# import argparse
# import os, time
# import numpy as np
# from tqdm import tqdm

# from mypath import Path
# from dataloaders import make_data_loader
# from dataloaders import make_test_data_loader
# from modeling.sync_batchnorm.replicate import patch_replication_callback
# from modeling.deeplab import *
# from utils.loss import SegmentationLosses
# from utils.calculate_weights import calculate_weigths_labels
# from utils.lr_scheduler import LR_Scheduler
# from utils.saver import Saver
# from utils.summaries import TensorboardSummary
# from torchvision.utils import make_grid
# from utils.metrics import Evaluator
# from PIL import Image
# from dataloaders.utils import decode_seg_map_sequence
# import cv2
# import matplotlib
# from torchvision import transforms
# from PIL import Image
# from dataloaders import custom_transforms_test as tr


import argparse
import os, time
import numpy as np


from modeling.sync_batchnorm.replicate import patch_replication_callback
from modeling.deeplab import *
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

    def validation(self, epoch, img_dir, save_path):
        self.model.eval()
        self.evaluator.reset()
        image = Image.open(img_dir)
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

        # grid_image = make_grid(image[:3].clone().cpu().data, 3, normalize=True)
        # grid_image = grid_image.numpy()
        # grid_image = np.moveaxis(grid_image,0,2)
        # matplotlib.image.imsave('D:/Excise/Deecamp/backend/untitled/detect/test1/_image.png', grid_image)

        grid_image = make_grid(decode_seg_map_sequence(torch.max(output[:3], 1)[1].detach().cpu().numpy()), 3, normalize=False, range=(0, 255))
        grid_image = grid_image.numpy()
        grid_image = np.moveaxis(grid_image,0,2)
        matplotlib.image.imsave(save_path, grid_image)
        print('------OK!------')


def main():
    parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Training")
    parser.add_argument('--backbone', type=str, default='resnet',
                        choices=['resnet', 'xception', 'drn', 'mobilenet'],
                        help='backbone name (default: resnet)')
    parser.add_argument('--out-stride', type=int, default=16,
                        help='network output stride (default: 8)')
    parser.add_argument('--dataset', type=str, default='pascal',
                        choices=['pascal', 'coco', 'cityscapes'],
                        help='dataset name (default: pascal)')
    parser.add_argument('--use-sbd', action='store_true', default=True,
                        help='whether to use SBD dataset (default: True)')
    parser.add_argument('--workers', type=int, default=4,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--base-size', type=int, default=513,
                        help='base image size')
    parser.add_argument('--crop-size', type=int, default=513,
                        help='crop image size')
    parser.add_argument('--sync-bn', type=bool, default=None,
                        help='whether to use sync bn (default: auto)')
    parser.add_argument('--freeze-bn', type=bool, default=False,
                        help='whether to freeze bn parameters (default: False)')
    parser.add_argument('--loss-type', type=str, default='ce',
                        choices=['ce', 'focal'],
                        help='loss func type (default: ce)')
    # training hyper params
    parser.add_argument('--epochs', type=int, default=None, metavar='N',
                        help='number of epochs to train (default: auto)')
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--batch-size', type=int, default=None,
                        metavar='N', help='input batch size for \
                                training (default: auto)')
    parser.add_argument('--test-batch-size', type=int, default=None,
                        metavar='N', help='input batch size for \
                                testing (default: auto)')
    parser.add_argument('--use-balanced-weights', action='store_true', default=False,
                        help='whether to use balanced weights (default: False)')
    # optimizer params
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (default: auto)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        metavar='M', help='w-decay (default: 5e-4)')
    parser.add_argument('--nesterov', action='store_true', default=False,
                        help='whether use nesterov (default: False)')
    # cuda, seed and logging
    parser.add_argument('--no-cuda', action='store_true', default=
                        False, help='disables CUDA training')
    parser.add_argument('--gpu-ids', type=str, default='0',
                        help='use which gpu to train, must be a \
                        comma-separated list of integers only (default=0)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    # checking point
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')
    parser.add_argument('--checkname', type=str, default=None,
                        help='set the checkpoint name')
    # finetuning pre-trained models
    parser.add_argument('--ft', action='store_true', default=False,
                        help='finetuning on a different dataset')
    # evaluation option
    parser.add_argument('--eval-interval', type=int, default=1,
                        help='evaluuation interval (default: 1)')
    parser.add_argument('--no-val', action='store_true', default=False,
                        help='skip validation during training')

    parser.add_argument('--img_dir', type=str, default=r'D:/Excise/Deecamp/backend/untitled/pic/picpictest.jpg')
    parser.add_argument('--save_path', type=str, default=r'D:/Excise/Deecamp/backend/untitled/detect/test1/predict.png')

    args = parser.parse_args()
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
    tester.validation(epoch, args.img_dir, args.save_path)

if __name__ == "__main__":
   main()