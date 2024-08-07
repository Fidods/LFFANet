from myutils.utils import *
import argparse


def parse_args():
    """Testing Options for Segmentation Experiments"""
    parser = argparse.ArgumentParser(description='LFFANet for IRSTD')

    # choose model
    parser.add_argument('--model', type=str, default='LFFANet',
                        help='LFFANet')

    # Deep supervision for AMFU-nets
    # data and pre-process
    parser.add_argument('--dataset', type=str, default='NUAA-SIRST',#IRSTD-1k
                        help='dataset name: NUDT-SIRST, NUAA-SIRST, NUST-SIRST')
    parser.add_argument('--root', type=str, default='/home/cyy/dataset/detectdata',
                        help='/home/cyy/dataset/detectdata')
    parser.add_argument('--model_dir', type=str,  ######## Change
                        default='D:/code/LISDNet/result/LFFANet/mIoU__LFFANet_NUAA-SIRST_epoch.pth.tar',
                        # Trained weight directory
                        help='')
    parser.add_argument('--mode', type=str, default='TXT', help='mode name:  TXT, Ratio')
    parser.add_argument('--suffix', type=str, default='.png')
    parser.add_argument('--workers', type=int, default=4,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--in_channels', type=int, default=3,
                        help='in_channel=3 for pre-process')
    parser.add_argument('--base_size', type=int, default=256,
                        help='base image size')
    parser.add_argument('--crop_size', type=int, default=256,
                        help='crop image size')

    #  hyper params for training
    parser.add_argument('--test_batch_size', type=int, default=1,
                        metavar='N', help='input batch size for \
                        testing (default: 32)')
    # select GPUs
    parser.add_argument('--gpus', type=str, default='0',
                        help='Training with GPUs, you can specify 1,3 for example.')

    # ROC threshold number of image
    parser.add_argument('--ROC_thr', type=int, default=10,
                        help='crop image size')

    # the parser
    args = parser.parse_args()

    return args