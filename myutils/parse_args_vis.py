from utils import *
import argparse


def parse_args():
    """Testing Options for Segmentation Experiments"""
    parser = argparse.ArgumentParser(description='LISDNet for IRSTD')

    # choose model
    parser.add_argument('--model', type=str, default='LISDNet',
                        help='LISDNet')
    # data and pre-process
    parser.add_argument('--img_dir', type=str, default='/home/cyy/dataset/detectdata/mydata/vistest',
                        help='images directory')
    parser.add_argument('--model_dir', type=str,
                        default='/home/cyy/code/xxm/amfu/AMFU-net/result/NUAA-SIRST_MyNet_13_03_2024_15_30_52_wDS/mIoU__MyNet_NUAA-SIRST_epoch.pth.tar',
                        # Trained weight directory
                        help='Trained weight directory')
    parser.add_argument('--suffix', type=str, default='.png')
    parser.add_argument('--workers', type=int, default=4,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--base_size', type=int, default=256,
                        help='base image size')
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