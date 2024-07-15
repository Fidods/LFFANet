from thop import profile
from tqdm import tqdm
from thop import profile
from models.LFFANet import *
from models.MyNet__2 import MyNet__2

from myutils.parse_args_test import *

# Torch and visulization
from torchvision import transforms
from torch.utils.data import DataLoader

# Metric, loss .etc
from myutils.utils import *
from myutils.metric import *
from myutils.loss import *
from myutils.load_param_data import load_dataset, load_param


class Trainer(object):
    def __init__(self, args):

        # Initial

        self.ROC = ROCMetric(1, 10)
        self.PD_FA = PD_FA(1, 10)
        self.nIoU_metric = SamplewiseSigmoidMetric(1, score_thresh=0.5)
        self.mIoU = mIoU(1)
        self.save_prefix = '_'.join([args.model, args.dataset])
        self.best_PD = 0
        self.best_FA = 1000000000000000

        dataset_dir = os.path.join(args.root, args.dataset)
        train_img_ids, val_img_ids = load_dataset(dataset_dir)

        # Preprocess and load data
        input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
        testset = TestSetLoader(dataset_dir, img_id=val_img_ids, base_size=args.base_size, crop_size=args.crop_size,
                                transform=input_transform, suffix=args.suffix)
        self.test_data = DataLoader(dataset=testset, batch_size=args.test_batch_size, num_workers=args.workers,
                                    drop_last=False)

        # Network selection
        if args.model == 'LFFANet':
            tmodel = LFFANet()

        model = tmodel.cuda()
        # model.apply(weights_init_xavier)
        print("Model Initializing")
        self.model = model

        # Initialize evaluation metrics
        self.best_recall = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.best_precision = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        # Load trained model
        checkpoint = torch.load(args.model_dir)
        self.model.load_state_dict(checkpoint['state_dict'])

        # Test
        self.model.eval()
        tbar = tqdm(self.test_data)
        losses = AverageMeter()

        # Tic Toc
        self.timecheck = []
        tp = 0
        fp = 0
        with torch.no_grad():
            num = 0
            for i, (data, labels) in enumerate(tbar):
                data = data.cuda()
                labels = labels.cuda()

                preds = self.model(data)
                loss = 0
                for pred in preds:
                    loss += SoftIoULoss(pred, labels)
                loss /= len(preds)

                num += 1

                losses.update(loss.item(), pred.size(0))
                self.ROC.update(preds[len(preds) - 1], labels)
                self.mIoU.update(preds[len(preds) - 1], labels)
                self.PD_FA.update(preds[len(preds) - 1], labels)
                self.nIoU_metric.update(preds[len(preds) - 1], labels)

                ture_positive_rate, false_positive_rate, recall, precision = self.ROC.get()
                tp += ture_positive_rate
                fp += false_positive_rate
                self.best_recall.append(recall)
                self.best_precision.append(precision)
                FA, PD = self.PD_FA.get(len(self.test_data))
                _, mean_IOU = self.mIoU.get()
                _, n_IOU = self.nIoU_metric.get()
                tbar.set_description('test loss %.4f, mean_IoU: %.4f, n_IOU: %.4f' % (losses.avg, mean_IOU, n_IOU))
            if FA[0] * 1000000 < self.best_FA:
                self.best_FA = FA[0] * 1000000
            if PD[0] > self.best_PD:
                self.best_PD = PD[0]
            tp /= len(tbar)
            fp /= len(tbar)
            input = torch.randn(1, 3, 256, 256).cuda()
            flops, params = profile(model, inputs=(input,))
            print("FLOPs=", str(flops / 1e9) + '{}'.format("G"))
            print("Params=", str(params / 1e6) + '{}'.format("M"))
            print(self.best_FA)
            print(self.best_PD)
            print(tp)
            print(fp)
            import matplotlib.pyplot as plt
            plt.xlim([0, 0.001])
            plt.plot(fp, tp, 'b-', label="LFFANet")
            plt.xlabel("tp")
            plt.ylabel("fp")
            plt.title("ROC")
            plt.legend(loc='lower right')
            plt.show()



def main(args):
    trainer = Trainer(args)


if __name__ == "__main__":
    args = parse_args()
    print('---------------------', args.model, '---------------------')
    main(args)
