# import torch and utils
import matplotlib.pyplot as plt
from torch.optim import lr_scheduler
from tqdm             import tqdm
import torch.optim    as optim
from torchvision      import transforms
from torch.utils.data import DataLoader


from models.LFFANet import *

from myutils.parse_args_train import  parse_args
import torch.optim    as optim
# metric, loss .etc
from myutils.utils import *
from myutils.metric import *
from myutils.loss2 import *
from myutils.load_param_data import  load_dataset
from models.my_init_weights import *

def set_seeds(seed=3407):
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
class Trainer(object):
    def __init__(self, args):
        # Initial
        self.args = args
        self.ROC  = ROCMetric(1, 10)
        self.mIoU = mIoU(1)
        self.save_prefix = '_'.join([args.model, args.dataset])
        self.save_dir    = make_dir(args.dataset, args.model)
        self.x=[]#保存x，y坐标
        self.tr_loss=[]
        self.te_loss=[]
        save_train_log(args, self.save_dir)
        # Read image index from TXT
        dataset_dir =os.path.join(args.root,args.dataset)
        train_img_ids, val_img_ids = load_dataset(dataset_dir)

        # Preprocess and load data
        input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
        trainset        = TrainSetLoader(dataset_dir,img_id=train_img_ids,base_size=args.base_size,crop_size=args.crop_size,transform=input_transform,suffix=args.suffix)
        testset         = TestSetLoader (dataset_dir,img_id=val_img_ids,base_size=args.base_size, crop_size=args.crop_size, transform=input_transform,suffix=args.suffix)
        self.train_data = DataLoader(dataset=trainset, batch_size=args.train_batch_size, shuffle=True, num_workers=args.workers,drop_last=True)
        self.test_data  = DataLoader(dataset=testset,  batch_size=args.test_batch_size, num_workers=args.workers,drop_last=False)

        # Network selection
        # my network
        if args.model == 'LFFANet':
            model=LFFANet()

        # model init
        model           = model.cuda()
        #model.apply(weights_init_kaiming)
        model.apply(weights_init_orthogonal)
        print("Model Initializing")
        self.model      = model

        # Optimizer and lr scheduling
        if args.optimizer   == 'Adam':
            self.optimizer  = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
        elif args.optimizer == 'Adagrad':
            self.optimizer  = optim.Adagrad(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
        self.scheduler = lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=30)
        # Evaluation metrics
        self.best_iou       = 0
        self.best_recall    = [0,0,0,0,0,0,0,0,0,0,0]
        self.best_precision = [0,0,0,0,0,0,0,0,0,0,0]
    # Training
    def training(self,epoch):

        tbar = tqdm(self.train_data)
        iter = len(self.train_data)
        netpred=0
        self.model.train()
        losses = AverageMeter()

        for i, ( data, labels) in enumerate(tbar):
            data   = data.cuda()
            labels = labels.cuda()
            loss = 0
            pred = self.model(data)
            if args.model == 'AGPCNet':
                loss = SoftIoULoss(pred, labels)
                lens = len(pred)
            else:
                lens = len(pred)
                for k in range(lens):
                    temp = SoftIoULoss(pred[k], labels)
                    loss += temp
                    if k == lens - 1:
                        netpred = temp
                loss /= lens


            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            losses.update(loss.item(),lens)
            tbar.set_description('Epoch %d, training loss %.4f, pred loss %.4f' % (epoch, losses.avg,netpred))
        self.x.append(epoch)
        self.train_loss = losses.avg
        self.tr_loss.append(self.train_loss)

    # Testing
    def testing (self, epoch):
        tbar = tqdm(self.test_data)
        netpred=0
        self.model.eval()
        self.mIoU.reset()
        losses = AverageMeter()
        #iter = len(self.train_data)
        with torch.no_grad():
            for i, ( data, labels) in enumerate(tbar):
                data = data.cuda()
                labels = labels.cuda()
                loss=0
                lens=0
                if args.deep_supervision == 'DSV':
                    pred = self.model(data)
                    if args.model == 'AGPCNet':
                        loss=SoftIoULoss(pred,labels)
                        lens = len(pred)
                    else:
                        lens = len(pred)
                        for k in range(lens):
                            temp = SoftIoULoss(pred[k], labels)
                            loss += temp
                            if k == lens - 1:
                                netpred = temp
                        loss /= lens
                elif args.deep_supervision == 'None':
                    pred = self.model(data)
                    loss = SoftIoULoss(pred, labels)



                losses.update(loss.item(), lens)
                self.ROC .update(pred[lens-1], labels)
                self.mIoU.update(pred[lens-1], labels)
                ture_positive_rate, false_positive_rate, recall, precision = self.ROC.get()
                _, mean_IOU = self.mIoU.get()

                tbar.set_description('Epoch %d, test loss %.4f, pred loss %.4f, mean_IoU: %.4f' % (epoch, losses.avg,netpred, mean_IOU ))
                
            test_loss=losses.avg
            self.te_loss.append(test_loss)
        
        # save high-performance model
        save_model(mean_IOU, self.best_iou, self.save_dir, self.save_prefix,
                   self.train_loss, test_loss, recall, precision, epoch, self.model.state_dict())
        if self.best_iou < mean_IOU:
            self.best_iou = mean_IOU



def main(args):
    trainer = Trainer(args)

    for epoch in range(args.start_epoch, args.epochs):
        trainer.training(epoch)
        trainer.testing(epoch)
    print(trainer.best_iou)



if __name__ == "__main__":
    args = parse_args()
    #set_seeds(3407)
    print('---------------------',args.model,'---------------------')
    main(args)