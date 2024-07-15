# Basic module
import os.path
import time
import cv2
from torch.utils.data import DataLoader

# Torch and visulization
from torchvision      import transforms

from myutils.parse_args_vis import parse_args
# Metric, loss .etc
from myutils.utils import *
from myutils.loss import *
from myutils.load_param_data import load_param, load_dataset, load_datasetfortest

import onnxruntime as ort
# my model
from models.LFFANet import *
from tqdm             import tqdm
def load_datasetfortest(root):
    val_img_ids=[]
    file_names=os.listdir(root)
    for file_name in file_names:

        val_img_ids.append(file_name)

    return val_img_ids
class DemoLoader (Dataset):
    NUM_CLASS = 1
    def __init__(self, dataset_dir,img_id, transform=None,base_size=512,crop_size=480,suffix='.png'):
        super(DemoLoader, self).__init__()
        self.transform = transform
        self.images    = dataset_dir
        self.base_size = base_size
        self.crop_size = crop_size
        self.suffix    = suffix
        self._items = img_id

    def __getitem__(self, idx):
        img_id = self._items[idx]
        img_path=self.images+'/'+img_id
        img = Image.open(img_path).convert('RGB')
        img=self._demo_sync_transform(img)
        img = self.transform(img)
        return img

    def _demo_sync_transform(self, img):
        base_size = self.base_size
        img  = img.resize ((base_size, base_size), Image.BILINEAR)
        img = np.array(img)
        return img
    def __len__(self):
        return len(self._items)
class Trainer():
    def __init__(self,args):

        self.img_dir   =args.img_dir
        save_path=os.path.join(self.img_dir,"result")
        self.save_path='.vis/'
        self.model_path=args.model_dir

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        self.val_img_ids = load_datasetfortest(self.img_dir)
        # Preprocess and load data
        input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
        data_vis = DemoLoader(self.img_dir,self.val_img_ids, base_size=args.base_size, crop_size=256,
                          transform=input_transform, suffix='')
        self.test_data = DataLoader(dataset=data_vis, batch_size=1, num_workers=1,
                                    drop_last=False)
        model       = LISDNet()
        model = model.cuda()

        print("Model Initializing")
        self.model = model
        self.alltime=0
        # Load Checkpoint
        checkpoint = torch.load(self.model_path)
        self.model.load_state_dict(checkpoint['state_dict'])
        # Test
        self.model.eval()

    def vis(self):
        tbar = tqdm(self.test_data)
        num = 0
        for i, (data) in enumerate(tbar):
            img = data.cuda()
            time1 = time.time()

            preds = self.model(img)
            time2 = time.time()
            self.alltime += time2 - time1
            pred = preds[-1]
            self.save_pred(pred, self.save_path, self.val_img_ids, num)
            num += 1
        print("总用时为：{} s".format(self.alltime))
        print("单张图片用时为：{} ms".format((self.alltime / len(self.test_data)*1000)))
        print("FPS为：{} ".format(1000/(self.alltime / len(self.test_data))))
    def save_Pred_GTtwo_one(self,pred, labels, target_image_path, val_img_ids, num,base_size):

        predsss = np.array((pred > 0).cpu()).astype('int64') * 255
        predsss = np.uint8(predsss)
        label = labels * 255
        label = np.uint8(label.cpu())
        img_size=(base_size,base_size)
        image_path = os.path.join(self.img_dir, (val_img_ids[num]))

        or_img = cv2.imread(image_path)
        or_img = cv2.resize(or_img, img_size)

        img = Image.fromarray(predsss.reshape(img_size))
        pred_img = np.array(img)

        label = Image.fromarray(label.reshape(img_size))

        result_image = np.zeros((base_size, base_size*2+5, 3), dtype=np.uint8)
        result_image[:, :512, :] = or_img
        result_image[:, 512 + 6:, 0] = pred_img

        cv2.imwrite(os.path.join(target_image_path, (val_img_ids[num])), result_image)
    def save_pred(self,pred,target_image_path, val_img_ids, num, suffix,base_size):
        predsss = np.array((pred > 0).cpu()).astype('int64') * 255
        predsss = np.uint8(predsss)
        img = Image.fromarray(predsss.reshape(base_size, base_size))
        pred_img = np.array(img)
        cv2.imwrite(os.path.join(target_image_path, (val_img_ids[num])), pred_img)
    def save_Pred_GT__(self,pred,image,target_image_path,num):
        predsss = np.array((pred > 0).cpu()).astype('int64') * 255
        image = image * 255
        image = np.uint8(image.cpu())

        pred_img = np.array(image)

        img = Image.fromarray(predsss.reshape(256, 256))
        img.save(target_image_path + '/' + '%s_Pred' % (target_image_path[num]) + '.jpg')

        img = Image.fromarray(image.reshape(256, 256))
        pred_img = np.array(img)

        img.save(target_image_path + '/' + '%s_TRUE' % (target_image_path[num]) + '.jpg')

    def center(self, mask):  ##返回检测到的目标的中心点位置
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Iterate through each contour
        result = []
        for contour in contours:
            # Calculate the centroid of the contour
            M = cv2.moments(contour)
            if M["m00"] != 0:
                centroid_x = int(M["m10"] / M["m00"])
                centroid_y = int(M["m01"] / M["m00"])
                centroid = (centroid_x, centroid_y)

                # Draw a circle at the centroid position
                result.append(centroid)
        return result




if __name__ == "__main__":
    args=parse_args()
    trainer = Trainer(args)
    trainer.vis()