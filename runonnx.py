import os
import time

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torchvision import transforms


import onnxruntime as ort
def load_datasetfortest(root):
    val_img_ids=[]
    file_names=os.listdir(root)
    for file_name in file_names:

        val_img_ids.append(file_name)

    return val_img_ids
class DemoLoader (Dataset):
    """Iceberg Segmentation dataset."""
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
        img  = img.resize ((512, 512), Image.BILINEAR)

        # final transform
        img = np.array(img)
        return img

    def __len__(self):
        return len(self._items)

class Trainer():
    def __init__(self):

        self.img_dir   ='/home/cyy/dataset/detectdata/mydata/vistest'
        save_path=os.path.join(self.img_dir,"result")
        self.save_path='/home/cyy/dataset/detectdata/mydata/visutal_test'
        self.model_path='/home/cyy/code/xxm/amfu/LISDNet.onnx'



        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        self.val_img_ids = load_datasetfortest(self.img_dir)
        # Preprocess and load data
        input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
        data = DemoLoader(self.img_dir,self.val_img_ids, base_size=256, crop_size=256,
                          transform=input_transform, suffix='')

        # img_origin      = args.img_demo_dir+'/'+args.img_demo_index+args.suffix
        self.test_data = DataLoader(dataset=data, batch_size=1, num_workers=1,
                                    drop_last=False)

        self.model=ort.InferenceSession(self.model_path)

        self.output_name = self.get_output_name()

        print("Model Initializing")
        self.alltime=0
        # Load Checkpoint


        # Test
        #self.model.eval()
    def running(self):
        tbar = tqdm(self.test_data)
        num=0
        for i, (data) in enumerate(tbar):
            img = data
            time1=time.time()
            #input_feed=self.get_input_feed(img)
            img.unsqueeze(0)
            print(img.shape)
            preds = self.model.run(self.output_name,{'input':[img]})
            time2=time.time()
            self.alltime+=time2-time1
            pred = preds[-1]


            data = data[:, 0:1, :, :]
            #save_Pred_GT(pred,data,save_path,val_img_ids,num,'.png')
            self.save_pred(pred, data, self.save_path, self.val_img_ids, num, '.png')
            num+=1

        print("总用时为：{}s".format(self.alltime))
        print("单张图片用时为：{}s".format(self.alltime/len(self.test_data)))
    def get_input_name(self):
        input_name=[]
        for node in self.model.get_inputs():
            input_name.append(node.name)
        return input_name
    def get_output_name(self):
        output_name = []
        for node in self.model.get_outputs():
            output_name.append(node.name)
        return output_name
    def get_input_feed(self,img_tensor):
        input_feed = {}
        for name in self.model.get_inputs():
            input_feed[name]=img_tensor
        return input_feed
    def save_Pred_GTtwo_one(self,pred, labels, target_image_path, val_img_ids, num, suffix):

        predsss = np.array((pred > 0).cpu()).astype('int64') * 255
        predsss = np.uint8(predsss)
        label = labels * 255
        label = np.uint8(label.cpu())

        image_path = os.path.join(self.img_dir, (val_img_ids[num]))
        #print(image_path)
        or_img = cv2.imread(image_path)
        or_img = cv2.resize(or_img, (512, 512))

        img = Image.fromarray(predsss.reshape(512, 512))
        pred_img = np.array(img)

        label = Image.fromarray(label.reshape(512, 512))


        result_image = np.zeros((512, 1030, 3), dtype=np.uint8)
        result_image[:, :512, :] = or_img
        result_image[:, 512 + 6:, 0] = pred_img

        cv2.imwrite(os.path.join(target_image_path, (val_img_ids[num]) + suffix), result_image)
    def save_pred(self,pred, labels, target_image_path, val_img_ids, num, suffix):
        predsss = np.array((pred > 0).cpu()).astype('int64') * 255
        predsss = np.uint8(predsss)
        image_path = os.path.join(self.img_dir, (val_img_ids[num]))
        img = Image.fromarray(predsss.reshape(512, 512))
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


def main():
    trainer = Trainer()
    trainer.running()

if __name__ == "__main__":


    main()