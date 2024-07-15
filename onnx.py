import torch

import onnx
from models.LFFANet import LISDNet

net=LISDNet()
checkpoint = torch.load("/home/cyy/code/xxm/amfu/AMFU-net/result/mydata_MyNet_57.99/mIoU__MyNet_NUAA-SIRST_epoch.pth.tar")
net.load_state_dict(checkpoint['state_dict'])
net.eval()

input=torch.randn(1,3,512,512,requires_grad=True)

torch.onnx.export(net,
                  input,
                  "LISDNet.onnx",
                  opset_version=12,
                  export_params=True,
                  do_constant_folding=True,
                  input_names=['input'],
                  output_names=['out1','out2','out3','out4','out5','pred',],
                  dynamic_axes={'input':{0:'B'},
                                'out1':{0:'B'},
                                'out2':{0:'B'},
                                'out3':{0:'B'},
                                'out4':{0:'B'},
                                'out5':{0:'B'},
                                'pred':{0:'B'},})
print("ok")