import torch
import numpy as np
from PIL import Image
import cv2
import torch.nn as nn
from argparse import ArgumentParser

from mmdet.apis import (inference_detector,
                        init_detector, show_result_pyplot)
import os
import mmcv


class heatmap(nn.Module):
    def __init__(self, args, module_name=[], p_in=[], p_out=[]):
        super(heatmap, self).__init__()
        self.args = args
        self.module_name = module_name
        self.p_in = p_in
        self.p_out = p_out

    # 定义hook_fn，顾名思义就是把数值从
    def hook_fn(self, module, inputs, outputs):
        #print(self.module_name)
        self.module_name.append(module.__class__)
        self.p_in.append(inputs)
        self.p_out.append(outputs)

    def forward(self, model, img):
        # build the model from a config file and a checkpoint file
        #print(self.p_out)
        # model = init_detector(self.args.config, self.args.checkpoint, device=self.args.device)
        model.bbox_head.retina_cls.register_forward_hook(self.hook_fn)
        result = inference_detector(model, img)
        print(len(self.p_out))
        for k in range(len(self.p_out)):
            #print(self.p_in[k][0].shape)
            #print(self.p_out[k].shape)
            # show_feature_map(img, self.p_in[0][0], )
            show_feature_map(img, torch.sigmoid(self.p_out[k]), k + 2, self.args.save_dir)

#%%
def show_feature_map(img_src, conv_features, p_num, save_dir):
    '''可视化卷积层特征图输出
    img_src:源图像文件路径
    conv_feature:得到的卷积输出,[b, c, h, w]
    '''
    img = mmcv.imread(img_src)
    conv_features = conv_features.cpu()
    heat = conv_features.squeeze(0)#降维操作,尺寸变为(2048,7,7)
    heatmap = torch.mean(heat,dim=0)#对各卷积层(2048)求平均值,尺寸变为(7,7)

    heatmap = heatmap.numpy()#转换为numpy数组
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)#minmax归一化处理
    heatmap = cv2.resize(heatmap,(img.shape[1],img.shape[0]))#变换heatmap图像尺寸,使之与原图匹配,方便后续可视化
    heatmap = np.uint8(255*heatmap)#像素值缩放至(0,255)之间,uint8类型,这也是前面需要做归一化的原因,否则像素值会溢出255(也就是8位颜色通道)
    #heatmap = cv2.applyColorMap(heatmap,cv2.COLORMAP_HSV)#颜色变换
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # 将热力图应用于原始图像
    superimg = heatmap * 0.4 + img   #heatmap*0.4+np.array(img)[:,:,::-1]     #图像叠加，注意翻转通道，cv用的是bgr

    name = os.path.basename(img_src).split('.')[0]
    cv2.imwrite(os.path.join(save_dir, name + "_" + str(p_num) + '.png'), superimg)  # 将图像保存到硬盘

def main():
    parser = ArgumentParser()
    # parser.add_argument('img_path', help='Image file')
    parser.add_argument('save_dir', help='Dir to save heatmap image')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--device', default='cuda:0', help='Device used for inference')
    args = parser.parse_args()

    model = init_detector(args.config, args.checkpoint, device=args.device)
    for line in  open('/home/mst10512/mmdetection_219/data/VOCdevkit/VOC2007/ImageSets/Main/test.txt'):
        img_path = '/home/mst10512/mmdetection_219/data/VOCdevkit/VOC2007/JPEGImages/' + line.replace('\n', '') +'.jpg'
    #for filename in os.listdir(r"./" + args.img_path):
        HeatMap = heatmap(args, [], [], [])
        #img = args.img_path + "/" + filename
        HeatMap(model, img_path)


if __name__ == '__main__':
    main()