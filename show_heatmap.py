import torch
import numpy as np
from PIL import Image
from torchvision import transforms, models
import matplotlib.cm
import copy
import cv2
import matplotlib.pyplot as plt
from argparse import ArgumentParser

from mmdet.apis import (inference_detector,
                        init_detector, show_result_pyplot)
import os
import mmcv

#%%
# 定义列表用于存储中间层的输入或者输出
module_name = []
p_in = []
p_out = []

# 定义hook_fn，顾名思义就是把数值从
def hook_fn(module, inputs, outputs):
    print(module_name)
    module_name.append(module.__class__)
    p_in.append(inputs)
    p_out.append(outputs)

# test a single image
#result = inference_detector(model, img)
# show the results
#show_result_pyplot(model, img, result, score_thr=score_thr)

#%%
def show_feature_map(img_src, conv_features, p_num, save_dir):
    '''可视化卷积层特征图输出
    img_src:源图像文件路径
    conv_feature:得到的卷积输出,[b, c, h, w]
    '''
    img = Image.open(img_src).convert('RGB')
    # height, width = img.size
    conv_features = conv_features.cpu()
    heat = conv_features.squeeze(0)#降维操作,尺寸变为(2048,7,7)
    heatmap = torch.mean(heat,dim=0)#对各卷积层(2048)求平均值,尺寸变为(7,7)
    # heatmap = torch.max(heat,dim=1).values.squeeze()

    heatmap = heatmap.numpy()#转换为numpy数组
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)#minmax归一化处理
    heatmap = cv2.resize(heatmap,(img.size[0],img.size[1]))#变换heatmap图像尺寸,使之与原图匹配,方便后续可视化
    heatmap = np.uint8(255*heatmap)#像素值缩放至(0,255)之间,uint8类型,这也是前面需要做归一化的原因,否则像素值会溢出255(也就是8位颜色通道)
    heatmap = cv2.applyColorMap(heatmap,cv2.COLORMAP_HSV)#颜色变换
    #plt.imshow(heatmap)
    #plt.show()
    # heatmap = np.array(Image.fromarray(heatmap).convert('L'))
    superimg = heatmap*0.4+np.array(img)[:,:,::-1]     #图像叠加，注意翻转通道，cv用的是bgr

    name = os.path.basename(img_src).split('.')[0]
    cv2.imwrite(os.path.join(save_dir, name + "_" + str(p_num) + '.png'), superimg)  # 将图像保存到硬盘
    # cv2.imwrite('./superimg.jpg',superimg)#保存结果

def main():
    parser = ArgumentParser()
    parser.add_argument('img', help='Image file')
    parser.add_argument('save_dir', help='Dir to save heatmap image')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--device', default='cuda:0', help='Device used for inference')
    args = parser.parse_args()


    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    model.bbox_head.retina_cls.register_forward_hook(hook_fn)
    result = inference_detector(model, args.img)
    print(len(p_out))
    for k in range(len(p_out)):
        print(p_in[k][0].shape)
        print(p_out[k].shape)
        # show_feature_map(img_file, p_in[k][0])
        show_feature_map(args.img, torch.sigmoid(p_out[k]), k+2, args.save_dir)
        #print()

if __name__ == '__main__':
    main()