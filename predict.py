import os
import cv2
import json
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from models import unet

def predict(config):
    device = torch.device('cuda:0')
    model = unet.UNet(num_classes=config['num_classes'])
    
    check_point = os.path.join(config['save_model']['save_path'], 'unet.pth')
    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.ToTensor()
            # transforms.Normalize(mean=[0.485, 0.456, 0.406],
            #                      std=[0.229, 0.224, 0.225])
        ]
    )
    model.load_state_dict(torch.load(check_point), False)
    model.cuda()
    model.eval()
    #输出文件夹
    pre_base_path = os.path.join(config['pre_dir'], 'predict_unet')
    if os.path.exists(pre_base_path) is False:
        os.mkdir(pre_base_path)
    pre_mask_path = os.path.join(pre_base_path, 'mask')
    if os.path.exists(pre_mask_path) is False:
        os.mkdir(pre_mask_path)
    pre_vis_path = os.path.join(pre_base_path, 'vis')
    if os.path.exists(pre_vis_path) is False:
        os.mkdir(pre_vis_path)
    
    with open(config['img_txt'], 'r', encoding='utf-8') as f:
        for line in f.readlines():
            image_name, _ = line.strip().split('\t')
            im = np.asarray(Image.open(image_name))
            im = im.reshape((512, 512, 1))
            im = transform(im).float().cuda()
            im = im.reshape((1,1,512,512))

            output = model(im)
            _, pred = output.max(1)
            pred = pred.view(512, 512)
            mask_im = pred.cpu().numpy().astype(np.uint8)

            file_name = image_name.split('/')[-1]
            save_label = os.path.join(pre_mask_path, file_name)
            cv2.imwrite(save_label, mask_im)
            print("写入{}成功".format(save_label))
            save_visual = os.path.join(pre_vis_path, file_name)
            print("开始写入{}".format(save_visual))
            translabeltovisual(save_label, save_visual)
            print("写入{}成功".format(save_visual))

def translabeltovisual(save_label, path): 
    visual_img = []
    im = cv2.imread(save_label,0)
    img_array = np.asarray(im)
    for i in img_array:
        for j in i:
            if j == 1:
                visual_img.append(255)
            else:
                visual_img.append(0) 
    visual_img = np.array(visual_img)
    visual_img = visual_img.reshape((512, 512))
    cv2.imwrite(path, visual_img)



if __name__ == "__main__":
    with open('predict_config.json', encoding='utf-8') as f:
        config = json.load(f)

    predict(config)


