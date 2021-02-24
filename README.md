# gxzy_demo

## UNet lung-CT semantic segmentation
将UNet用于肺部影像语义分割，数据源于Kaggle:https://www.kaggle.com/kmader/finding-lungs-in-ct-data，该影像及其label均为单通道 ，大小512*512，对于label图像进行了处理将255变为了1，并分为训练集、测试集、验证集（187、53、27张）

### quick start
#### train
1.在lung-CT文件夹里面下载处理好的图片  
2.python3 main.py  
#### perdict
1.如果自己没有训练模型，在saved文件夹里面下载训练好的模型  
2.python3 predict.py

### 替换数据
#### 二分类
1.修改train_config.json里面的路径变为自己的数据路径  
2.如果输入图片为三通道，修改model/unet.py的UNet class的inchannels为3，并在predict.py里面修改Img_channel=3  
3.如图片大小不是512*512，请在predict.py里面修改Height,Width  
