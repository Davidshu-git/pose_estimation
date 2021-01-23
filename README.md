# 儿童运动障碍姿态估计骨骼生成
基于open-pose的轻量化版本的骨骼点生成器，代码基于[Daniil-Osokin
/lightweight-human-pose-estimation.pytorch](https://github.com/Daniil-Osokin/lightweight-human-pose-estimation.pytorch)
## 主要内容
- [环境配置](#环境配置)
- [预准备文件](#预准备文件)
- [运行](#运行)
- [引用](#引用)
## 环境配置
- Python 3.7.6
- PyTorch
```shell
pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
```
- opencv
```shell
pip install opencv-contrib-python
```
- cocotools
```shell
pip install pycocotools==2.0.0
```
- cython/pandas/matplotlib/openyxl
```shell
pip install cython pandas matplotlib openpyxl
```
## 预准备文件
- 采用已经训练好的模型参数，参数下载链接：https://download.01.org/opencv/openvino_training_extensions/models/human_pose_estimation/checkpoint_iter_370000.pth
## 运行
- 将模型checkpoint_iter_370000.pth放入文件夹checkpoint之中
- 执行`python main.py --video <处理视频数据路径>
- 将处理视频数据路径设置为0将使用默认摄像头进行实时动作捕捉
## 引用:
主要使用了这篇工作提供的模型：
```
@inproceedings{osokin2018lightweight_openpose,
    author={Osokin, Daniil},
    title={Real-time 2D Multi-Person Pose Estimation on CPU: Lightweight OpenPose},
    booktitle = {arXiv preprint arXiv:1811.12004},
    year = {2018}
}
```
