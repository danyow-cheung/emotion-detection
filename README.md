# 基于yolov5的情感识别

使用模型：yolov5m 

数据标注使用roboflow

1. 训练模型终端代码

   `python3 detect.py --source data/happy.mov --save-txt` 

2. 检测视频并保存的终端代码

   `python3 detect.py --weights runs/train/exp/weights/last.pt --source ../data/video/happy.mov --save-txt`

