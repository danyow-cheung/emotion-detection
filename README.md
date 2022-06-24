# 基于yolov5的情感识别
使用模型：yolov5m 
数据标注:roboflow

1. 训练模型终端代码

   `python3 detect.py --source data/happy.mov --save-txt` 
   

2. 检测视频并保存的终端代码

   `python3 detect.py --weights runs/train/exp/weights/last.pt --source ../data/video/happy.mov --save-txt`
   
   
结果展示：

![happy-1](https://user-images.githubusercontent.com/76671016/175295182-9b5704c3-0223-4ddb-90f1-df5d3dbc7042.png)
![sad-1](https://user-images.githubusercontent.com/76671016/175295207-ed816258-ec41-4226-9a47-2ecb09aca55a.png)




其中：yolov5/runs/train/exp为yolov5s训练出的模型，exp1文件夹为yolov5m
