import cv2
import numpy as np
import time
from tqdm import tqdm
from ultralytics import YOLO
from ultralytics.yolo.data.augment import LetterBox
from ultralytics.yolo.utils import ops
import matplotlib.pyplot as plt
from ultralytics.nn.autobackend import AutoBackend
import torch
import time




# 有 GPU 就用 GPU，没有就用 CPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = AutoBackend(weights='weight/Person.pt',
                    device=device,
                    dnn=False,                   # 是否将 OpenCV 的 DNN 库用于推理
                    data='weight/Person.yaml',
                    fp16=False,
                    fuse=True,
                    verbose=False)

_ = model.eval().to(device)

imgsz = [640, 640]



# 框（rectangle）可视化配置
bbox_color = (150, 0, 0)             # 框的 BGR 颜色
bbox_thickness = 2                   # 框的线宽

# 框类别文字
bbox_labelstr = {
    'font_size':1,         # 字体大小
    'font_thickness':2,    # 字体粗细
    'offset_x':0,          # X 方向，文字偏移距离，向右为正
    'offset_y':-10,        # Y 方向，文字偏移距离，向下为正
}





def process_frame(img_bgr,
                  # 1,2,3,4 分别对应左上，右上，右下，左下四个点
                  hl1=2.0 / 10,  # 监测区域高度距离图片顶部比例
                  wl1=3.0 / 10,  # 监测区域高度距离图片左部比例
                  hl2=2.0 / 10,  # 监测区域高度距离图片顶部比例
                  wl2=7.0 / 10,  # 监测区域高度距离图片左部比例
                  hl3=7.0 / 10,  # 监测区域高度距离图片顶部比例
                  wl3=7.0 / 10,  # 监测区域高度距离图片左部比例
                  hl4=7.0 / 10,  # 监测区域高度距离图片顶部比例
                  wl4=3.0 / 10,  # 监测区域高度距离图片左部比例
                  ):
    '''
    输入摄像头画面 bgr-array，输出图像 bgr-array
    '''

    # 记录该帧开始处理的时间
    start_time = time.time()

    # BGR 转 RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # 缩放图像尺寸
    pre_transform_result = LetterBox(new_shape=imgsz, auto=True)(image=img_rgb)

    #创建一个全黑的mask，大小为图像大小
    mask = np.zeros([pre_transform_result.shape[0], pre_transform_result.shape[1]], dtype=np.uint8)

    #定义一个区域，此区域由4个点构成，此区域即为需要检测的区域
    pts = np.array([[int(pre_transform_result.shape[1] * wl1), int(pre_transform_result.shape[0] * hl1)],  # pts1
                    [int(pre_transform_result.shape[1] * wl2), int(pre_transform_result.shape[0] * hl2)],  # pts2
                    [int(pre_transform_result.shape[1] * wl3), int(pre_transform_result.shape[0] * hl3)],  # pts3
                    [int(pre_transform_result.shape[1] * wl4), int(pre_transform_result.shape[0] * hl4)]], np.int32)

    #将需要检测的区域设置为白色，并填充在mask上，此时的mask，需要检测的区域为全白，其余区域为全黑
    mask = cv2.fillPoly(mask, [pts], (255, 255, 255))

    #使用cv2.add应用mask
    pre_transform_result = cv2.add(pre_transform_result, np.zeros(np.shape(pre_transform_result), dtype=np.uint8), mask=mask)

    # 预处理-归一化
    input_tensor = pre_transform_result / 255

    # 预处理-构造输入张量               n hwc
    input_tensor = np.expand_dims(input_tensor, axis=0)  # 加 Batch 维度
    input_tensor = input_tensor.transpose((0, 3, 1, 2))  # N, C, H, W
    input_tensor = np.ascontiguousarray(input_tensor)  # 将内存不连续存储的数组，转换为内存连续存储的数组，使得内存访问速度更快
    input_tensor = torch.from_numpy(input_tensor).to(device).float()  # 转 Pytorch Tensor
    # input_tensor = input_tensor.half() # 是否开启半精度，即 uint8 转 fp16，默认转 fp32

    # 执行推理预测
    preds = model(input_tensor)

    # 后处理-置信度阈值过滤、非极大值抑制NMS过滤
    pred = ops.non_max_suppression(preds, conf_thres=0.25, iou_thres=0.7, nc=1)[0]

    # 解析目标检测预测结果
    # 将缩放之后图像的预测结果，投射回原始尺寸
    pred[:, :4] = ops.scale_boxes(pre_transform_result.shape[:2], pred[:, :4], img_bgr.shape).round()
    pred_det = pred[:, :6].cpu().numpy()
    num_bbox = len(pred_det)
    # bboxes_cls = pred_det[:, 5]  # 类别
    # bboxes_conf = pred_det[:, 4]  # 置信度
    bboxes_xyxy = pred_det[:, :4].astype('uint32')  # 目标检测框 XYXY 坐标

    # OpenCV可视化
    for idx in range(num_bbox):  # 遍历每个框

        # 获取该框坐标
        bbox_xyxy = bboxes_xyxy[idx]

        # 获取框的预测类别（对于关键点检测，只有一个类别）
        bbox_label = model.names[0]

        # 画框
        img_bgr = cv2.rectangle(img_bgr, (bbox_xyxy[0], bbox_xyxy[1]), (bbox_xyxy[2], bbox_xyxy[3]), bbox_color,
                                bbox_thickness)

        # 写框类别文字：图片，文字字符串，文字左上角坐标，字体，字体大小，颜色，字体粗细
        img_bgr = cv2.putText(img_bgr, bbox_label,
                              (bbox_xyxy[0] + bbox_labelstr['offset_x'], bbox_xyxy[1] + bbox_labelstr['offset_y']),
                              cv2.FONT_HERSHEY_SIMPLEX, bbox_labelstr['font_size'], bbox_color,
                              bbox_labelstr['font_thickness'])

    #画出需要检测的区域
    pts = np.array([[int(img_bgr.shape[1] * wl1), int(img_bgr.shape[0] * hl1)],  # pts1
                    [int(img_bgr.shape[1] * wl2), int(img_bgr.shape[0] * hl2)],  # pts2
                    [int(img_bgr.shape[1] * wl3), int(img_bgr.shape[0] * hl3)],  # pts3
                    [int(img_bgr.shape[1] * wl4), int(img_bgr.shape[0] * hl4)]], np.int32)  # pts4
    # pts = pts.reshape((-1, 1, 2))
    zeros = np.zeros((img_bgr.shape), dtype=np.uint8)
    mask = cv2.fillPoly(zeros, [pts], color=(0, 165, 255))
    img_bgr = cv2.addWeighted(img_bgr, 1, mask, 0.2, 0)
    cv2.polylines(img_bgr, [pts], True, (255, 255, 0), 3)
    # plot_one_box(dr, im0, label='Detection_Region', color=(0, 255, 0), line_thickness=2)




    # 记录该帧处理完毕的时间
    end_time = time.time()
    # 计算每秒处理图像帧数FPS
    FPS = 1 / (end_time - start_time)

    # 在画面上写字：图片，字符串，左上角坐标，字体，字体大小，颜色，字体粗细
    FPS_string = 'FPS  {:.2f}'.format(FPS)  # 写在画面上的字符串
    img_bgr = cv2.putText(img_bgr, FPS_string, (25, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (255, 0, 255), 2)

    return img_bgr




#定义网络摄像头地址
url="http://admin:admin@192.168.137.249:8081"
# 获取摄像头，传入0表示获取系统默认摄像头，传入url访问网络摄像头
cap = cv2.VideoCapture(url)

# 打开cap
cap.open(url)

# 无限循环，直到break被触发
while cap.isOpened():

    # 获取画面
    success, frame = cap.read()

    if not success:  # 如果获取画面不成功，则退出
        print('获取画面不成功，退出')
        break

    ## 逐帧处理
    frame = process_frame(frame)

    # 展示处理后的三通道图像
    cv2.imshow('my_window', frame)

    key_pressed = cv2.waitKey(60)  # 每隔多少毫秒毫秒，获取键盘哪个键被按下
    # print('键盘上被按下的键：', key_pressed)

    if key_pressed in [ord('q'), 27]:  # 按键盘上的q或esc退出（在英文输入法下）
        break

# 关闭摄像头
cap.release()

# 关闭图像窗口
cv2.destroyAllWindows()

