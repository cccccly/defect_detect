
import os
import cv2
import math
from torch.autograd import Variable
from torchvision.transforms import transforms
import torch.nn as nn
import torch
from PIL import Image
import numpy as np

from wagtail.admin import messages
from django.shortcuts import render
from django.http import HttpResponse
from images.models import CustomImage
from dataAnalyze.models import StaticsDetail
from django.core.exceptions import ObjectDoesNotExist

from wagtail.admin.utils import popular_tags_for_model
from wagtail.documents.models import Document
from wagtail.images.permissions import permission_policy
from wagtail.admin.utils import PermissionPolicyChecker, permission_denied, popular_tags_for_model
# Create your views here.

permission_checker = PermissionPolicyChecker(permission_policy)

extension_ratio = 1
ratio = 0.5
# 创建Gabor滤波核
def BuildGaborKernels(ksize=5, lamda=2, sigma=1.12):
    # 生成多尺度，多方向的Gabor特征
    filters = []
    for theta in np.array([0, np.pi / 4, np.pi / 2, np.pi * 3 / 4]):
        kern = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, 0.5, 0, ktype=cv2.CV_32F)
        filters.append(kern)
    return filters


# 提取图像的Gabor特征
def GaborFeature(image):
    # 先创建Gabor滤波核
    kernels = BuildGaborKernels(ksize=3, lamda=3.5, sigma=1.75)
    dst_imgs = []
    img = np.zeros_like(image)
    for kernel in kernels:
        tmp = cv2.filter2D(image, cv2.CV_8UC3, kernel)
        img = np.maximum(img, tmp, img)
        dst_imgs.append(img)

    dst = dst_imgs[0]
    for temp in range(len(dst_imgs)):
        dst = cv2.addWeighted(dst, 0.5, dst_imgs[temp], 0.5, 0)  # 将不同方向的Gabor特征进行融合
    return dst  # 返回Gabor滤波后的图像


# 可以读取带中文路径的图
def cv_imread(image_path):
    cv_img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), 1)
    return cv_img


# 去除黑色空区，增强对比度
def remove_enhance(image):
    # image必须是三通道图像
    height, width = image.shape[0], image.shape[1]  # 获取图片的宽高信息
    HSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  # 将图像转换为HSV格式
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 将图像转换为单通道灰度图

    # 图像黑色区域RGB范围
    color = [([0, 0, 0], [60, 60, 60])]

    for (lower, upper) in color:
        # 创建NumPy数组
        lower = np.array(lower, dtype="uint8")  # 颜色下限
        upper = np.array(upper, dtype="uint8")  # 颜色上限
        # 根据阈值找到对应颜色
        mask = cv2.inRange(HSV, lower, upper)  # 查找处于范围区间
        mask = 255 - mask
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    maxArea = 0
    for numcontours, contour in enumerate(contours):
        if cv2.contourArea(contour) > maxArea:
            maxArea = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)

    blocPos = 0  # 记录黑色空区的相对位置，靠左还是靠右？    # 改动！！！
    cutOff = 0  # 记录裁剪了多宽的区域   # 改动！！！

    # 去除黑色区域，并将图像拉伸至1224 * 600
    if w < width:  # 图像是否存在黑色区域的判断条件     # 改动！！！
        if x != 0:  # 黑色区域在左侧
            blocPos = -1  # 改动！！！
            cutOff = int(x + 60 * ratio)  # 改动！！！
            img = img[y:y + h, int(x + 60 * ratio): x + w]  # 改动！！！
        else:  # 黑色区域在右侧
            blocPos = 1  # 改动！！！
            cutOff = int(width - w + 60 * ratio)  # 改动！！！
            img = img[y:y + h, x: int(x + w - 60 * ratio)]

    # 创建CLAHE对象
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # 限制对比度的自适应阈值均衡化
    dst = clahe.apply(img)
    return dst, blocPos, cutOff  # 返回预处理之后的图像，该图像为单通道灰度图


# 伽马变换，用于增强图像
def gammaTransform(c, gamma, image):
    h, w = image.shape  # 分别为图像高度、宽度以及深度
    new_img = np.zeros((h, w), dtype=np.float32)  # 返回一个给定形状和类型的用0填充的数组
    for i in range(h):
        for j in range(w):
            new_img[i, j] = c * math.pow(image[i, j], gamma)
    cv2.normalize(new_img, new_img, 0, 255, cv2.NORM_MINMAX)
    new_img = cv2.convertScaleAbs(new_img)
    return new_img


# 同态滤波函数，用于解决图像照度不均的问题
def homofilter(image):
    m, n = image.shape
    rL = 0.25
    rH = 5
    c = 3
    d0 = 9
    image_1 = np.log(image + 1)
    image_fft = np.fft.fft2(image_1)
    n1 = np.floor(m / 2)
    n2 = np.floor(n / 2)
    D = np.zeros((m, n))
    H = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            D[i, j] = ((i - n1) ** 2 + (j - n2) ** 2)
            H[i, j] = (rH - rL) * (np.exp(c * (-D[i, j] / (d0 ** 2)))) + rL
    image_2 = np.fft.ifft2(H * image_fft)
    image_3 = np.real(np.exp(image_2))
    return image_3  # 返回同态滤波后的图像


#还原训练所搭建的神经网络结构
class Unit(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Unit, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(num_features=out_channels)   #规范层将所有输入标准化为具有零平均值或者单位变异数，防止梯度消失或者爆炸
        self.relu = nn.ReLU()

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        output = self.relu(output)
        return output


class CNN(nn.Module):  #定义64层的深度神经网络结构
    def __init__(self,num_classes=22):
        super(CNN, self).__init__()

        self.unit1 = Unit(in_channels=3, out_channels=32)  #(32,32,32)
        self.unit2 = Unit(in_channels=32, out_channels=32)
        self.unit3 = Unit(in_channels=32, out_channels=32)

        self.pool1 = nn.MaxPool2d(kernel_size=2)  #(32,16,16)

        self.unit4 = Unit(in_channels=32, out_channels=64)  #(64,16,16)
        self.unit5 = Unit(in_channels=64, out_channels=64)
        self.unit6 = Unit(in_channels=64, out_channels=64)
        self.unit7 = Unit(in_channels=64, out_channels=64)

        self.pool2 = nn.MaxPool2d(kernel_size=2)  #(64,8,8)

        self.unit8 = Unit(in_channels=64, out_channels=128)  #(128,8,8)
        self.unit9 = Unit(in_channels=128, out_channels=128)
        self.unit10 = Unit(in_channels=128, out_channels=128)
        self.unit11 = Unit(in_channels=128, out_channels=128)

        self.pool3 = nn.MaxPool2d(kernel_size=2)  #(128,4,4)

        self.unit12 = Unit(in_channels=128, out_channels=128)
        self.unit13 = Unit(in_channels=128, out_channels=128)
        self.unit14 = Unit(in_channels=128, out_channels=128)

        self.avgpool = nn.AvgPool2d(kernel_size=4)  #(128,1,1)

        self.net = nn.Sequential(self.unit1, self.unit2, self.unit3, self.pool1, self.unit4, self.unit5, self.unit6,
                                 self.unit7, self.pool2, self.unit8, self.unit9, self.unit10, self.unit11, self.pool3,
                                 self.unit12, self.unit13, self.unit14, self.avgpool)

        self.out = nn.Linear(128, 22)


    def forward(self, input):
        output = self.net(input)
        output = output.view(output.size(0),-1)
        output = self.out(output)
        return output

checkpoint = torch.load(os.path.dirname(os.path.dirname(__file__))+'/media/cnn_params.pkl')
#导入训练好的模型参数
model = CNN(num_classes = 22)
model.load_state_dict(checkpoint)
model.eval()

# 使用模型预测瑕疵类别之前，先对图像尽心会预处理操作
def preprocess(image):
    # image必须是三通道图像
    img = cv2.resize(image, None, fx=0.5, fy=0.5)  # 图像分辨率（1224 * 600）
    HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # 将图像转换为HSV格式
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 将图像转换为单通道灰度图

    # 图像黑色区域RGB范围
    color = [([0, 0, 0], [60, 60, 60])]

    for (lower, upper) in color:
        # 创建NumPy数组
        lower = np.array(lower, dtype="uint8")  # 颜色下限
        upper = np.array(upper, dtype="uint8")  # 颜色上限
        # 根据阈值找到对应颜色
        mask = cv2.inRange(HSV, lower, upper)  # 查找处于范围区间
        mask = 255 - mask
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    maxArea = 0
    for numcontours, contour in enumerate(contours):
        if cv2.contourArea(contour) > maxArea:
            maxArea = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)

    # 去除黑色区域，并将图像拉伸至1224 * 600
    if w < 1224:  # 图像是否存在黑色区域的判断条件
        if x != 0:  # 黑色区域在左侧
            img = img[y:y + h, x + 30: x + w]
        else:  # 黑色区域在右侧
            img = img[y:y + h, x: x + w - 30]

        img = cv2.resize(img, (1224, 600))

    # 创建CLAHE对象
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # 限制对比度的自适应阈值均衡化
    dst = clahe.apply(img)
    return dst  # 返回预处理之后的图像，该图像为单通道灰度图


# 瑕疵类型预测函数
def predict_defect(image_path):
    image = cv_imread(image_path)
    # 对图像进行预处理
    image = preprocess(image)  # 此时image为增强对比度后的单通道灰度图

    # 对图像进行转换
    transformation = transforms.Compose([
        transforms.Resize((32, 32)),  # 图像分辨率调整为（32 * 32）
        transforms.RandomHorizontalFlip(),  # 对图像随机水平翻转
        transforms.RandomCrop(32),  # 对图像随机裁剪，第一个参数指定裁剪尺寸为32*32
        transforms.ToTensor(),  # 将图像转换成pytorch能够使用的格式并归一化至【0，1】
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))  # 将数据按通道进行标准化，让所有像素范围处于【-1，1】之间
    ])

    # 将opencv图像转换为PIL图像
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    image_tensor = transformation(image).float()
    image_tensor = image_tensor.unsqueeze_(0) # 额外添加一个批次维度

    inputs = Variable(image_tensor)
    output = model(inputs)  # 预测图片中的瑕疵所属类别
    defect_kind = output.data.numpy().argmax()
    return defect_kind  #返回预测类的索引


def detect(request):
    # 定义字典确定产品规格
    dic1 = {'L': "临时品种", 'X': "打样品种", 'C': "常规品种"}
    dic2 = {'N': "尼丝纺", 'T': "塔丝隆", 'P': "春亚纺", 'S': "桃皮绒",
            'J': "锦涤纺", 'R': "麂皮绒", 'D': "涤塔夫", 'Q': "其它品种"}
    dic3 = {'T': "平纹", 'W': "斜纹", 'B': "格子", 'S': "缎纹"}

    # 定义字典，其中键为类别标号，值为具体瑕疵名
    dic = {'0': "停车痕（松）", '1': "停车痕（紧）", '2': "其他", '3': "尽机",
           '4': "并纬", '5': "折返", '6': "擦伤", '7': "擦白", '8': "断纬",
           '9': "断经1", '10': "断经2", '11': "油污", '12': "浆斑",
           '13': "空织", '14': "糙纬", '15': "经条", '16': "缩纬",
           '17': "缺纬1", '18': "缺纬2", '19': "起机", '20': "错花", '21': "无瑕疵"}

    mediaPath = os.path.dirname(os.path.dirname(__file__))+'/media'

    images = CustomImage.objects.filter(IsDetect=False)

    for cur in images:
        global extension_ratio
        extension_ratio = 1
        specs = "未检测"
        batchNum = "未检测"
        defecttype = "未检测"


        imageName = cur.title[0:6]

        image_path = os.path.join(mediaPath, str(cur.file))
        image = cv_imread(image_path)  # 图像为三通道
        image = cv2.resize(image, None, fx=ratio, fy=ratio)

        # 1.去除黑色空区
        # 2.自适应对比度增强
        new_img, blocPos, cutOff = remove_enhance(image)  # 改动！！！
        height, width = new_img.shape[0], new_img.shape[1]

        # 3.同态滤波
        new_img = homofilter(new_img)
        new_img = cv2.normalize(new_img, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)

        # 4.伽马变换
        new_img = gammaTransform(1, 0.7, new_img)

        # 5.Gabor滤波
        dst = GaborFeature(new_img)

        # 6.求梯度
        gradX = cv2.Sobel(dst, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
        gradY = cv2.Sobel(dst, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)
        gradient = cv2.subtract(gradX, gradY)
        gradient = cv2.convertScaleAbs(gradient)

        # 7.全局自适应阈值二值化
        ret, th = cv2.threshold(gradient, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 8.填充死角暗区，减弱噪声影响
        xlen = int(80 * ratio)
        ylen = int(120 * ratio)
        points1 = [(0, 0), (0, height), (xlen, height), (xlen, 0)]  # 改动！！！
        points2 = [(0, height - ylen), (0, height), (width, height), (width, height - ylen)]  # 改动！！！
        points3 = [(width - xlen, 0), (width - xlen, height), (width, height), (width, 0)]  # 改动！！！
        color = [0, 0, 0]
        fill = cv2.fillPoly(th, [np.array(points1)], color)
        fill = cv2.fillPoly(fill, [np.array(points2)], color)
        fill = cv2.fillPoly(fill, [np.array(points3)], color)

        # 9.腐蚀与膨胀
        erosion = cv2.erode(fill, None, iterations=1)
        inflation = cv2.dilate(erosion, None, iterations=1)

        # 10.形态学填充
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
        result = cv2.morphologyEx(inflation, cv2.MORPH_CLOSE, kernel)

        # 11.选取最小外接矩形
        contours, hierarchy = cv2.findContours(result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print(contours)

        pt_x = 0  # 若输出的疵点坐标为（0，0）则表示图像中没有瑕疵 中心点
        pt_y = 0
        pt_height = 0
        pt_width = 0
        num = np.size(contours)
        print(num)
        hasDefect = False

        if num > 0:
            hasDefect = True
            # 取主要瑕疵区域
            c = sorted(contours, key=cv2.contourArea, reverse=True)[0]
            rect = cv2.minAreaRect(c)

            # 11.画出旋转角为0的外界矩形
            box = np.int0(cv2.boxPoints(rect))
            # 寻找横纵坐标最值
            Xs = [i[0] for i in box]
            Ys = [i[1] for i in box]
            x1 = int(max(min(Xs) - 40 * ratio, 0))  # 改动！！！
            x2 = int(min(max(Xs) + 40 * ratio, width))  # 改动！！！
            y1 = int(max(min(Ys) - 40 * ratio, 0))  # 改动！！！
            y2 = int(min(max(Ys) + 40 * ratio, height))  # 改动！！！
            pt_x = int((x1 + x2)/2)
            pt_y = int((y1 + y2)/2)
            pt_width = int(x2 - x1)
            pt_height = int(y2 - y1)
            # 判断是否被裁剪过
            if blocPos == -1:  # 有黑块且在左边     # 改动！！！
                pt_x += cutOff  # 改动！！！

        cur.HasDefect = hasDefect
        cur.focal_point_x = int(pt_x/ratio)
        cur.focal_point_y = int(pt_y/ratio)
        cur.focal_point_width = int(pt_width/ratio)
        cur.focal_point_height = int(pt_height/ratio)
        cur.ExtensionRatio = extension_ratio
        cur.IsDetect = True

        # 瑕疵分类
        if hasDefect == True:
            index = predict_defect(image_path)
            result = dic[str(index)]
            cur.DefectType = result
        cur.save()

        # 将该图片的信息存入统计详情表
        staticDetail = StaticsDetail.objects.all()
        try:
            scur = staticDetail.get(BatchNum=cur.BatchNum)

        except ObjectDoesNotExist:
            scur = StaticsDetail.objects.create(BatchNum=cur.BatchNum)
        scur.Specs = cur.Specs
        scur.ClothCode = cur.ClothCode
        scur.CountAll += 1
        if cur.HasDefect == True:
            scur.DefectCount += 1
            if cur.DefectType == "油污":
                scur.YW += 1
            elif cur.DefectType == "浆斑":
                scur.JB += 1
            elif cur.DefectType == "停车痕（紧）":
                scur.TCHJ += 1
            elif cur.DefectType == "停车痕（松）":
                scur.TCHS += 1
            elif cur.DefectType == "并纬":
                scur.BW += 1
            elif cur.DefectType == "擦白":
                scur.CB += 1
            elif cur.DefectType == "擦伤":
                scur.CS += 1
            elif cur.DefectType == "糙纬":
                scur.CW += 1
            elif cur.DefectType == "错花":
                scur.CH += 1
            elif cur.DefectType == "断经1":
                scur.DJ1 += 1
            elif cur.DefectType == "断经2":
                scur.DJ2 += 1
            elif cur.DefectType == "断纬":
                scur.DW += 1
            elif cur.DefectType == "尽机":
                scur.JJ += 1
            elif cur.DefectType == "经条":
                scur.JT += 1
            elif cur.DefectType == "空织":
                scur.KZ += 1
            elif cur.DefectType == "起机":
                scur.QJ += 1
            elif cur.DefectType == "缺纬1":
                scur.QW1 += 1
            elif cur.DefectType == "缺纬2":
                scur.QW2 += 1
            elif cur.DefectType == "缩纬":
                scur.SW += 1
            elif cur.DefectType == "折返":
                scur.ZF += 1
            elif cur.DefectType == "其他":
                scur.QT += 1
        scur.save()

    images = CustomImage.objects.all()

    batchNum = Document.objects.values('BatchNum').distinct()
    BatchNums = []
    for i in range(len(batchNum)):
        BatchNums.append(batchNum[i]['BatchNum'])
    spec = Document.objects.values('Specs').distinct()
    Specs = []
    for i in range(len(spec)):
        Specs.append(spec[i]['Specs'])

    messages.success(request, ("成功检测完毕!!"))
    collections = permission_policy.collections_user_has_any_permission_for(
        request.user, ['add', 'change']
    )

    return render(request, 'wagtailimages/images/index.html', {
            'images': images,
            'query_string': None,
            'is_searching': bool(None),
            'BatchNums': BatchNums,
            'Specs': Specs,

            'user_can_add': permission_policy.user_has_permission(request.user, 'add'),
        })