import cv2
import os
import numpy as np
# 该模块包含在 tester.py 文件中调用的所有常用函数

# 给定下面的图像，函数返回检测到的人脸的矩形以及灰度图像
def faceDetection(test_img):
    gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)  # 将彩色图像转换为灰度
    face_haar_cascade = cv2.CascadeClassifier('HaarCascade/haarcascade_frontalface_default.xml')  # 加载 haar classifier
    faces = face_haar_cascade.detectMultiScale(gray_img, scaleFactor=1.32,
                                               minNeighbors=5)  # detectMultiScale 返回矩形

    return faces, gray_img


# 给定下面的目录函数返回部分 gray_img 是 face 及其标签/ID
def labels_for_training_data(directory):
    faces = []
    faceID = []

    for path, subdirnames, filenames in os.walk(directory):
        for filename in filenames:
            if filename.startswith("."):
                print("Skipping system file")  # 跳过以 . 开头的文件
                continue

            id = os.path.basename(path)  # 获取子目录名称
            img_path = os.path.join(path, filename)  # 获取图片路径
            print("img_path:", img_path)
            print("id:", id)
            test_img = cv2.imread(img_path)  # 加载每张图片
            if test_img is None:
                print("Image not loaded properly")
                continue
            faces_rect, gray_img = faceDetection(
                test_img)  # 调用 faceDetection 函数以返回在特定图像中检测到的人脸
            if len(faces_rect) != 1:
                continue  # 假设只有单人图像被馈送到classifier
            (x, y, w, h) = faces_rect[0]
            roi_gray = gray_img[y:y + w, x:x + h]  # 裁剪区域，即灰度图像中的面部区域
            faces.append(roi_gray)
            faceID.append(int(id))
    return faces, faceID


# 下面的函数训练 haar classifier，并将前一个函数返回的faces/face ID 作为其参数
def train_classifier(faces, faceID):
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.train(faces, np.array(faceID))
    return face_recognizer


# 下面的函数在图像中检测到的人脸周围绘制边界框
def draw_rect(test_img, face):
    (x, y, w, h) = face
    cv2.rectangle(test_img, (x, y), (x + w, y + h), (255, 0, 0), thickness=5)


# 下面的函数为检测到的标签写入人名
def put_text(test_img, text, x, y):
    cv2.putText(test_img, text, (x, y), cv2.FONT_HERSHEY_DUPLEX, 2, (255, 0, 0), 4)
