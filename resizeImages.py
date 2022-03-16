import cv2
import os

# 此模块将图像从给定目录调整为 100*100 像素并将所有图像写入给定目录
count = 0

for path, subdirnames, filenames in os.walk("trainingImages"):

    for filename in filenames:
        if filename.startswith("."):
            print("Skipping File:", filename)  # 跳过以 . 开头的文件
            continue
        img_path = os.path.join(path, filename)  # 获取图片路径
        print("img_path", img_path)
        id = os.path.basename(path)  # 获取子目录名称
        img = cv2.imread(img_path)
        if img is None:
            print("Image not loaded properly")
            continue
        resized_image = cv2.resize(img, (100, 100))
        new_path = "resizedTrainingImages" + "/" + str(id)
        print("desired path is",
              os.path.join(new_path, "frame%d.jpg" % count))  # 将所有图像写入 resizedTrainingImages/id 目录
        cv2.imwrite(os.path.join(new_path, "frame%d.jpg" % count), resized_image)
        count += 1
