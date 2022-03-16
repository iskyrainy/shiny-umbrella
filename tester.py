import cv2
import faceRecognition as fr

# 该模块获取存储在磁盘中的图像并进行人脸识别
test_img = cv2.imread('TestImages/brother.jpg')  # test_img path
faces_detected, gray_img = fr.faceDetection(test_img)
print("faces_detected:", faces_detected)

faces, faceID = fr.labels_for_training_data('trainingImages')
face_recognizer = fr.train_classifier(faces, faceID)
face_recognizer.write('trainingData.yml')

face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('trainingData.yml')  # 为后续运行加载训练数据

name = {0: "Priyanka", 1: "Kangana", 2: "whiteGirl", 3: "brother", 4: "lakeGirl", 5: "redGirl", 6: "Tshirt",
        7: "sitGirl", 8: "kid", 9: "look", 10: "flower"}  # 创建包含每个标签名称的字典

for face in faces_detected:
    (x, y, w, h) = face
    roi_gray = gray_img[y:y + h, x:x + h]
    label, confidence = face_recognizer.predict(roi_gray)
    print("confidence:", confidence)
    print("label:", label)
    fr.draw_rect(test_img, face)
    predicted_name = name[label]
    if confidence > 37:  # 置信度 37
        continue
    fr.put_text(test_img, predicted_name, x, y)

resized_img = cv2.resize(test_img, (1000, 1000))
cv2.imshow("face dtecetion tutorial", resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
