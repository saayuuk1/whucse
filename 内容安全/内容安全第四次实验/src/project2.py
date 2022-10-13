from deepface import DeepFace
import cv2
import dlib

# 人脸检测器
detector = dlib.get_frontal_face_detector()
# 特征提取器
predictor = dlib.shape_predictor('../models/shape_predictor_68_face_landmarks.dat')

# 人脸情感检测
def analyze(img, i):
    img_path = '../data/video%d.jpg' % (i)
    # 将抽取的视频帧保存在本地
    cv2.imwrite(img_path, img)
    # 人脸属性检测
    ana = DeepFace.analyze(img_path=img_path, actions=['age', 'gender', 'race', 'emotion'], enforce_detection=False)
    age = ana['age']
    gender = ana['gender']
    race = ana['dominant_race']
    emotion = ana['dominant_emotion']

    # 人脸关键点检测
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dets = detector(gray, 1)
    for face in dets:
        # 寻找68个人脸关键点
        shape = predictor(img, face)
        # 绘制所有点
        for pt in shape.parts():
            cv2.circle(img, (pt.x, pt.y), 2, (0, 255, 0), 1)
        # 绘制性别检测结果
        cv2.putText(img, 'gender: %s' % (gender), (90, 400), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (35, 170, 242), 4)
        # 绘制种族检测结果
        cv2.putText(img, 'race: %s' % (race), (90, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (35, 170, 242), 4)
        # 绘制年龄检测结果
        cv2.putText(img, 'age: %s' % (age), (90, 500), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (35, 170, 242), 4)
        # 绘制情感检测结果
        cv2.putText(img, 'emotion: %s' % (emotion), (90, 550), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (35, 170, 242), 4)
        cv2.namedWindow('image', cv2.WINDOW_FREERATIO)
        # 展示图像
        cv2.imshow('image', img)
        cv2.waitKey(1)

if __name__ == '__main__':
    cap = cv2.VideoCapture('../data/video.mp4')
    i = 0
    while True:
        ret, img = cap.read()
        i += 1
        if ret == False:
            break
        # 每隔10帧检测一次人脸属性
        if i % 10 != 0:
            continue
        analyze(img, i)
    