import cv2
import dlib

# 人脸检测器
detector = dlib.get_frontal_face_detector()
# 特征提取器
predictor = dlib.shape_predictor('../models/shape_predictor_68_face_landmarks.dat')

# 人脸关键点检测
def landmarks(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dets = detector(gray, 1)
    for face in dets:
        # 寻找68个人脸关键点
        shape = predictor(img, face)
        # 绘制所有点
        for pt in shape.parts():
            cv2.circle(img, (pt.x, pt.y), 2, (0, 255, 0), 1)
        cv2.namedWindow('image', cv2.WINDOW_FREERATIO)
        cv2.imshow('image', img)
        cv2.waitKey(1)

# 人脸识别
def recognition(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dets = detector(gray, 1)
    for face in dets:
        left = face.left()
        right = face.right()
        top = face.top()
        bottom = face.bottom()
        # 绘制人脸识别结果边框
        cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.namedWindow('image', cv2.WINDOW_FREERATIO)
        cv2.imshow('image', img)
        cv2.waitKey(1)

if __name__ == '__main__':
    cap = cv2.VideoCapture('../data/video.mp4')
    while True:
        ret, img = cap.read()
        if ret == False:
            break
        landmarks(img)
        #recognition(img)

    cap.release()
    cv2.destroyAllWindows()