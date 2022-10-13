from skimage.feature import hog
import joblib, glob, os, cv2

from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
import numpy as np 

from imutils.object_detection import non_max_suppression
from skimage import color
from skimage.transform import pyramid_gaussian

pos_im_path = '../PersonDetection/DATAIMAGE/positive/'
neg_im_path = '../PersonDetection/DATAIMAGE/negative/'
model_path = '../PersonDetection/models/models1.dat'
predict_path = '../PersonDetection/test/test.jpg'

#滑动窗口
def sliding_window(image, window_size, step_size):
    for y in range(0, image.shape[0], step_size[1]):
        for x in range(0, image.shape[1], step_size[0]):
            yield (x, y, image[y: y + window_size[1], x: x + window_size[0]])

def hog_train():
    #标签为1表示POS，标签为0表示NEG
    data = []
    labels = []
    train_data = []
    train_labels = []
    test_data = []
    test_labels = []

    #读取POS类数据集
    for filename in glob.glob(os.path.join(pos_im_path, "*.png")):
        #读取图像
        img = cv2.imread(filename, 0)
        #统一图像尺寸
        img = cv2.resize(img, (64, 128))
        #提取HOG特征
        fd = hog(img, orientations=9, pixels_per_cell=(8, 8), visualize=False, cells_per_block=(3, 3))
        #保存特征信息
        data.append(fd)
        labels.append(1)

    #读取NEG类数据集
    for filename in glob.glob(os.path.join(neg_im_path, "*.jpg")):
        #读取图像
        img = cv2.imread(filename, 0)
        #统一图像尺寸
        img = cv2.resize(img, (64, 128))
        #提取HOG特征
        fd = hog(img, orientations=9, pixels_per_cell=(8, 8), visualize=False, cells_per_block=(3, 3))
        #保存特征信息
        data.append(fd)
        labels.append(0)
    data = np.float32(data)
    labels = np.array(labels)

    #将数据集中的70%分为训练集，30%分为测试集
    num = int(len(data) * 0.7)
    train_data = data[:num]
    train_labels = labels[:num]
    test_data = data[num:]
    test_labels = labels[num:]

    #使用SVM\KNN模型
    model = LinearSVC()
    #model = KNeighborsClassifier(n_neighbors=5)
    model.fit(train_data, train_labels)
    joblib.dump(model,  model_path)

    #model = joblib.load(model_path)

    #在测试集上计算模型准确率
    predicted = model.predict(test_data)
    mask = predicted == test_labels
    correct = np.count_nonzero(mask)
    result = (float(correct) / len(test_labels)) * 100
    print('测试集准确率：%f%%' % (result))

def hog_predict():
    image = cv2.imread(predict_path)
    image = cv2.resize(image,(400,256))
    size = (64,128)
    step_size = (9,9)
    downscale = 1.25
    #检测结果列表
    detections = []
    scale = 0
    #载入模型
    model = joblib.load(model_path)
    for im_scaled in pyramid_gaussian(image, downscale = downscale):
        #使用滑动窗口对每个窗口计算HOG特征值
        if im_scaled.shape[0] < size[1] or im_scaled.shape[1] < size[0]:
            break
        for (x, y, window) in sliding_window(im_scaled, size, step_size):
            if window.shape[0] != size[1] or window.shape[1] != size[0]:
                continue
            window = color.rgb2gray(window)
                
            fd=hog(window, orientations=9,pixels_per_cell=(8,8),visualize=False,cells_per_block=(3,3))
            fd = fd.reshape(1, -1)
            pred = model.predict(fd)
            #模型判断该窗口是否检测到行人
            if pred == 1:
                    
                if model.decision_function(fd) > 0.5:
                    detections.append((int(x * (downscale**scale)), int(y * (downscale**scale)), model.decision_function(fd), 
                    int(size[0] * (downscale**scale)),
                    int(size[1] * (downscale**scale))))
    
        scale += 1
    clone = image.copy()
    rects = np.array([[x, y, x + w, y + h] for (x, y, _, w, h) in detections])
    sc = [score[0] for (x, y, score, w, h) in detections]
    sc = np.array(sc)
    #非极大值抑制算法保证保证图像中每个对象只检测一次
    pick = non_max_suppression(rects, probs = sc, overlapThresh = 0.3)
    #在图像中绘制模型的预测结果
    for(x1, y1, x2, y2) in pick:
        cv2.rectangle(clone, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(clone,'Person',(x1-2,y1-2),1,0.75,(121,12,34),1)
    cv2.imshow('Person Detection',clone)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    #选择功能
    ch = input('Please choose a number: \n1. Train and Test\n2. Predict\n')
    if int(ch) == 1:
        hog_train()
    elif int(ch) == 2:
        hog_predict()
    else:
        print('Number invalid')