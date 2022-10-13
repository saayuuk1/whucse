import cv2
import numpy as np

#待检测图像路径
image_path = '../yolov3/test.jpg'
#预训练模型路径
config_path = '../yolov3/yolov3.cfg'
weights_path = '../yolov3/yolov3.weights'
classes_path = '../yolov3/yolov3.txt'

def get_output_layers(net):
    
    layer_names = net.getLayerNames()
    
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers


def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):

    label = str(classes[class_id])

    color = COLORS[class_id]

    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)

    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

#读取图像
image = cv2.imread(image_path)

Width = image.shape[1]
Height = image.shape[0]
scale = 0.00392

#获取所有类别
classes = None

with open(classes_path, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

#为不同类别随机生成不同颜色
COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

#构建预训练模型
net = cv2.dnn.readNet(weights_path, config_path)

blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)

net.setInput(blob)

#从网络中得到预测数据
outs = net.forward(get_output_layers(net))

class_ids = []
confidences = []
boxes = []
conf_threshold = 0.5
nms_threshold = 0.4

#对于每个输出层的检测数据，获取置信度、类的id、边界框参数，同时过滤置信度小于0.5的检测数据
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            center_x = int(detection[0] * Width)
            center_y = int(detection[1] * Height)
            w = int(detection[2] * Width)
            h = int(detection[3] * Height)
            x = center_x - w / 2
            y = center_y - h / 2
            class_ids.append(class_id)
            confidences.append(float(confidence))
            boxes.append([x, y, w, h])

#根据非极大值抑制进行筛选
indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

#绘制边界框
for i in indices:
    #i = i[0]
    box = boxes[i]
    x = box[0]
    y = box[1]
    w = box[2]
    h = box[3]
    draw_prediction(image, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))

#显示拥有边界框的输出图像
cv2.imshow("object detection", image)
cv2.waitKey()
cv2.destroyAllWindows()
