import matplotlib.pyplot as plt
import cv2 as cv

def comparehist(method):
    # 存储颜色直方图
    hists = []
    # 存储图像
    images = []
    # 本地加载的图像
    filenames = ['c1.jpg', 'c2.jpg', 'c3.jpg', 'c4.jpg']
    # 提取每个图像的颜色直方图
    for i, fileName in enumerate(filenames):
        # 读取图像
        image = cv.imread(fileName)
        # 转换成RGB
        images.append(cv.cvtColor(image, cv.COLOR_BGR2RGB))
        # 提取颜色直方图
        hist = cv.calcHist([image], [0, 1, 2], None, [16, 16, 16], [0, 256, 0, 256, 0, 256])
        # 归一均衡化
        hists.append(cv.normalize(hist, hist).flatten())

    results = []
    # 将源直方图与目标直方图进行比较
    for hist in hists:
        # 以第一张图片作为源图
        d = cv.compareHist(hists[0], hist, method)
        if method == cv.HISTCMP_BHATTACHARYYA:
            results.append(1 - (d ** 2))
        else:
            results.append(d)
    
    # 绘图
    fig = plt.figure('Intersect')
    fig.suptitle('Intersect', y=0.8, fontsize=16)
    for i, result in enumerate(results):
        ax = fig.add_subplot(1, len(images), i+1)
        ax.set_title('{:.6f}'.format(result))
        plt.imshow(images[i])
        plt.axis('off')
    plt.show()

# 相关 卡方 相交 巴氏距离
methods = [cv.HISTCMP_CORREL, cv.HISTCMP_CHISQR, cv.HISTCMP_INTERSECT, cv.HISTCMP_BHATTACHARYYA]
comparehist(methods[2])