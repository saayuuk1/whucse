from gensim.models import word2vec
import gensim
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt

data = ['江苏', '南京', '成都', '四川', '湖北', '武汉', '河南', '郑州', '甘肃', '郑州', '湖南', '长沙', '陕西', '西安', '吉林', '长春',\
        '广东', '广州', '浙江', '杭州']

# 导入模型
path = r'.\word2vec.model'
wv_model = gensim.models.Word2Vec.load(path)

# 构造符合pca.fit_transform输入的二维数组
embeddings = []
for i in data:
    embeddings.append(wv_model[i])

# 构建PCA模型
pca = PCA(n_components=2)
results = pca.fit_transform(embeddings)

# 配置绘图选项以显示中文
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 绘图
sns.scatterplot(x=results[:, 0], y=results[:, 1])
for index, xy in enumerate(zip(results[:, 0], results[:, 1])):
    plt.annotate(data[index], xy=xy, xytext=(xy[0]+0.1, xy[1]+0.1))