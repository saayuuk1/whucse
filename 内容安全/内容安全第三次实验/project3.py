from keras.layers import Activation, Dense, Dropout
from keras.models import Sequential
import librosa
import librosa.display
import numpy as np
import pandas as pd
import os
import seaborn as sns

# 对数据进行处理
from sklearn.preprocessing import LabelEncoder

# 模型结构
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.utils import np_utils
from sklearn.metrics import confusion_matrix, classification_report

# 数据预处理，将音频文件数据转换为mfcc特征数据保存
def data_preprocessing():
    #读取数据列表
    data = pd.read_csv('D:/Downloads/UrbanSound8K/metadata/UrbanSound8K.csv')

    # 读取wav文件函数
    def path_class(data, filename):
        excerpt = data[data['slice_file_name'] == filename]
        path_name = os.path.join('D:/Downloads/UrbanSound8K/audio', 'fold'+str(excerpt.fold.values[0]), filename)
        return path_name, excerpt['class'].values[0]

    # 读取wav声音文件，并提取mfcc特征，以及label标签，将其保存
    dataset = []
    for i in range(data.shape[0]):
        
        fullpath, class_id = path_class(data, data.slice_file_name[i])
        try:
            X, sample_rate = librosa.load(fullpath, res_type='kaiser_fast')
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)
        except Exception:
            print("Error encountered while parsing file: ", data.slice_file_name[i])
            mfccs,class_id = None, None
        
        dataset.append((mfccs, class_id))

    # 保存预处理后的数据集
    np.save("dataset",dataset,allow_pickle=True)

# 定义评价指标
def acc(y_test,prediction):
    ### PRINTING ACCURACY OF PREDICTION
    ### RECALL
    ### PRECISION
    ### CLASIFICATION REPORT
    ### CONFUSION MATRIX
    cm = confusion_matrix(y_test, prediction)
    recall = np.diag(cm) / np.sum(cm, axis = 1)
    precision = np.diag(cm) / np.sum(cm, axis = 0)
    
    print ('Recall:', recall)
    print ('Precision:', precision)
    print ('\n clasification report:\n', classification_report(y_test,prediction))
    print ('\n confussion matrix:\n',confusion_matrix(y_test, prediction))
    
    ax = sns.heatmap(confusion_matrix(y_test, prediction),linewidths= 0.5,cmap="YlGnBu")

# 判断预处理后的数据集是否存在
if not os.path.exists('dataset.npy'):
    data_preprocessing()

# 导入数据
data = pd.DataFrame(np.load("dataset.npy",allow_pickle= True))
data.columns = ['feature', 'label']

X = np.array(data.feature.tolist())
y = np.array(data.label.tolist())

# 数据分割
from sklearn.model_selection import train_test_split
X,val_x,y,val_y = train_test_split(X,y)

# 对标签进行one-hot处理
lb = LabelEncoder()
from keras.utils import np_utils
y = np_utils.to_categorical(lb.fit_transform(y))
val_y = np_utils.to_categorical(lb.fit_transform(val_y))

# 准备标签集
num_labels = y.shape[1]
nets = 5

model = [0] * nets

# 定义模型结构
for net in range(nets):
    model[net] = Sequential()


    model[net].add(Dense(512, input_shape=(40,)))
    model[net].add(Activation('relu'))
    model[net].add(Dropout(0.45))


    model[net].add(Dense(256))
    model[net].add(Activation('relu'))
    model[net].add(Dropout(0.45))


    model[net].add(Dense(num_labels))
    model[net].add(Activation('softmax'))



    model[net].compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='RMSprop')

# 训练网络
history = [0] * nets
epochs = 132
for j in range(nets):
    X_train2, X_val2, Y_train2, Y_val2 = X,val_x, y, val_y
    history[j] = model[j].fit(X,Y_train2, batch_size=256,
        epochs = epochs,   
        validation_data = (X_val2,Y_val2),  verbose=0)
    print("CNN {0:d}: Epochs={1:d}, Train accuracy={2:.5f}, Validation accuracy={3:.5f}".format(
        j+1,epochs,max(history[j].history['accuracy']),max(history[j].history['val_accuracy']) ))

# 查看评价指标以及混淆矩阵
results = np.zeros( (val_x.shape[0],10) ) 
for j in range(nets):
    results = results  + model[j].predict(val_x)
results = np.argmax(results,axis = 1)
val_y_n = np.argmax(val_y,axis =1)
acc(val_y_n,results)