# 设置训练参数
class TrainingConfig(object):
    batch_size = 64
    # 学习速率
    lr = 0.001
    epoches = 20
    print_step = 5
    emb_size = 100  # 词向量的维数
    pretrained_emb = './ResumeNER/pretrained_word_emb/word2vec.txt' # 词向量路径

class LSTMConfig(object):

    hidden_size = 100 # lstm隐向量的维数


class CNNConfig(object):
    num_layer = 3 # CNN层数
    hidden_size = 100  # CNN输出维度


