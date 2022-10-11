from data_process import *
from model import *
import os
import torch.nn as nn

#训练函数
def train(epochs):
	loss_ls = []

	model.train() #模型设置成训练模式
	for epoch in range(epochs): #训练epochs轮
		loss_sum = 0  #记录每轮loss
		for batch in train_iter:
			input_, aspect, label = batch
			optimizer.zero_grad() #每次迭代前设置grad为0

			#不同的模型输入不同，请同学们看model.py文件
			# output = model(input_)
			output = model(input_, aspect)
			
			loss = criterion(output, label) #计算loss
			loss.backward() #反向传播
			optimizer.step() #更新模型参数
			loss_sum += loss.item() #累积loss
		loss_ls.append(loss_sum / len(train_iter))
		print('epoch: ', epoch, 'loss:', loss_sum / len(train_iter))
	
	print(loss_ls)

	test_acc = evaluate() #模型训练完后进行测试
	print('test_acc:', test_acc)

#测试函数
def evaluate():
	model.eval()
	total_acc, total_count = 0, 0
	loss_sum = 0

	with torch.no_grad(): #测试时不计算梯度
		for batch in test_iter:
			input_, aspect, label = batch

			# predicted_label = model(input_)
			predicted_label = model(input_, aspect)
			
			loss = criterion(predicted_label, label) #计算loss
			total_acc += (predicted_label.argmax(1) == label).sum().item() #累计正确预测数
			total_count += label.size(0) #累积总数
			loss_sum += loss.item() #累积loss
		print('test_loss:', loss_sum / len(test_iter))

	return total_acc/total_count

# 预测函数
def predict():
	while True:
		# 用户输入预测数据
		print('Please input a sentence: ', end='')
		sentence = input()
		aspect = 'default'
		print('Please input a aspect: ', end='')
		aspect = input()
		data = '[{"sentence": "%s", "aspect": "%s"}]' % (sentence, aspect)
		
		# 处理输入
		pred_dataset = MyDataset_(data)
		pred_iter = DataLoader(pred_dataset, batch_size=1, shuffle=False, collate_fn=batch_process_)

		model.eval()
		
		with torch.no_grad():
			for batch in pred_iter:
				input_, aspect = batch

				# predicted_label = model(input_)
				predicted_label = model(input_, aspect)

				print(predicted_label)
				predicted_label = predicted_label.argmax(1)
				sentiment_ls = ['negative', 'positive', 'neutral']
				print('The sentiment of this sentence is: %s' % (sentiment_ls[predicted_label]))

		print(' ')

TORCH_SEED = 21 #随机数种子
os.environ["CUDA_VISIBLE_DEVICES"] = '0' #设置模型在几号GPU上跑
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  #设置device

# 设置随机数种子，保证结果一致
os.environ['PYTHONHASHSEED'] = str(TORCH_SEED)
torch.manual_seed(TORCH_SEED)
torch.cuda.manual_seed_all(TORCH_SEED)
np.random.seed(TORCH_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

#创建数据集
train_dataset = MyDataset('./data/acsa_train.json')
test_dataset = MyDataset('./data/acsa_test.json')
train_iter = DataLoader(train_dataset, batch_size=25, shuffle=True, collate_fn=batch_process)
test_iter = DataLoader(test_dataset, batch_size=25, shuffle=False, collate_fn=batch_process)

# 加载我们的Embedding矩阵
embedding = torch.tensor(np.load('./emb/my_embeddings.npz')['embeddings'], dtype=torch.float)

#定义模型
# model = LSTM_Network(embedding).to(device)
# model = AELSTM_Network(embedding).to(device)
model = ATAELSTM_Network(embedding).to(device)

#定义loss函数、优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adagrad(model.parameters(), lr=0.01, weight_decay=0.001)

#开始训练
train(40)
print(' ')

path = 'models/ATAE-BiLSTM.pth'

# 保存模型
torch.save(model.state_dict(), path)

# 加载模型
# model.load_state_dict(torch.load(path))

# 打印当前用于预测用户输入的模型名
print('Model: ' + path.split('/')[-1][:-4])

predict()







