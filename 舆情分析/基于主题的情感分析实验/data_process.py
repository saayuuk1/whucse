from torch.utils.data import Dataset, DataLoader
from torchtext.data.utils import get_tokenizer
from collections import Counter
from torch.nn.utils.rnn import pad_sequence
import linecache
import json
import numpy as np
import torch
import pickle as pkl

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device) #查看我们使用的设备，如果是cpu证明GPU没有开启，请自行检查原因
PAD, UNK = '<pad>', '<unk>' # 定义特殊token
# vocab = {PAD: 0, UNK: 1}
vocab = pkl.load(open('./emb/vocab.pkl', 'rb')) #加载字典

#定义自己的Dataset类，用于加载和处理数据
class MyDataset(Dataset):
	def __init__(self, path):
		self.tokenizer = get_tokenizer('basic_english') # 分词器

		self.text_list = []
		self.label_list = []
		self.aspect_list = []

		#读取数据并且处理数据
		with open(path, 'r') as f:
			self.file = json.load(f)
			for line in self.file:

				self.text_list.append(self.tokenizer(line['sentence']))
				self.aspect_list.append(self.tokenizer(line['aspect']))

				if line["sentiment"] == "negative":
					self.label_list.append(0)
				elif line["sentiment"] == "positive":
					self.label_list.append(1)
				elif line["sentiment"] == "neutral":
					self.label_list.append(2)
	#获取数据长度
	def __len__(self):
		return len(self.file)
	#按照index获取数据
	def __getitem__(self, index):
		return self.text_list[index], self.aspect_list[index], self.label_list[index]

#定义自己的Dataset类，用于加载和处理数据
class MyDataset_(Dataset):
	def __init__(self, data):
		self.tokenizer = get_tokenizer('basic_english') # 分词器

		self.text_list = []
		self.aspect_list = []

		#读取数据并且处理数据
		self.file = json.loads(data)
		for line in self.file:

			self.text_list.append(self.tokenizer(line['sentence']))
			self.aspect_list.append(self.tokenizer(line['aspect']))
	#获取数据长度
	def __len__(self):
		return len(self.file)
	#按照index获取数据
	def __getitem__(self, index):
		return self.text_list[index], self.aspect_list[index]


#用于DataLoader装载数据时进一步处理batch数据
def batch_process(batch):
	text_list, aspect_list, label_list = zip(*batch)
	text_list_ = []
	aspect_list_ = []

	#token转化成id
	for i in range(len(text_list)):
		text_list_.append(torch.tensor([vocab[token] if token in list(vocab.keys()) else vocab[UNK] for token in text_list[i]]))
		aspect_list_.append(torch.tensor([vocab[token] if token in list(vocab.keys()) else vocab[UNK] for token in aspect_list[i]]))


	text_list_ = pad_sequence(text_list_, batch_first=True, padding_value=0) #padding数据

	#将数据类型转化成tensor
	aspect_list_ = torch.tensor(aspect_list_, dtype=torch.long)
	label_list = torch.tensor(label_list, dtype=torch.long)

	return text_list_.to(device), aspect_list_.to(device), label_list.to(device)


#用于DataLoader装载数据时进一步处理batch数据
def batch_process_(batch):
	text_list, aspect_list = zip(*batch)
	text_list_ = []
	aspect_list_ = []

	#token转化成id
	for i in range(len(text_list)):
		text_list_.append(torch.tensor([vocab[token] if token in list(vocab.keys()) else vocab[UNK] for token in text_list[i]]))
		aspect_list_.append(torch.tensor([vocab[token] if token in list(vocab.keys()) else vocab[UNK] for token in aspect_list[i]]))


	text_list_ = pad_sequence(text_list_, batch_first=True, padding_value=0) #padding数据

	#将数据类型转化成tensor
	aspect_list_ = torch.tensor(aspect_list_, dtype=torch.long)

	return text_list_.to(device), aspect_list_.to(device)










