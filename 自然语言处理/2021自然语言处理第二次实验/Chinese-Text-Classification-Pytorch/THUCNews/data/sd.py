import random
data = dict()
data['0'], data['1'], data['3'], data['4'] = [], [], [], []
with open(r'./train.txt', 'r', encoding='utf-8') as f:
    for line in f.readlines():
        if line.split('\t')[-1][0] in ['0', '1', '3', '4']:
            data[line.split('\t')[-1][0]].append(line.split('\t')[0])

with open(r'./new_train.txt', 'w', encoding='utf-8') as f1:
    for item in ['0', '1', '3', '4']:
        for line in data[item]:
            if item == '3':
                f1.write(line+'\t'+'2'+'\n')
            elif item == '4':
                f1.write(line+'\t'+'3'+'\n')
            else:
                f1.write(line+'\t'+item+'\n')

            
import random
data = dict()
data['0'], data['1'], data['3'], data['4'] = [], [], [], []
with open(r'./test.txt', 'r', encoding='utf-8') as f:
    for line in f.readlines():
        if line.split('\t')[-1][0] in ['0', '1', '3', '4']:
            data[line.split('\t')[-1][0]].append(line.split('\t')[0])

with open(r'./new_test.txt', 'w', encoding='utf-8') as f1:
    for item in ['0', '1', '3', '4']:
        for line in data[item]:
            if item == '3':
                f1.write(line+'\t'+'2'+'\n')
            elif item == '4':
                f1.write(line+'\t'+'3'+'\n')
            else:
                f1.write(line+'\t'+item+'\n')

import random
data = dict()
data['0'], data['1'], data['3'], data['4'] = [], [], [], []
with open(r'./dev.txt', 'r', encoding='utf-8') as f:
    for line in f.readlines():
        if line.split('\t')[-1][0] in ['0', '1', '3', '4']:
            data[line.split('\t')[-1][0]].append(line.split('\t')[0])

with open(r'./new_dev.txt', 'w', encoding='utf-8') as f1:
    for item in ['0', '1', '3', '4']:
        for line in data[item]:
            if item == '3':
                f1.write(line+'\t'+'2'+'\n')
            elif item == '4':
                f1.write(line+'\t'+'3'+'\n')
            else:
                f1.write(line+'\t'+item+'\n')