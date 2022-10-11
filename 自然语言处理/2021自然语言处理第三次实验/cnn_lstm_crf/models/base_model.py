import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from .config import TrainingConfig, LSTMConfig, CNNConfig

class CNNmodel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layer=3, dropout=0.2):
        super(CNNmodel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layer = num_layer

        self.cnn_layer0 = nn.Conv1d(self.input_dim, self.hidden_dim, kernel_size=1, padding=0)
        self.cnn_layers = nn.ModuleList(nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=9, padding=4) for i in range(self.num_layer-1))
        self.drop = nn.Dropout(dropout)

    def forward(self, input_feature):
        batch_size = input_feature.size(0)
        seq_len = input_feature.size(1)

        input_feature = input_feature.transpose(2, 1).contiguous()
        cnn_output = self.cnn_layer0(input_feature)  #(b,h,l)
        cnn_output = self.drop(cnn_output)
        cnn_output = torch.tanh(cnn_output)

        for layer in range(self.num_layer-1):
            cnn_output = self.cnn_layers[layer](cnn_output)
            cnn_output = self.drop(cnn_output)
            cnn_output = torch.tanh(cnn_output)

        cnn_output = cnn_output.transpose(2, 1).contiguous()
        return cnn_output


class BaseModel(nn.Module):
    def __init__(self, word_emb, vocab_size, emb_size, hidden_size, out_size, opt):
        """初始化参数：
            vocab_size:字典的大小
            emb_size:词向量的维数
            hidden_size：隐向量的维数
            out_size:标注的种类
        """
        super(BaseModel, self).__init__()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.opt = opt
        if opt.use_w2v:
            self.embedding = nn.Embedding.from_pretrained(word_emb, freeze=False)
        else:
            self.embedding = nn.Embedding(vocab_size, emb_size)

        self.bilstm = nn.LSTM(emb_size, hidden_size,
                              batch_first=True,
                              bidirectional=True)

        self.cnn = CNNmodel(emb_size, hidden_size, num_layer=CNNConfig.num_layer).to(self.device)
        if opt.model == 'cnn':
            self.lin = nn.Linear(hidden_size, out_size)
        else:
            self.lin = nn.Linear(2 * hidden_size, out_size)

    def forward(self, sents_tensor, lengths):
        emb = self.embedding(sents_tensor)  # [B, L, emb_size]
        if self.opt.model == 'cnn':
            output = self.cnn(emb)
        else:
            packed = pack_padded_sequence(emb, lengths, batch_first=True)
            rnn_out, _ = self.bilstm(packed)
            # rnn_out:[B, L, hidden_size*2]
            output, _ = pad_packed_sequence(rnn_out, batch_first=True)
        scores = self.lin(output)  # [B, L, out_size]

        return scores

    def test(self, sents_tensor, lengths, _):
        """第三个参数不会用到，加它是为了与BiLSTM_CRF保持同样的接口"""
        logits = self.forward(sents_tensor, lengths)  # [B, L, out_size]
        _, batch_tagids = torch.max(logits, dim=2)

        return batch_tagids
