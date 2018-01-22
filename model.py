import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from loader import *

class LSTMClassifier(nn.Module):
    def __init__(self, args):
        super(LSTMClassifier, self).__init__()
        self.args = args
        self.hidden_size = args.hidden_size
        self.dropout = nn.Dropout(0.2)
        self.embedding = nn.Embedding(self.args.vocab_size, self.args.embed_dim)
        # self.lstm = nn.ModuleList()
        self.lstm_1 = nn.LSTM(args.embed_dim, self.hidden_size, 1, batch_first=True)
        # self.lstm_2 = nn.LSTM(self.hidden_size, self.hidden_size, 1, batch_first=True)
        # self.lstm.append(nn.LSTM(args.embed_dim, self.hidden_size, 1, batch_first=True))
        # self.lstm.append(nn.LSTM(self.hidden_size, self.hidden_size, 1, batch_first=True))
        self.fc = nn.Linear(args.hidden_size, self.args.label_size)

    def makeMask(self, outputs, seq_len):
        seq_mask = torch.zeros(outputs.size())
        for i, _seq_len in enumerate(seq_len):
            seq_mask[i][0:int(_seq_len)] = 1
        return Variable(seq_mask).cuda()

    def forward(self, x):
        # x = self.dropout(x)
        x, seq_len = x
        # h0, c0 = (Variable(torch.zeros(1, x.size()[0], self.hidden_size)).cuda(),
        #         Variable(torch.zeros(1, x.size()[0], self.hidden_size)).cuda())
        # h1, c1 = (Variable(torch.zeros(1, x.size()[0], self.hidden_size)).cuda(),
        #         Variable(torch.zeros(1, x.size()[0], self.hidden_size)).cuda())
        # print h0
        emb_input = self.embedding(x)
        droped_input = self.dropout(emb_input)
        packed_input = nn.utils.rnn.pack_padded_sequence(droped_input, list(seq_len.int()), batch_first=True)
        outputs, hidden = self.lstm_1(packed_input)
        # outputs, hidden = self.lstm_2(outputs)
        # seq_mask = self.makeMask(outputs, seq_len)
        # masked_outputs = seq_mask * outputs
        masked_outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        # print masked_outputs
        masked_sum =  torch.sum(masked_outputs, 1) 
        outputs = masked_sum / Variable(seq_len.unsqueeze(-1).expand_as(masked_sum).float()).cuda()
        logit = self.fc(outputs)
        return logit
