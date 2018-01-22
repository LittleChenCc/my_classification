import os
import argparse
import datetime
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torchtext.data as data
import torchtext.datasets as datasets
import model
import train
from loader import *
parser = argparse.ArgumentParser(description='LSTM text classificer')
# learning
parser.add_argument('-lr', type=float, default=0.001, help='initial learning rate [default: 0.001]')
parser.add_argument('-epochs', type=int, default=256, help='number of epochs for train [default: 256]')
parser.add_argument('-batch-size', type=int, default=64, help='batch size for training [default: 64]')
parser.add_argument('-log-interval', type=int, default=1,
                    help='how many steps to wait before logging training status [default: 1]')
parser.add_argument('-test-interval', type=int, default=100,
                    help='how many steps to wait before testing [default: 100]')
parser.add_argument('-save-interval', type=int, default=500, help='how many steps to wait before saving [default:500]')
parser.add_argument('-save-dir', type=str, default='model_dir', help='where to save the snapshot')
# data
parser.add_argument('-shuffle', action='store_true', default=False, help='shuffle the data every epoch')
# model
parser.add_argument('-dropout', type=float, default=0.5, help='the probability for dropout [default: 0.5]')
parser.add_argument('-max-norm', type=float, default=3.0, help='l2 constraint of parameters [default: 3.0]')
parser.add_argument('-embed-dim', type=int, default=100, help='number of embedding dimension [default: 100]')
parser.add_argument('-hidden-size', type=int, default=100, help='number of hidden state dimension [default: 100]')
parser.add_argument('-kernel-num', type=int, default=100, help='number of each kind of kernel')
parser.add_argument('-kernel-sizes', type=str, default='3,4,5',
                    help='comma-separated kernel size to use for convolution')
parser.add_argument('-static', action='store_true', default=False, help='fix the embedding')
# device
parser.add_argument('-device', type=int, default=-1, help='device to use for iterate data, -1 mean cpu [default: -1]')
parser.add_argument('-no-cuda', action='store_true', default=False, help='disable the gpu')
# option
# parser.add_argument('-snapshot', type=str, default=None, help='filename of model snapshot [default: None]')
parser.add_argument('-snapshot', type=str, default="./snapshot/2017-06-18_16-39-47/snapshot_steps16000.pt",
                    help='filename of model snapshot [default: None]')
parser.add_argument('-no-pretrain', action='store_false', default=True, help='load pretrained vector.')

# parser.add_argument('-predict', type=str, default="Hello my dear , I love you so much .", help='predict the sentence given')
parser.add_argument('-predict', type=str, default=None, help='predict the sentence given')
parser.add_argument('-test', action='store_true', default=True, help='train or test')
args = parser.parse_args()


# load data
print("\nLoading data...")
text_field = data.Field(lower=True, batch_first=True, include_lengths=True)
label_field = data.Field(sequential=False, batch_first=True)

train_data = load_datasets('/home/jkchen/mydata/AsinPN1500/all/all.train', text_field, label_field)
dev_data = load_datasets('/home/jkchen/mydata/AsinPN1500/all/all.dev', text_field, label_field)
test_data = load_datasets('/home/jkchen/mydata/AsinPN1500/all/all.test', text_field, label_field)

text_field.build_vocab(train_data, dev_data)
label_field.build_vocab(train_data)
print 'vocab size {}'.format(len(text_field.vocab))
train_iter, dev_iter, test_iter = data.Iterator.splits((train_data, dev_data, test_data),
                                                             batch_sizes=(args.batch_size, args.batch_size, args.batch_size),
                                                             device=-1,
                                                             repeat=False,
                                                             sort_within_batch=True, 
                                                             sort_key=lambda x: len(x.text))
args.vocab_size = len(text_field.vocab)
args.label_size = len(label_field.vocab)
# args.cuda = args.no_cuda and torch.cuda.is_available();
# del args.no_cuda
# args.kernel_sizes = [int(k) for k in args.kernel_sizes.split(',')]
# args.save_dir = os.path.join(args.save_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

# print("\nParameters:")
# for attr, value in sorted(args.__dict__.items()):
#     print("\t{}={}".format(attr.upper(), value))
args.cuda = True
trainmodel = model.LSTMClassifier(args)
if args.no_pretrain:
    pretrainVector = PretrainedVectors('/home/jkchen/mydata/emb/glove.6B.{}d.txt.pt'.format(args.embed_dim))
    trainmodel.embedding.weight.data.copy_(loadFieldVector(text_field, pretrainVector))
    print 'finish load embedding.'
# print trainmodel.lstm[1]
# print F.cosine_similarity(trainmodel.embedding(Variable(torch.LongTensor([text_field.vocab.stoi['2']]))),
#                         trainmodel.embedding(Variable(torch.LongTensor([text_field.vocab.stoi['1']]))))

for name, param in trainmodel.named_parameters():
    print name, param.size()

train.train(train_iter, dev_iter, test_iter, trainmodel, args)

# model
# if args.snapshot is None:
#     cnn = model.LSTMClassifier(args)
# else:
#     print('\nLoading model from [%s]...' % args.snapshot)
#     try:
#         cnn = torch.load(args.snapshot)
#     except:
#         print("Sorry, This snapshot doesn't exist.");
#         exit()
#
# # train or predict
# if args.predict is not None:
#     label = train.predict(args.predict, cnn, text_field, label_field)
#     print('\n[Text]  {}[Label] {}\n'.format(args.predict, label))
# elif args.test:
#     try:
#         train.eval(test_iter, cnn, args)
#     except Exception as e:
#         print("\nSorry. The test dataset doesn't  exist.\n")
# else:
#     print()
#     train.train(train_iter, dev_iter, cnn, args)