from torchtext import data
import torch
import torch.nn as nn
from torchtext import datasets
from torchtext.vocab import Vectors
import codecs

def load_datasets(path, text_field, label_field):
    labels = []
    texts = []
    for line in codecs.open(path, 'r', 'ISO-8859-1'):
        label, text = line.split('\t')
        labels.append(label)
        texts.append(text.strip().split(' '))
    fields = [('text', text_field), ('label', label_field)]
    examples = []
    for text, label in zip(texts, labels):
        examples.append(data.Example.fromlist([text, label], fields=fields))
    return data.Dataset(examples, fields)

class PretrainedVectors(object):
    """
    :param path:  word_vec file path
    """
    def __init__(self, path, unk_init=torch.Tensor.zero_):
        self.unk_init = unk_init
        self.load_vector(path)

    def __getitem__(self, token):
        if token in self.stoi:
            return self.vectors[self.stoi[token]]
        else:
            return self.unk_init(torch.Tensor(1, self.dim))

    def load_vector(self, path):
        print 'loading Embedding'
        self.stoi, self.vectors, self.dim = torch.load(path)

def loadFieldVector(text_field, pretrainedVectors):
    """
    :param text_field: The text Field
    :param pretrainedVectors: pretrained vectors from local
    :return: embed_tensor
    """
    vec_dim = pretrainedVectors.dim
    embed_tensor = torch.Tensor(len(text_field.vocab), vec_dim)
    for i, token in enumerate(text_field.vocab.itos):
        embed_tensor[i] = pretrainedVectors[token.strip()]
    return embed_tensor