import os
import sys
import torch
import torch.autograd as autograd
import torch.nn.functional as F
import torch.nn as nn


def train(train_iter, dev_iter, test_iter, model, args):
    if args.cuda:
        model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    batch_num = len(train_iter)
    model.train()
    for epoch in range(1, args.epochs+1):
        steps = 0
        corrects_all = 0.0
        test_all = 0.0
        for batch in train_iter:
            feature, batch_len, target = batch.text[0], batch.text[1], batch.label
            # feature.data.t_(), target.data.sub_(1)  # batch first, index align
            if args.cuda:
                feature, target = feature.cuda(), target.cuda()
            optimizer.zero_grad()
            logit = model([feature, batch_len])
            # print logit
            loss_function = nn.CrossEntropyLoss()
            loss = loss_function(logit, target)
            # loss = F.cross_entropy(logit, target)
            loss.backward()
            optimizer.step()
            corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
            corrects_all += corrects
            test_all += batch_len.size()[0]
            steps += 1
            if steps % 5 == 0:
                accuracy = corrects_all/test_all * 100.0
                sys.stdout.write(
                    '\repoch {} Train[{}/{}] - loss: {:.6f}  acc: {:.4f}%({}/{})'.format(epoch,
                                                                             steps,
                                                                             batch_num,
                                                                             loss.data[0],
                                                                             accuracy,
                                                                             corrects_all,
                                                                             test_all))

            # if steps % args.save_interval == 0:
            #     if not os.path.isdir(args.save_dir): os.makedirs(args.save_dir)
            #     save_prefix = os.path.join(args.save_dir, 'snapshot')
            #     save_path = '{}_steps{}.pt'.format(save_prefix, steps)
            #     torch.save(model, save_path)
        eval(dev_iter, model, args, 'dev')
        eval(test_iter, model, args, 'test')

def eval(data_iter, model, args, mode):
    model.eval()
    corrects, avg_loss = 0.0, 0.0
    for batch in data_iter:
        feature, batch_len, target = batch.text[0], batch.text[1], batch.label
        # feature.data.t_(), target.data.sub_(1)  # batch first, index align
        if args.cuda:
            feature, target = feature.cuda(), target.cuda()

        logit = model([feature, batch_len])
        loss_function = nn.CrossEntropyLoss()
        loss = loss_function(logit, target)

        avg_loss += loss.data[0]
        corrects += (torch.max(logit, 1)
                     [1].view(target.size()).data == target.data).sum()
    size = len(data_iter.dataset)
    avg_loss = avg_loss/len(data_iter)
    accuracy = corrects/size * 100.0
    model.train()
    print('\n{} - loss: {:.6f}  acc: {:.4f}%({}/{}) \n'.format(mode, 
                                                            avg_loss,
                                                            accuracy,
                                                            corrects,
                                                            size))


def predict(text, model, text_field, label_feild):
    assert isinstance(text, str)
    model.eval()
    text = text_field.tokenize(text)
    text = text_field.preprocess(text)
    text = [[text_field.vocab.stoi[x] for x in text]]
    x = text_field.tensor_type(text)
    x = autograd.Variable(x, volatile=True)
    #print(x)
    output = model(x)
    _, predicted = torch.max(output, 1)
    return label_feild.vocab.itos[predicted.data[0][0]+1]