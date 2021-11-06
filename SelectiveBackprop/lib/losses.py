import torch.nn.functional as F
import torch
import torch.nn as nn
from IPython import embed


def CrossEntropySquaredLoss(reduce=True):
    if reduce:
        def fn(outputs, labels):
            batch_size = outputs.size()[0]            # batch_size
            outputs = F.log_softmax(outputs, dim=1)   # compute the log of softmax values
            outputs = outputs[range(batch_size), labels] # pick the values corresponding to the labels
            cross_entropy_loss = -torch.mean(outputs)
            return cross_entropy_loss ** 2 / 10.
    else:
        def fn(outputs, labels):
            batch_size = outputs.size()[0]            # batch_size
            outputs = F.log_softmax(outputs, dim=1)   # compute the log of softmax values
            outputs = outputs[range(batch_size), labels] # pick the values corresponding to the labels
            cross_entropy_loss = - outputs
            return cross_entropy_loss ** 2 / 10.
    return fn

def CrossEntropyRegulatedLoss(reduce=True):
    if reduce:
        def fn(outputs, labels):
            batch_size = outputs.size()[0]            # batch_size
            outputs = F.log_softmax(outputs, dim=1)   # compute the log of softmax values
            class_outputs = outputs[range(batch_size), labels] # pick the values corresponding to the labels

            max_others = None
            for output, label in zip(outputs, labels):
                other_outputs = torch.cat((output[0:label], output[label+1:]))
                max_other = torch.max(other_outputs)
                if max_others is None:
                    max_others = max_other.unsqueeze(-1)
                else:
                    max_others = torch.cat((max_others, max_other.unsqueeze(-1)))
            cross_entropy_loss = -torch.mean(class_outputs + 0.05 * max_others)
            return cross_entropy_loss
    else:
        def fn(outputs, labels):
            batch_size = outputs.size()[0]            # batch_size
            outputs = F.log_softmax(outputs, dim=1)   # compute the log of softmax values
            class_outputs = outputs[range(batch_size), labels] # pick the values corresponding to the labels

            max_others = None
            for output, label in zip(outputs, labels):
                other_outputs = torch.cat((output[0:label], output[label+1:]))
                max_other = torch.max(other_outputs)
                if max_others is None:
                    max_others = max_other.unsqueeze(-1)
                else:
                    max_others = torch.cat((max_others, max_other.unsqueeze(-1)))
            cross_entropy_loss = -(class_outputs + 0.05 * max_others)
            return cross_entropy_loss
    return fn

def CrossEntropyRegulatedBoostedLoss(reduce=True):
    if reduce:
        def fn(outputs, labels):
            batch_size = outputs.size()[0]            # batch_size
            num_classes = outputs.size()[1]            # num classes
            outputs = F.softmax(outputs, dim=1)       # compute the softmax values
            class_outputs = outputs[range(batch_size), labels] # pick the values corresponding to the labels

            losses = None
            for output, class_prob, label in zip(outputs, class_outputs, labels):
                other_outputs = torch.cat((output[0:label], output[label+1:]))
                max_other_prob = torch.max(other_outputs)
                loss = - torch.log(class_prob) - torch.log(1 - (max_other_prob - (1 - class_prob) / (num_classes - 1)))
                cross_entropy = torch.log(class_prob)
                if losses is None:
                    losses = loss.unsqueeze(-1)
                else:
                    losses = torch.cat((losses, loss.unsqueeze(-1)))
            reduced_loss = torch.mean(losses)
            return reduced_loss
    else:
        def fn(outputs, labels):
            batch_size = outputs.size()[0]            # batch_size
            num_classes = outputs.size()[1]            # num classes
            outputs = F.softmax(outputs, dim=1)       # compute the softmax values
            class_outputs = outputs[range(batch_size), labels] # pick the values corresponding to the labels

            losses = None
            for output, class_prob, label in zip(outputs, class_outputs, labels):
                other_outputs = torch.cat((output[0:label], output[label+1:]))
                max_other_prob = torch.max(other_outputs)
                loss = - torch.log(class_prob) - torch.log(1 - (max_other_prob - (1 - class_prob) / (num_classes - 1)))
                if losses is None:
                    losses = loss.unsqueeze(-1)
                else:
                    losses = torch.cat((losses, loss.unsqueeze(-1)))
            return losses
    return fn

def CrossEntropyLoss(reduce=True):
    if reduce:
        def fn(outputs, labels):
            batch_size = outputs.size()[0]            # batch_size
            outputs = F.log_softmax(outputs, dim=1)   # compute the log of softmax values
            outputs = outputs[range(batch_size), labels] # pick the values corresponding to the labels
            cross_entropy_loss = -torch.mean(outputs)
            return cross_entropy_loss
    else:
        def fn(outputs, labels):
            batch_size = outputs.size()[0]            # batch_size
            outputs = F.log_softmax(outputs, dim=1)   # compute the log of softmax values
            outputs = outputs[range(batch_size), labels] # pick the values corresponding to the labels
            cross_entropy_loss = - outputs
            return cross_entropy_loss
    return fn

def CrossEntropyLossRicap(reduce=True):
    def fn(outputs, c_, w_):
        W_ = [w_[aa] for aa in w_.keys()]
        criterion = nn.CrossEntropyLoss(reduction = 'none')
        prob = F.softmax(outputs, dim=1)
        l_1 = criterion(outputs, c_[:,0])
        l_2 = criterion(outputs, c_[:,1])
        l_3 = criterion(outputs, c_[:,2])
        l_4 = criterion(outputs, c_[:,3])
        loss_all = W_[0]*l_1 + W_[1]*l_2 + W_[2]*l_3 + W_[3]*l_4
        loss = torch.mean(loss_all)
        if reduce:
            return loss
        else:
            return loss_all
    return fn

def CrossEntropyLossMixup(reduce=True):
    def fn(outputs, c_, w_):
        criterion = nn.CrossEntropyLoss(reduction = 'none')
        l_1 = criterion(outputs, c_[:,0])
        l_2 = criterion(outputs, c_[:,1])
        loss_all = w_[0]*l_1 + w_[1]*l_2
        loss = torch.mean(loss_all)
        if reduce:
            return loss
        else:
            return loss_all
    return fn


def MSELoss(reduce=True):
    if reduce:
        def fn(outputs, labels):
            batch_size = outputs.size()[0]
            num_classes = outputs.size()[1]
            outputs = F.softmax(outputs, dim=1)
            targets = torch.eye(num_classes)
            l2_dists = None
            for output, label in zip(outputs, labels):
                target = targets[int(label.item())]
                l2_dist = torch.dist(target, output.cpu())
                if l2_dists is None:
                    l2_dists = l2_dist.unsqueeze(-1)
                else:
                    l2_dists = torch.cat((l2_dists, l2_dist.unsqueeze(-1)))
            return torch.mean(l2_dists)
    else:
        def fn(outputs, labels):
            batch_size = outputs.size()[0]
            num_classes = outputs.size()[1]
            outputs = F.softmax(outputs, dim=1)
            targets = torch.eye(num_classes)
            l2_dists = None
            for output, label in zip(outputs, labels):
                target = targets[int(label.item())]
                l2_dist = torch.dist(target, output.cpu())
                if l2_dists is None:
                    l2_dists = l2_dist.unsqueeze(-1)
                else:
                    l2_dists = torch.cat((l2_dists, l2_dist.unsqueeze(-1)))
            return l2_dists
    return fn
