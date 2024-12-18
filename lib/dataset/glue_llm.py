import csv
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
import random
from datasets import load_dataset, load_from_disk
import torch.nn.functional as F
from lib.model.BCF import Tokenizer, BertClassificationModel
from tqdm import tqdm
import os
import numpy as np


class DataIOGLUE(object):
    def __init__(self, name='sst2'):
        self.name = name
        self.dic_keys = []
        self.train_word, self.train_label, \
        self.dev_word, self.dev_label, \
        self.test_word, \
        self.test_label = self.read_train_dev_test(name)
        self.num_classes = self.get_classes()

    def get_classes(self):
        classes = []
        for item in self.train_label:
            if item in classes:
                continue
            else:
                classes.append(item)
        return len(classes)

    def _load_dataset(self, a, name):
        return load_dataset(a, name)

    def read_train_dev_test(self, nam):
        if nam == 'mnli':
            dataset = self._load_dataset('glue', 'mnli')
            self.dic_keys = dataset['train'].features.keys()
            train_word, train_label = self.get_data_from_dataset(dataset, 'train'), dataset['train']['label']
            dev_word, dev_label = self.get_data_from_dataset(dataset, 'validation_matched'), \
                                  dataset['validation_matched']['label']
            test_word, test_label = self.get_data_from_dataset(dataset, 'test_matched'), \
                                    dataset['test_matched']['label']
            return train_word, train_label, dev_word, dev_label, test_word, test_label
        elif nam == 'mnli_mismatched':
            dataset = self._load_dataset('glue', 'mnli')
            self.dic_keys = dataset['train'].features.keys()
            train_word, train_label = self.get_data_from_dataset(dataset, 'train'), dataset['train']['label']
            dev_word, dev_label = self.get_data_from_dataset(dataset, 'validation_mismatched'), \
                                  dataset['validation_mismatched']['label']
            test_word, test_label = self.get_data_from_dataset(dataset, 'test_mismatched'), \
                                    dataset['test_mismatched']['label']
            return train_word, train_label, dev_word, dev_label, test_word, test_label
        dataset = self._load_dataset('glue', nam)
        self.dic_keys = list(dataset['train'].features.keys())
        train_word, train_label = self.get_data_from_dataset(dataset, 'train'), dataset['train']['label']
        dev_word, dev_label = self.get_data_from_dataset(dataset, 'validation'), dataset['validation']['label']
        test_word, test_label = self.get_data_from_dataset(dataset, 'test'), dataset['test']['label']
        return train_word, train_label, dev_word, dev_label, test_word, test_label

    def get_data_from_dataset(self, dataset, name):
        sentences = []
        sentences_list = []
        for key in dataset[name].features.keys():
            sentences_list.append(dataset[name][key])
        for i in range(len(sentences_list[0])):
            sentences_pair = []
            for j in range(len(sentences_list) - 2):
                sentences_pair.append(sentences_list[j][i])
            sentences.append(sentences_pair)
        return sentences



class GLUEDataset_ori(Dataset):
    def __init__(self, sentences, labels, tokenizer, num_classes=2, name='sst2'):
        super(GLUEDataset_ori, self).__init__()
        self.sentences = sentences
        self.labels = labels
        self.set = MyDataset(self.sentences, self.labels)
        self.labeled_set = None
        self.tokenizer = tokenizer
        self.rank_list = None
        self.name = name
        self.num_classes = num_classes

    def __getitem__(self, item):
        return self.sentences[item], self.labels[item]

    def get_labeled_set(self):
        if self.num_classes == 1:
            self.labeled_set = [[]]
            for i in range(len(self.sentences)):
                self.labeled_set[0].append([self.sentences[i], self.labels[i]])
            return
        sentences = [[] for i in range(self.num_classes)]
        for i in range(len(self.sentences)):
            sentences[self.labels[i]].append(self.sentences[i])
        self.labeled_set = sentences

    def gen_prompt(self, sen):
        if self.name == 'sst2':
            return sen[0] + ("" if sen[0].strip().endswith(".") else ".") \
                   + "\nQuestion: Is this sentence positive or negative?\nAnswer:"
        if self.name == 'cola':
            return sen[0] + ("" if sen[0].strip().endswith(".") else ".") \
                   + ".\nQuestion: Is this sentence grammatically acceptable or not?\nAnswer:"
        if self.name == 'mnli':
            return sen[0] + ("" if sen[0].strip().endswith(".") else ".") \
                   + "\nQuestion: " + sen[1] + ("" if sen[1].strip().endswith(".") else ".") \
                   + " True, False or Neither?\nAnswer:"
        if self.name == 'mrpc':
            return "Sentence 1: " + sen[0] + ("" if sen[0].strip().endswith(".") else ".") \
                   + "\nSentence 2: " + sen[1] + ("" if sen[1].strip().endswith(".") else ".") \
                   + "\nQuestion: Do both sentences mean the same thing?\nAnswer:"
        if self.name == 'qnli':
            return sen[0] + ("" if sen[0].strip().endswith(".") else ".") \
                   + '\n' + sen[1] + ("" if sen[1].strip().endswith(".") else ".") \
                   + '\nQuestion: Does this response answer the question?\nAnswer:'
        if self.name == 'qqp':
            return "\nSentence 1: " + sen[0] + ("" if sen[0].strip().endswith(".") else ".") \
                   + "\nSentence 2: " + sen[1] + ("" if sen[1].strip().endswith(".") else ".") \
                   + '\nQuestion: Do both sentences mean the same thing?\nAnswer:'
        if self.name == 'wnli':
            return sen[0] + ("" if sen[0].strip().endswith(".") else ".") \
                   + "\nQuestion: " + sen[1] + " True or False?\nAnswer:"
        if self.name == 'rte':
            return sen[0] + ("" if sen[0].strip().endswith(".") else ".") \
                   + "\nQuestion: " + sen[1] + " True or False?\nAnswer:"

    def as_prompt(self, sen_lab, prompt_begin='', dataset_name='sst2'):
        ans_dataset = {
            'sst2': ["negative", "positive"],
            'cola': ["no", "yes"],
            'mrpc': ["no", "yes"],
            'qnli': ["yes", "no"],
            'qqp': ["no", "yes"],
            'rte': ["True", "False"],
            'mnli': ["True", "Neither", "False"],
            'wnli': ["False", "True"]
        }
        prompt = prompt_begin
        sen_lab = list(sen_lab)
        random.shuffle(sen_lab)
        for i in sen_lab:
            sentence, label = i
            prompt = prompt + self.gen_prompt(sentence) + ans_dataset[dataset_name][label] + '\n###\n'
        return prompt

    def __len__(self):
        return len(self.labels)

    def train_loader(self, data_set=None, batch=1, shuffle=True):
        if not data_set:
            return DataLoader(self.set, batch_size=batch, shuffle=shuffle)
        return DataLoader(data_set, batch_size=batch, shuffle=shuffle)

    def get_rank(self, path='./logs/record1/tr_sst_distilbert-base-uncased.txt', siz=None):
        f = open(path)
        rank_list = [[] for i in range(self.num_classes)]
        count = 0
        st = set()
        while 1:
            s = f.readline()
            if not s:
                break
            s = s.split()
            if len(s) > 1:
                id = int(s[0])
                norm = float(s[1])
                if id in st:
                    continue
                sentences, label = self.sentences[id], self.labels[id]
                if self.num_classes == 1:
                    rank_list[0].append(Item(id, norm))
                    continue
                rank_list[label].append(Item(id, norm))
                st.add(id)
            else:
                norm = float(s[0])
                sentences, label = self.sentences[count], self.labels[count]
                rank_list[label].append(Item(count, norm))
                count += 1
        for i in range(self.num_classes):
            rank_list[i].sort(key=lambda item: item.value)
        # print(len(rank_list[0]))
        self.rank_list = rank_list

    def get_token(self, dataset, batch_size=1, prompt_begin=''):
        dataloader = []
        show = 0
        i_tos = [i for i in range(len(dataset))]
        random.shuffle(i_tos)
        for i in range(0, min(len(i_tos), 500), batch_size):
            inps = []
            labs = []
            for j in range(batch_size):
                id = i_tos[i + j]
                if id >= len(dataset):
                    break
                sen, lab = dataset[id]
                if not show:
                    print(prompt_begin + self.gen_prompt(sen))
                    show = 1
                inps.append(prompt_begin + self.gen_prompt(sen))
                labs.append(lab)
            dataloader.append([
                self.tokenizer(
                    inps,
                    truncation=True,
                    padding=True,
                    add_special_tokens=True,
                    return_tensors='pt'
                ),
                torch.tensor(labs)
            ])
        return dataloader

    def get_attack_set(self, size=10, reverse=1, path='./logs/record1/tr_sst_distilbert-base-uncased.txt'):
        self.get_rank(path=path)
        sentences = []
        labels = []
        for i in range(self.num_classes):
            rank = self.rank_list[i]   # the rank of sentence with label i
            if reverse:
                rank.reverse()
            for j in range(size):
                sentence, label = self.set[rank[j].id]
                sentences.append(sentence)
                labels.append(label)
        return MyDataset(sentences, labels)

    def get_random_set(self, size=50):
        if not self.labeled_set:
            self.get_labeled_set()
        sentences, labels = [], []
        for i in range(self.num_classes):
            random.shuffle(self.labeled_set[i])
            for j in range(size):
                sentences.append(self.labeled_set[i][j])
                labels.append(i)
        return MyDataset(sentences, labels)


def get_glue(name='sst2', tokenizer=None, show=False):
    dataset = DataIOGLUE(name=name)
    train_set = GLUEDataset_ori(dataset.train_word, dataset.train_label, tokenizer, num_classes=dataset.num_classes,
                                name=name)
    dev_set = GLUEDataset_ori(dataset.dev_word, dataset.dev_label, tokenizer, num_classes=dataset.num_classes,
                              name=name)
    if show:
        return train_set, dev_set, dataset
    return train_set, dev_set


class MyDataset(Dataset):
    def __init__(self, photo: list, label: list):
        self.pho = photo
        self.label = label

    def __getitem__(self, item):
        return self.pho[item], self.label[item]

    def __len__(self):
        return len(self.pho)


class MyDataset3(Dataset):
    def __init__(self, photo: list, label: list, ids: list):
        self.pho = photo
        self.label = label
        self.ids = ids

    def __getitem__(self, item):
        return self.pho[item], self.label[item], self.ids[item]

    def __len__(self):
        return len(self.pho)


class Item:
    def __init__(self, _id, value):
        self.id = _id
        self.value = value
