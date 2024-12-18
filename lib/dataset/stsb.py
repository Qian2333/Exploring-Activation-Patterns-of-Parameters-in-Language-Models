from datasets import load_dataset
from tqdm import tqdm
import random
import torch

def get_stsb(tokenizer=None, train_set=False, nsample=None, batch_size=1):
    testset = load_dataset('/mnt/data2/Mydatasets/SetFit/stsb', split='validation')
    test_set = []
    i_tos = [i for i in range(len(testset['idx']))]
    if nsample:
        # random.shuffle(i_tos)
        i_tos = i_tos[:nsample]
    for i in i_tos:
        test_set.append(
            [
                tokenizer(
                    testset[i]['text1'],
                    return_tensors='pt'
                ).input_ids,
                tokenizer(
                    testset[i]['text2'],
                    return_tensors='pt'
                ).input_ids,
                testset[i]['label']
            ]
        )
    return test_set


def get_sick(tokenizer=None, train_set=False, nsample=None, batch_size=1):
    if train_set:
        testset = load_dataset('/mnt/data2/Mydatasets/sick', split='train')
    else:
        testset = load_dataset('/mnt/data2/Mydatasets/sick', split='validation')
    test_set = []
    i_tos = [i for i in range(len(testset['id']))]
    if nsample:
        random.shuffle(i_tos)
        i_tos = i_tos[:nsample]
    for i in i_tos:
        test_set.append(
            [
                tokenizer(
                    testset[i]['sentence_A'],
                    # padding='max_length',
                    # max_length=1400,
                    return_tensors='pt'
                ).input_ids,
                tokenizer(
                    testset[i]['sentence_B'],
                    # padding='max_length',
                    # max_length=1400,
                    return_tensors='pt'
                ).input_ids,
                testset[i]['relatedness_score']
            ]
        )
    return test_set


if __name__ == '__main__':
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained('/mnt/data/public/models/7B_hf/', use_fast=False)


"""



python lib/dataset/boolq.py


huggingface-cli download --repo-type dataset --resume-download nyu-mll/glue --local-dir /mnt/data2/Mydatasets/glue



"""


