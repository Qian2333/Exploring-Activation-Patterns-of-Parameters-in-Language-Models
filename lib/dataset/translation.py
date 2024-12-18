# dataset = load_dataset("davidstap/ted_talks", "zh-cn_en", trust_remote_code=True)
from datasets import load_dataset
from tqdm import tqdm
import random
import torch


def get_ted_talk(tokenizer=None, nsample=128, batch_size=1):
    # tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    testset = load_dataset("davidstap/ted_talks", "zh-cn_en", split='train', trust_remote_code=True)

    test_set_zh = []
    test_set_en = []
    i_tos = [i for i in range(len(testset['zh-cn']))]
    if nsample:
        random.shuffle(i_tos)
        i_tos = i_tos[:nsample]

    for i in i_tos:
        test_set_zh.append(
            tokenizer(
                testset[i]['zh-cn'],
                # padding='max_length',
                # max_length=1400,
                return_tensors='pt'
            ).input_ids
        )
        test_set_en.append(
            tokenizer(
                testset[i]['en'],
                # padding='max_length',
                # max_length=1400,
                return_tensors='pt'
            ).input_ids
        )
    return test_set_zh, test_set_en
