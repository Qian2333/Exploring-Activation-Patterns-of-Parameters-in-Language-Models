# _*_ coding:utf-8 _*_
# 利用深度学习做情感分析，基于Imdb 的50000个电影评论数据进行；
import os
import matplotlib.pyplot as plt

# os.environ["CUDA_VISIBLE_DEVICES"] = "4, 5, 6, 7"
from lib.util.logger import Logger, str_pad
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import re
from random import sample
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from transformers import BertModel, BertTokenizer
from lib.dataset import get_boolq_calibration, get_dataset
from lib.util.toolbag import setup_seed
from lib.util.weight_score import get_gradient_tensors_by_modules
from lib.model.llm_new import get_model
from tqdm import tqdm
import time
import argparse


def get_llm(model_name=None, cache_dir="/mnt/data/public/models/7B_hf/"):
    model = AutoModelForCausalLM.from_pretrained(
        cache_dir,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto"
    )

    model.seqlen = model.config.max_position_embeddings
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--times', default=64, type=int,
                        help='sample times')
    parser.add_argument('-sd', '--seed', default=1, type=int,
                        help='seed for random')
    parser.add_argument('-md', '--model', default='/data/wyd/MyModels/llama2_7bhf', type=str,
                        help='path for llama')
    args = parser.parse_args()

    print(args)
    setup_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        use_fast=False
    )
    net1 = get_llm(cache_dir=args.model).eval()
    optimizer1 = optim.SGD(net1.parameters(), lr=1e-5)

    data0 = []
    root = 'logs/dis_fig_new'
    if not os.path.exists(root):
        os.mkdir(root)
    ['boolq', 'humaneval', 'mmlu', 'siqa', 'hellaswag', 'c4', 'wikitext2']
    ans = []
    wikitext2_ans = []
    for dataset_name in ['boolq', 'humaneval', 'mmlu', 'siqa', 'hellaswag']:

        data1 = get_dataset(name=dataset_name, tokenizer=tokenizer, nsample=args.times, shot=3)
        print('data1 ready')

        ans_sta = None
        for i in tqdm(range(args.times)):
            # print(data1[i])
            # exit(0)
            result1 = net1.forward(data1[i].to(torch.device("cuda:0"))).logits[0, -1, :]
            tr1 = get_gradient_tensors_by_modules(
                net1, optimizer1,
                torch.max(result1, dim=-1)[0]
            )
            if ans_sta == None:
                ans_sta = []
                for tensor in tr1:
                    # ans_sta.append([torch.abs(tensor).cpu()])
                    ans_sta.append(torch.abs(tensor))
            else:
                for i, tensor in enumerate(tr1):
                    # ans_sta[i].append(torch.abs(tensor).cpu())
                    ans_sta[i] += torch.abs(tensor)


        for i in tqdm(range(0, len(ans_sta), 7)):
            # for j in range(7):
            #     ans_sta[i + j] = torch.stack(ans_sta[i + j])
            # vec = torch.cat(ans_sta[i:i + 7], dim=1)
            # a = torch.sum(vec > 0.00005, dim=1)
            # for j in range(args.times):
            #     data0.append([i // 7 + 1, dataset_name, a[j].item()])

            vec = torch.cat(ans_sta[i:i + 7], dim=-1) / args.times
            if len(ans_sta) // 7 == 32:
                a = torch.sum(vec > 0.000002)
                # a = torch.mean(vec)
            else:
                a = torch.sum(vec > 0.000005)
            data0.append([i // 7 + 1, dataset_name, a.item()])
            if dataset_name == 'c4':
                ans.append(a.item())
            if dataset_name == 'wikitext2':
                wikitext2_ans.append(a.item())
    print(ans)
    print(wikitext2_ans)

    import seaborn as sns
    from pandas import DataFrame
    data0 = DataFrame(data0, columns=['Layer', 'Dataset', 'Number of activated parameters'])
    print(data0)
    
    sns.set_theme(style="whitegrid")
    fig = sns.lineplot(data=data0, x='Layer', y='Number of activated parameters', hue='Dataset')
    fig = fig.get_figure()
    fig.savefig(root + '/show.png')
    plt.close()


if __name__ == '__main__':

    main()

"""

python distribution_draw.py -md /mnt/data2/wyd/MyModels/llama2_7b_hf


python distribution_draw.py -md /mnt/data/public/models/7B_hf/

python distribution_draw.py -md /mnt/data2/wyd/MyModels/llama2_13b_hf

python distribution_draw.py -md /mnt/data2/wyd/MyModels/llama3_8b

[1061292, 6319729, 14767367, 19149928, 21451890, 22124198, 21653518, 22533946, 22715992, 23012050, 23320873, 21679099, 21088533, 21843715, 21866557, 22593499, 23646562, 22568772, 21387521, 20828237, 21268883, 18788506, 17122682, 15369127, 14025833, 12663446, 12024558, 10586793, 10162286, 8258830, 8591076, 8899162]

python -m debugpy --listen 127.0.0.1:5678 --wait-for-client distribution_draw.py -md /data/public/models/llama2/Llama-2-7b-hf/

"""