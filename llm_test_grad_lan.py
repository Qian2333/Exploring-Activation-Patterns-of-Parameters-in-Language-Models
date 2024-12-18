# _*_ coding:utf-8 _*_
# 利用深度学习做情感分析，基于Imdb 的50000个电影评论数据进行；
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "4, 5, 6, 7"
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import torch
from lib.util.logger import Logger, str_pad
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import re
from random import sample
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from transformers import BertModel, BertTokenizer
from lib.dataset.translation import get_ted_talk
from lib.util.toolbag import setup_seed
from lib.util.weight_score import get_gradient_tensors_by_modules
from lib.model.llm_new import get_model
from tqdm import tqdm
import time
import argparse


def get_llm(model_name=None, cache_dir="/mnt/data/public/models/7B_hf/"):
    if model_name == 'qwen_7b':
        model = AutoModelForCausalLM.from_pretrained(
            cache_dir,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="auto",
            trust_remote_code=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            cache_dir,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="auto"
        )

    model.seqlen = model.config.max_position_embeddings
    return model



model_file = {
    'llama2_13b': '/mnt/data2/wyd/MyModels/llama2_13b_hf/',
    'llama2_7b': '/mnt/data2/wyd/MyModels/llama2_7b_hf/' ,
    'qwen_7b': '/mnt/data2/wyd/MyModels/qwen7b/',
    'llama3_8b': '/mnt/data2/wyd/MyModels/llama3_8b'
}
# 语意相似度 不同层的功能 模型能力 数据相似度 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--times', default=128, type=int,
                        help='sample times')
    parser.add_argument('-sd', '--seed', default=1, type=int,
                        help='seed for random')
    parser.add_argument('-md', '--model', default='llama2_7b', type=str,
                        help='path for llama')
    args = parser.parse_args()

    setup_seed(args.seed)

    cache_dir = model_file[args.model]

    if args.model == 'qwen_7b':
        tokenizer = AutoTokenizer.from_pretrained(
            cache_dir,
            use_fast=False,
            trust_remote_code=True
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            cache_dir,
            use_fast=False
        )

    data1, data2 = get_ted_talk(tokenizer=tokenizer, nsample=args.times)
    print('data ready')

    net1 = get_llm(model_name=args.model, cache_dir=cache_dir).eval()

    optimizer1 = optim.SGD(net1.parameters(), lr=1e-5)

    # print(net1)
    # exit(0)
    #     ans_sta = torch.zeros([224]).to(torch.device("cuda:1"))
    # ans_sta = [[] for _ in range(224)]
    layer_l = 7
    if args.model == 'qwen_7b':
        layer_l = 5
    ans_sta = None
    for i in tqdm(range(args.times)): 
        result1 = net1.forward(data1[i].to(torch.device("cuda:0"))).logits[0, -1, :]
        tr1 = get_gradient_tensors_by_modules(
            net1, optimizer1,
            torch.max(result1, dim=-1)[0]
        )
        result2 = net1.forward(data2[i].to(torch.device("cuda:0"))).logits[0, -1, :]
        tr2 = get_gradient_tensors_by_modules(
            net1, optimizer1,
            torch.max(result2, dim=-1)[0]
        )
        if ans_sta == None:
            ans_sta = [[] for _ in range(len(tr1) // layer_l)]
        for i in range(0, len(tr1), layer_l):
            mds1, mds2 = [], []
            for j in range(layer_l):
                mds1.append(tr1[i + j].to(torch.device("cuda:1")))
                mds2.append(tr2[i + j].to(torch.device("cuda:1")))
            vec1 = torch.cat(mds1, dim=0)
            vec2 = torch.cat(mds2, dim=0)
            ans_sta[i // layer_l].append((torch.sum(torch.abs(vec1 * vec2)) / torch.sqrt(torch.sum(vec1 * vec1)) / \
                                torch.sqrt(torch.sum(vec2 * vec2))).item())
        # for i in range(0, len(tr1)):
        # 	vec1 = tr1[i].to(torch.device("cuda:1")).to(torch.float64)
        # 	vec2 = tr2[i].to(torch.device("cuda:1")).to(torch.float64)
        # 	ans_sta[i].append((torch.sum(torch.abs(vec1 * vec2)) / torch.sqrt(torch.sum(vec1 * vec1)) / \
        # 						torch.sqrt(torch.sum(vec2 * vec2))).item())
    data0 = []
    for i in range(0, len(ans_sta)):
        for item in ans_sta[i]:
            data0.append([i + 1, item])
    import seaborn as sns
    from pandas import DataFrame
    data0 = DataFrame(data0, columns=['layer', 'similarity'])
    print(data0)
    fig = sns.boxplot(x='layer', y='similarity', data=data0)
    xaxis = fig.get_xticks()
    # for i in range(len(xaxis)):
    # 	if i % 4 != 0:
    # 		xaxis[i].text = ''
    fig.set_xticks(xaxis[3::4])
    fig.set_yticks([i * 0.2 for i in range(6)])
    # print(xaxis)
    # exit(0)
    fig = fig.get_figure()
    fig.savefig('logs/plot3/' + args.model + '.png')

    ans_ls = []
    for i in range(16, 30):
        ans_ls += ans_sta[i]
    ans_ls = torch.tensor(ans_ls)
    print('llmdcos', torch.mean(ans_ls), torch.std(ans_ls))

    for i in range(len(ans_sta)):
        item = torch.tensor(ans_sta[i])
        ans_sta[i] = [torch.mean(item), torch.std(item)]

    for i in range(0, len(ans_sta)):
        print('layer', i, ans_sta[i][0].item(), ans_sta[i][1].item())
        # logger2.info('layer ' + str(i) + ' ' + str(ans_sta[i][0].item()) + ' ' + str(ans_sta[i][1].item()) + '\n')

#     ans_sta /= args.times
#     print(ans_sta)


if __name__ == '__main__':
    main()

"""


python llm_test_grad_lan.py -s1 boolq -s2 boolq -md /data/public/models/qwen-7B/

python llm_test_grad_lan.py -md qwen_7b

python llm_test_grad_lan.py -md llama2_7b

python llm_test_grad_lan.py -md llama3_8b

python llm_test_grad.py -s1 siqa -s2 siqa -md /data/public/models/llama2/Llama-2-7b-hf/
python llm_test_grad.py -s1 boolq -s2 boolq -md /data/public/models/llama2/Llama-2-7b-hf/


CUDA_VISIBLE_DEVICES="0, 1, 2, 3" python llm_test_grad.py -s1 siqa -s2 boolq -md /data/public/models/llama2/Llama-2-7b-hf/



python llm_test_grad.py -s1 boolq -s2 boolq -md /data/wyd/MyModels/llama3_8b


python llm_test_grad.py -s1 gsm8k -s2 gsm8k -md /data/public/models/llama2/Llama-2-7b-hf/

python llm_test_grad.py -s1 siqa -s2 siqa -md /data/public/models/llama2/Llama-2-7b-hf/


python llm_test_grad.py -md llama2_7b -s1 c4 -s2 c4
python llm_test_grad.py -md llama2_7b -s1 c4 -s2 humaneval



"""