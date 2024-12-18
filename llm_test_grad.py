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
from lib.dataset import get_dataset
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
    'llama3_8b': '/mnt/data2/wyd/MyModels/llama3_8b',
    'llama_7b': '/mnt/data/public/models/7B_hf/'
}
# 语意相似度 不同层的功能 模型能力 数据相似度 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s1', default='sst2', type=str,
                        help='the datasets')
    parser.add_argument('-s2', default='sst2', type=str,
                        help='the datasets')
    parser.add_argument('-t', '--times', default=64, type=int,
                        help='sample times')
    parser.add_argument('-sd', '--seed', default=1, type=int,
                        help='seed for random')
    parser.add_argument('-md', '--model', default='llama2_7b', type=str,
                        help='path for llama')
    args = parser.parse_args()

    setup_seed(args.seed)
    logger1 = Logger(name='llama_cos', tim=False)
    logger2 = Logger(name='llama_cos_' + args.s1 + '_' + args.s2)

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

    data1 = get_dataset(name=args.s1, tokenizer=tokenizer, nsample=args.times, shot=0, max_len=256)

    print('data1 ready')

    data2 = get_dataset(name=args.s2, tokenizer=tokenizer, nsample=args.times, shot=3, max_len=256)

    print('data2 ready')

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
    layer_num = 0
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
            layer_num = len(tr1) // layer_l
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
    fig.savefig('logs/plot2/' + args.model + '_' + args.s1 + '_' + args.s2 + '.png')

    ans_ls = []
    for i in range(layer_num // 2, layer_num - 2):
        ans_ls += ans_sta[i]
    ans_ls = torch.tensor(ans_ls)
    print('llmdcos', torch.mean(ans_ls), torch.std(ans_ls))
    logger1.info('model cache: ' + args.model + '|' + args.s1 + '|' + args.s2 \
                 + '\llmdcos: ' + str(torch.mean(ans_ls).item()) + '|' + str(torch.std(ans_ls).item()) + '\n')
    logger1.info('----------------------------------------------------------------------------------')

    for i in range(len(ans_sta)):
        item = torch.tensor(ans_sta[i])
        ans_sta[i] = [torch.mean(item), torch.std(item)]

    # for i in range(0, len(ans_sta), 7):
    # 	print('layer', i // 7)
    # 	for j in range(7):
    # 		print(ans_sta[i + j][0], end=' ')
    # 	print()
    # 	for j in range(7):
    # 		print(ans_sta[i + j][1], end=' ')
    # 	print()
    logger2.info('model cache: ' + args.model + '\n')
    for i in range(0, len(ans_sta)):
        print('layer', i, ans_sta[i][0].item(), ans_sta[i][1].item())
        logger2.info('layer ' + str(i) + ' ' + str(ans_sta[i][0].item()) + ' ' + str(ans_sta[i][1].item()) + '\n')

#     ans_sta /= args.times
#     print(ans_sta)


if __name__ == '__main__':
    main()

"""


python llm_test_grad.py -s1 boolq -s2 boolq -md /data/public/models/qwen-7B/


python llm_test_grad.py -s1 siqa -s2 siqa -md /data/public/models/llama2/Llama-2-7b-hf/
python llm_test_grad.py -s1 boolq -s2 boolq -md /data/public/models/llama2/Llama-2-7b-hf/


CUDA_VISIBLE_DEVICES="0, 1, 2, 3" python llm_test_grad.py -s1 siqa -s2 boolq -md /data/public/models/llama2/Llama-2-7b-hf/



python llm_test_grad.py -s1 boolq -s2 boolq -md /data/wyd/MyModels/llama3_8b


python llm_test_grad.py -s1 gsm8k -s2 gsm8k -md /data/public/models/llama2/Llama-2-7b-hf/

python llm_test_grad.py -s1 siqa -s2 siqa -md /data/public/models/llama2/Llama-2-7b-hf/

tensor([0.5338, 0.4195, 0.5432, 0.2943, 
0.3736, 0.2964, 0.3015, 0.2942, 0.3152,
        0.2920, 0.2900, 0.2900, 0.2602, 0.2075, 0.1706, 0.1782,
         0.1546, 0.1754,
        0.1499, 0.1308, 0.1696, 0.1410, 0.1456, 0.1324,
         0.0968, 0.1095, 0.1300,
        0.1828, 0.4779, 0.3690, 0.2645, 0.3490], device='cuda:1')

python llm_test_grad.py -s1 boolq -s2 humaneval -md /data/public/models/llama-7b/

tensor([0.5610, 0.4490, 0.5693, 0.3240, 0.3301, 0.2700, 0.3080, 0.3036, 0.3179,
        0.3103, 0.2521, 0.2988, 0.2815, 0.2653, 0.2029, 0.2106, 0.1713, 0.1918,
        0.1719, 0.1679, 0.2324, 0.1408, 0.1846, 0.3007, 0.1139, 0.2208, 0.2057,
        0.1380, 0.2754, 0.3541, 0.2639, 0.3745], device='cuda:1')
        
python llm_test_grad.py -s1 boolq -s2 mmlu -md /data/public/models/llama-7b/

tensor([0.5413, 0.4746, 0.6916, 0.3700, 0.4282, 0.3578, 0.3500, 0.3295, 0.3516,
        0.3435, 0.3240, 0.3133, 0.2776, 0.2680, 0.2373, 0.2664, 0.2522, 0.3099,
        0.2455, 0.2009, 0.2144, 0.1829, 0.2535, 0.1645, 0.1661, 0.1567, 0.2195,
        0.2529, 0.4960, 0.3206, 0.4038, 0.2882], device='cuda:1')
        
python llm_test_grad.py -s1 boolq -s2 boolq -md /data/public/models/llama-7b/

tensor([0.8500, 0.8361, 0.8769, 0.8418, 0.8429, 0.8403, 0.8282, 0.8415, 0.8144,
        0.8202, 0.8639, 0.8073, 0.8182, 0.8253, 0.8260, 0.8516, 0.8562, 0.8493,
        0.8844, 0.8504, 0.8666, 0.8503, 0.8670, 0.8875, 0.8739, 0.8667, 0.8596,
        0.9259, 0.8807, 0.8936, 0.8832, 0.9606], device='cuda:1')
        
        
python llm_test_grad.py -s1 humaneval -s2 humaneval -md /data/public/models/llama-7b/

tensor([0.6855, 0.6434, 0.7239, 0.5872, 0.6541, 0.5647, 0.5551, 0.5577, 0.5347,
        0.5321, 0.5523, 0.5413, 0.5648, 0.5857, 0.6162, 0.6572, 0.6252, 0.6204,
        0.6348, 0.6412, 0.6667, 0.6102, 0.6384, 0.6216, 0.6693, 0.6949, 0.6841,
        0.6915, 0.8191, 0.7972, 0.7836, 0.8773], device='cuda:1')

python llm_test_grad.py -s1 mmlu -s2 mmlu -md /data/public/models/llama-7b/

tensor([0.6460, 0.6106, 0.9071, 0.5414, 0.5829, 0.5650, 0.5709, 0.5912, 0.5711,
        0.6040, 0.6099, 0.6110, 0.6101, 0.6251, 0.6719, 0.6798, 0.7139, 0.7043,
        0.7108, 0.7184, 0.7198, 0.6752, 0.7043, 0.6086, 0.6485, 0.7397, 0.6880,
        0.7615, 0.7644, 0.7610, 0.6978, 0.7651], device='cuda:1')

以下结果为纯梯度，接下来考虑梯度*w

python llm_test_grad.py -s1 boolq -s2 humaneval
tensor([   nan, 0.4148, 0.4167, 0.4949, 0.5337, 0.4819, 0.4942, 0.4723, 0.5220,
        0.5212, 0.4885, 0.4924, 0.4715, 0.4359, 0.4026, 0.3961, 0.3768, 0.3492,
        0.3495, 0.3366, 0.3421, 0.3444, 0.3489, 0.3776, 0.3085, 0.3512, 0.3313,
        0.2641, 0.3218, 0.3824, 0.2973, 0.3724], device='cuda:1')

python llm_test_grad.py -s1 boolq -s2 boolq

tensor([   nan,    nan,    nan, 0.8430, 0.8506, 0.8386, 0.8558, 0.8543, 0.8362,
        0.8403, 0.8469, 0.8434, 0.8434, 0.8529, 0.8548, 0.8624, 0.8748, 0.8790,
        0.8807, 0.8713, 0.8767, 0.8648, 0.8786, 0.8691, 0.8750, 0.8757, 0.8654,
        0.9030, 0.8862, 0.8853, 0.9000, 0.9519], device='cuda:1')


python llm_test_grad.py -s1 humaneval -s2 humaneval

tensor([0.7240, 0.7076, 0.4992, 0.6804, 0.6808, 0.6705, 0.6680, 0.6684, 0.6761,
        0.6812, 0.6783, 0.6631, 0.6712, 0.6569, 0.6698, 0.6968, 0.6883, 0.6785,
        0.6919, 0.6927, 0.7029, 0.6942, 0.6988, 0.7062, 0.7082, 0.7162, 0.7167,
        0.7386, 0.7575, 0.7569, 0.7822, 0.8974], device='cuda:1')

python llm_test_grad.py -s1 boolq -s2 c4

python llm_test_grad.py -s1 hellaswag -s2 c4 -md /data/public/models/llama-7b/

tensor([0.3854, 0.3597, 0.5737, 0.3069, 0.3481, 0.3234, 0.3279, 0.3201, 0.3541,
        0.3389, 0.3080, 0.2877, 0.2624, 0.2535, 0.2061, 0.2015, 0.1904, 0.1678,
        0.1518, 0.1432, 0.1450, 0.1219, 0.1209, 0.1203, 0.1113, 0.1029, 0.1156,
        0.1363, 0.2207, 0.2315, 0.2464, 0.2515], device='cuda:1')

python llm_test_grad.py -s1 siqa -s2 c4 -md /data/public/models/llama-7b/

tensor([0.3958, 0.3718, 0.7137, 0.2921, 0.3226, 0.2800, 0.2795, 0.2788, 0.3072,
        0.2816, 0.2722, 0.2713, 0.2380, 0.2304, 0.1854, 0.1785, 0.1798, 0.1724,
        0.1386, 0.1429, 0.1452, 0.1215, 0.1327, 0.1187, 0.0989, 0.0854, 0.1055,
        0.1447, 0.2783, 0.2591, 0.2443, 0.2837], device='cuda:1')

python llm_test_grad.py -s1 piqa -s2 c4 -md /data/public/models/llama-7b/

tensor([0.3683, 0.3192, 0.7156, 0.3165, 0.2635, 0.2970, 0.2910, 0.2765, 0.3001,
        0.2888, 0.2623, 0.2463, 0.2099, 0.2013, 0.1837, 0.1521, 0.1485, 0.1442,
        0.1161, 0.1196, 0.1226, 0.0886, 0.1125, 0.1007, 0.0963, 0.0992, 0.0995,
        0.1237, 0.2233, 0.1644, 0.2022, 0.1974], device='cuda:1')


python llm_test_grad.py -s1 humaneval -s2 c4 -md /data/public/models/llama-7b/

tensor([0.3328, 0.2889, 0.5404, 0.2282, 0.2871, 0.2342, 0.2454, 0.2716, 0.2813,
        0.2603, 0.2448, 0.2297, 0.2109, 0.1579, 0.1432, 0.1316, 0.1238, 0.1299,
        0.1073, 0.1035, 0.1066, 0.0906, 0.0808, 0.0809, 0.0769, 0.0744, 0.0770,
        0.1054, 0.2124, 0.2203, 0.1777, 0.2468], device='cuda:1')

        700+9+8210+14192-952+10311+1715+600+324+1200+3750+8600-30

        
"""