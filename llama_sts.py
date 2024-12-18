from lib.dataset import get_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from scipy.stats import spearmanr
import argparse
import numpy as np
from tqdm import tqdm
import torch
from lib.util.logger import Logger
from lib.util.toolbag import setup_seed
import torch.optim as optim
import copy
from lib.util.weight_score import get_gradient_tensors_by_modules



def get_llm(model_name=None, cache_dir="/mnt/data/public/models/7B_hf/"):
    if model_name == 'qwen_7b':
        model = AutoModelForCausalLM.from_pretrained(
            cache_dir,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            cache_dir,
            torch_dtype=torch.float16,
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', default='stsb', type=str,
                        help='the datasets')
    parser.add_argument('-t', '--times', default=256, type=int,
                        help='sample times')
    parser.add_argument('-sd', '--seed', default=1, type=int,
                        help='seed for random')
    parser.add_argument('-md', '--model', default='llama2_7b', type=str,
                        help='path for llama')
    
    args = parser.parse_args()
    setup_seed(args.seed)
    logger1 = Logger(name='llama_sim', tim=False)

    cache_dir = model_file[args.model]

    if args.model == 'qwen_7b':
        tokenizer = AutoTokenizer.from_pretrained(
            cache_dir
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            cache_dir,
            use_fast=False
        )
    model = get_llm(args.model, cache_dir=cache_dir)

    stsb = get_dataset(args.s, tokenizer=tokenizer, nsample=args.times)

    labels = []

    optimizer1 = optim.SGD(model.parameters(), lr=1e-5)

    layer_l = 7
    if args.model == 'qwen_7b':
        layer_l = 5
    ans_sta = None
    for item in tqdm(stsb):
        text1, text2, label = item
        result1 = model.forward(text1.to(torch.device("cuda:0"))).logits[0, -1, :]
        tr1 = get_gradient_tensors_by_modules(
            model, optimizer1,
            torch.max(result1, dim=-1)[0]
        )
        result2 = model.forward(text2.to(torch.device("cuda:0"))).logits[0, -1, :]
        tr2 = get_gradient_tensors_by_modules(
            model, optimizer1,
            torch.max(result2, dim=-1)[0]
        )
        if ans_sta == None:
            ans_sta = [[] for _ in range(len(tr1) // layer_l)]
            layer_num = len(tr1) // layer_l
        for i in range(0, len(tr1), layer_l):
            mds1, mds2 = [], []
            for j in range(layer_l):
                mds1.append(tr1[i + j].to(torch.device("cuda:3")).to(torch.float32))
                mds2.append(tr2[i + j].to(torch.device("cuda:3")).to(torch.float32))
            vec1 = torch.cat(mds1, dim=0)
            vec2 = torch.cat(mds2, dim=0)
            ans_sta[i // layer_l].append((torch.sum(torch.abs(vec1 * vec2)) / torch.sqrt(torch.sum(vec1 * vec1)) / \
                                torch.sqrt(torch.sum(vec2 * vec2))).item())
            
        labels.append(label)

    outputs = np.array([0.0 for i in range(len(labels))])
    for i in range(12, len(ans_sta) - 2):
        # print(ans_sta[i])
        outputs += np.array(ans_sta[i])
    # print(outputs, labels)
    
    r, p = spearmanr(outputs, labels)
    print(r, p)
    logger1.info('model cache: ' + args.model + '|' + args.s \
                 + '\n12-2 llmdcos: ' + str(float(r)) + '|' + str(float(p)) + '\n')
    logger1.info('----------------------------------------------------------------------------------')

    outputs = np.array([0.0 for i in range(len(labels))])
    for i in range(12, len(ans_sta)):
        # print(ans_sta[i])
        outputs += np.array(ans_sta[i])
    # print(outputs, labels)
    
    r, p = spearmanr(outputs, labels)
    print(r, p)
    logger1.info('model cache: ' + args.model + '|' + args.s \
                 + '\n12 llmdcos: ' + str(float(r)) + '|' + str(float(p)) + '\n')
    logger1.info('----------------------------------------------------------------------------------')

    outputs = np.array([0.0 for i in range(len(labels))])
    for i in range(16, len(ans_sta) - 2):
        # print(ans_sta[i])
        outputs += np.array(ans_sta[i])
    # print(outputs, labels)
    
    r, p = spearmanr(outputs, labels)
    print(r, p)
    logger1.info('model cache: ' + args.model + '|' + args.s \
                 + '\n16 - 2llmdcos: ' + str(float(r)) + '|' + str(float(p)) + '\n')
    logger1.info('----------------------------------------------------------------------------------')

    outputs = np.array([0.0 for i in range(len(labels))])
    for i in range(16, len(ans_sta)):
        # print(ans_sta[i])
        outputs += np.array(ans_sta[i])
    # print(outputs, labels)
    
    r, p = spearmanr(outputs, labels)
    print(r, p)
    logger1.info('model cache: ' + args.model + '|' + args.s \
                 + '\n16 llmdcos: ' + str(float(r)) + '|' + str(float(p)) + '\n')
    logger1.info('----------------------------------------------------------------------------------')

    outputs = np.array([0.0 for i in range(len(labels))])
    for i in range(20, len(ans_sta) - 2):
        # print(ans_sta[i])
        outputs += np.array(ans_sta[i])
    # print(outputs, labels)
    
    r, p = spearmanr(outputs, labels)
    print(r, p)
    logger1.info('model cache: ' + args.model + '|' + args.s \
                 + '\n20-2 llmdcos: ' + str(float(r)) + '|' + str(float(p)) + '\n')
    logger1.info('----------------------------------------------------------------------------------')

    outputs = np.array([0.0 for i in range(len(labels))])
    for i in range(20, len(ans_sta)):
        # print(ans_sta[i])
        outputs += np.array(ans_sta[i])
    # print(outputs, labels)
    
    r, p = spearmanr(outputs, labels)
    print(r, p)
    logger1.info('model cache: ' + args.model + '|' + args.s \
                 + '\n20 llmdcos: ' + str(float(r)) + '|' + str(float(p)) + '\n')
    logger1.info('----------------------------------------------------------------------------------')


if __name__ == "__main__":
    main()


"""

0: statistic=0.468656079002973, pvalue=7.693580469259143e-18
1: statistic=0.6033020955528846, pvalue=3.1642652529422146e-31
2: statistic=0.5697840051461707, pvalue=2.6259359941914347e-27
3: statistic=0.5146736790671022, pvalue=9.290842347341396e-22
4: statistic=0.44439545211283316, pvalue=5.319279525899421e-16
5: statistic=0.40673238372117493, pvalue=2.0231232658066004e-13
6: statistic=0.3572837420597305, pvalue=1.7159910856594259e-10
7: statistic=0.3501853968591672, pvalue=4.125797162503305e-10
8: statistic=0.298528361563874, pvalue=1.3019249994987065e-07
9: statistic=0.2220054304521593, pvalue=0.00010266555698312788
10:statistic=0.2557021766980809, pvalue=7.031963452716685e-06
11:statistic=0.2861744893361799, pvalue=4.4112316653217573e-07
12:statistic=0.05492006501006809, pvalue=0.5854223142324053

"""

