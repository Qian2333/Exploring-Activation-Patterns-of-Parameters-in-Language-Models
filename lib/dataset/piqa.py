from datasets import load_dataset
from tqdm import tqdm
import random
import torch
#
# prompts = [
#     "Read the following passage and answer the question. \nThe passage: ",
#     "\nThe question: ",
#     "\nPlease answer true or false: "
# ]
# 61%
prompts = [
    "The following are multiple choice questions (with answers) about ",
    "\nQuestion: ",
    "?\nAnswer:"
]


choices = ['0', '1']
def gen_prompt(dataset, i):
    # prompt = "You will be presented with a task and two possible solutions. " \
    #          "Your goal is to select the solution that best achieves the given task.\n\nTask: " + \
    #          dataset['goal'][i] + '\n\nSolutions:\n'
    prompt = "Question:" + dataset[i]['goal'] + '\n'
    prompt = prompt + choices[0] + '. ' + dataset[i]['sol1'] + '\n'
    prompt = prompt + choices[1] + '. ' + dataset[i]['sol2'] + '\n'
    return prompt + 'Answer:'


def get_piqa(tokenizer=None, train_set=False, label=False, nsample=None, batch_size=1):
    # tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    testset = load_dataset('/mnt/data2/Mydatasets/piqa', split='validation', trust_remote_code=True)
    test_set = []
    i_tos = [i for i in range(len(testset['goal']))]
    if nsample:
        random.shuffle(i_tos)
        i_tos = i_tos[:nsample]
    if label:
        for i in i_tos:
            test_set.append([
                tokenizer(
                    gen_prompt(testset, i),
                    # padding='max_length',
                    # max_length=1400,
                    return_tensors='pt'
                ).input_ids,
                testset['label'][i]
            ])
        return test_set
    for i in i_tos:
        test_set.append(
            tokenizer(
                gen_prompt(testset, i),
                # padding='max_length',
                # max_length=1400,
                return_tensors='pt'
            ).input_ids
        )
    return test_set


def get_piqa_calibration(tokenizer=None, train_set=False, label=False, nsample=None, batch_size=1, max_len=2048):
    # tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    testset = load_dataset('/mnt/data2/Mydatasets/piqa', split='train', trust_remote_code=True)
    test_set = []
    i_tos = [i for i in range(len(testset['goal']))]
    if nsample:
        random.shuffle(i_tos)
        i_tos = i_tos[:nsample]
        j = nsample
        
    for i in i_tos:
        tokens = tokenizer(
            gen_prompt(testset, i),
            # padding='max_length',
            # max_length=1400,
            return_tensors='pt'
        ).input_ids
        prompt1 = gen_prompt(testset, i) + choices[int(testset[i]['label'])]
        while tokens.shape[1] < max_len:
            prompt1 += '\n###\n' + gen_prompt(testset, j)
            tokens = tokenizer(
                prompt1,
                # padding='max_length',
                # max_length=1400,
                return_tensors='pt'
            ).input_ids
            prompt1 += choices[int(testset[j]['label'])]
            j += 1
        test_set.append(tokens[:, :max_len])
    return test_set


def eval_piqa(model, tokenizer, test_set):
    ans = 0
    from tqdm import tqdm

    choice_tokens = []
    for choice in choices:
        choice_tokens.append(tokenizer(choice).input_ids[-1])
        
    for item in tqdm(test_set):
        [inp, lab] = item
        result = model.forward(input_ids=inp.cuda(0)).logits[0, -1, :]


        result = torch.max(torch.tensor([
            result[choice_tokens[0]], result[choice_tokens[1]]
        ]), dim=-1)[1].item()
        if result == (int(lab)):
            ans += 1


    print('piqa')
    print()
    print(ans / len(test_set))
    return ans / len(test_set)

if __name__ == '__main__':
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained('/mnt/data/public/models/7B_hf/', use_fast=False)


"""



python lib/dataset/boolq.py




"""


