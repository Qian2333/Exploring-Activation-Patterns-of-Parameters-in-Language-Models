from datasets import load_dataset
from tqdm import tqdm
import random
import torch

prompts = [
    "The following are multiple choice questions (with answers) about ",
    "\nQuestion: ",
    "?\nAnswer:"
]
choices = ['A', 'B', 'C', 'D']


def gen_prompt(dataset, i):
    prompt = "The following are multiple choice questions (with answers) about " + dataset['subject'][i] + '.\n\n'
    prompt = prompt + dataset[i]['question'] + '\n'
    for j in range(len(choices)):
        prompt = prompt + choices[j] + '. ' + dataset[i]['choices'][j] + '\n'
    return prompt + 'Answer:'


def gen_prompt_nshot(dataset, n=5):
    prompt = ''
    for i in range(n):
        prompt += gen_prompt(dataset, i) + choices[dataset['answer'][i]] + '\n###\n'
    return prompt


def get_batch_prompt(dataset, i, j):
    batch = []
    for k in range(i, j):
        batch.append(
            prompts[0] + dataset['question'][k] +
            prompts[1] + dataset['passage'][k] +
            prompts[2]
        )
    return batch


def get_mmlu(tokenizer=None, train_set=False, label=False, nsample=None, batch_size=1, n_shot=0):
    # tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    testset = load_dataset('cais/mmlu', 'all', split='validation', trust_remote_code=True)
    trainset = load_dataset('cais/mmlu', 'all', split='auxiliary_train', trust_remote_code=True)
    test_set = []
    prompt0 = gen_prompt_nshot(trainset, n_shot)
    i_tos = [i for i in range(len(testset['question']))]
    if nsample:
        random.shuffle(i_tos)
        i_tos = i_tos[:nsample]
    if label:
        for i in i_tos:
            test_set.append([
                tokenizer(
                    prompt0 + gen_prompt(testset, i),
                    # padding='max_length',
                    # max_length=1400,
                    return_tensors='pt'
                ).input_ids,
                testset['answer'][i]
            ])
        return test_set
    for i in i_tos:
        test_set.append(
            tokenizer(
                prompt0 + gen_prompt(testset, i),
                # padding='max_length',
                # max_length=1400,
                return_tensors='pt'
            ).input_ids
        )
    return test_set


def get_mmlu_calibration(tokenizer=None, train_set=False, label=False, nsample=None, batch_size=1, max_len=2048):
    # tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    testset = load_dataset('cais/mmlu', 'all', split='auxiliary_train', trust_remote_code=True)
    test_set = []
    i_tos = [i for i in range(len(testset['question']))]
    if nsample:
        random.shuffle(i_tos)
        i_tos = i_tos[:nsample]
        j = nsample
        
    for i in tqdm(i_tos):
        tokens = tokenizer(
            gen_prompt(testset, i),
            # padding='max_length',
            # max_length=1400,
            return_tensors='pt'
        ).input_ids
        prompt1 = gen_prompt(testset, i) + choices[int(testset[i]['answer'])]
        while tokens.shape[1] < max_len:
            prompt1 += '\n###\n' + gen_prompt(testset, j)
            tokens = tokenizer(
                prompt1,
                # padding='max_length',
                # max_length=1400,
                return_tensors='pt'
            ).input_ids
            prompt1 += choices[int(testset[j]['answer'])]
            j += 1
        test_set.append(tokens[:, :max_len])
    return test_set


def eval_mmlu(model, tokenizer, test_set):
    ans = 0
    from tqdm import tqdm
    choice_tokens = []
    for choice in choices:
        choice_tokens.append(tokenizer(choice).input_ids[-1])

    for item in tqdm(test_set):
        [inp, lab] = item
        result = model.forward(input_ids=inp.cuda(0)).logits[0, -1, :]
        result = torch.max(torch.tensor([
            result[choice_tokens[0]], result[choice_tokens[1]], result[choice_tokens[2]], result[choice_tokens[3]]
        ]), dim=-1)[1].item()
        if result == lab:
            ans += 1


    print('mmlu')
    print()
    print(ans / len(test_set))
    return ans / len(test_set)

if __name__ == '__main__':
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained('/mnt/data/public/models/7B_hf/', use_fast=False)
    testset = get_mmlu_calibration(tokenizer=tokenizer, nsample=128, max_len=4096)
    print(testset[0])


"""



python lib/dataset/mmlu.py




"""


