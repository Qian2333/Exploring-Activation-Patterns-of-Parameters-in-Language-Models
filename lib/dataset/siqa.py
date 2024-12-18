from datasets import load_dataset
from tqdm import tqdm
import random
import torch

choices = ['A', 'B', 'C']

def gen_prompt(dataset, i):
    prompt = 'The following are multiple choice questions (with answers) about social interaction.\n'
    prompt = prompt + dataset[i]['context'] + '\n'
    prompt = prompt + dataset[i]['question'] + '\n'
    prompt = prompt + choices[0] + '. ' + dataset[i]['answerA'] + '\n'
    prompt = prompt + choices[1] + '. ' + dataset[i]['answerB'] + '\n'
    prompt = prompt + choices[2] + '. ' + dataset[i]['answerC'] + '\n'
    return prompt + 'Answer:'


def get_siqa(tokenizer=None, train_set=False, label=False, nsample=None, batch_size=1):
    # tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    # testset = load_dataset('piqa', split='validation', trust_remote_code=True)
    if train_set:
        testset = load_dataset('/mnt/data2/Mydatasets/siqa', split='train')
    else:
        testset = load_dataset('/mnt/data2/Mydatasets/siqa', split='validation')
    test_set = []
    i_tos = [i for i in range(len(testset['context']))]
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


def get_siqa_calibration(tokenizer=None, train_set=False, label=False, nsample=None, batch_size=1, max_len=2048):

    testset = load_dataset('/mnt/data2/Mydatasets/siqa', split='train')
    test_set = []
    i_tos = [i for i in range(len(testset['context']))]
    if nsample:
        random.shuffle(i_tos)
        i_tos = i_tos[:nsample]
        j = nsample

    for i in tqdm(i_tos):
        prompt1 = gen_prompt(testset, i)

        num_count = 0
        while num_count < 100:
            prompt1 += '\n###\n' + gen_prompt(testset, j)
            j += 1
            num_count += 1

        tokens = tokenizer(
            prompt1,
            # padding='max_length',
            # max_length=1400,
            return_tensors='pt'
        ).input_ids
        while tokens.shape[1] < max_len:
            prompt1 += '\n###\n' + gen_prompt(testset, j)
            tokens = tokenizer(
                prompt1,
                # padding='max_length',
                # max_length=1400,
                return_tensors='pt'
            ).input_ids
            # prompt1 += choices[int(testset[j]['label']) - 1]
            j += 1
        test_set.append(tokens[:, :max_len])
    return test_set


def eval_siqa(model, tokenizer, test_set):
    ans = 0
    from tqdm import tqdm

    choice_tokens = []
    for choice in choices:
        choice_tokens.append(tokenizer(choice).input_ids[-1])
        
    for item in tqdm(test_set):
        [inp, lab] = item
        result = model.forward(input_ids=inp.cuda(0)).logits[0, -1, :]
        result = torch.max(torch.tensor([
            result[choice_tokens[0]], result[choice_tokens[1]], result[choice_tokens[2]]
        ]), dim=-1)[1].item()
        if result == (int(lab) - 1):
            ans += 1


    print('siqa')
    print()
    print(ans / len(test_set))
    return ans / len(test_set)

if __name__ == '__main__':
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained('/mnt/data/public/models/7B_hf/', use_fast=False)


"""



python lib/dataset/boolq.py




"""


