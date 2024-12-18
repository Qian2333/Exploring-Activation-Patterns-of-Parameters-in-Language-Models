from datasets import load_dataset
from tqdm import tqdm
import random
import torch


choices = ['A', 'B', 'C', 'D']

def gen_prompt(dataset, i):
    prompt = "The following are multiple choice to finish a sentence (with answer) about " + \
             dataset[i]['activity_label'] + '.\n\n'
    prompt = prompt + dataset[i]['ctx'] + '...\nHow does the description likely end?\n\n'
    # prompt = dataset['ctx'][i] + '...\nHow does the description likely end?\n\n'
    prompt = prompt + choices[0] + '. ' + dataset[i]['endings'][0] + '\n\n'
    prompt = prompt + choices[1] + '. ' + dataset[i]['endings'][1] + '\n\n'
    prompt = prompt + choices[2] + '. ' + dataset[i]['endings'][2] + '\n\n'
    prompt = prompt + choices[3] + '. ' + dataset[i]['endings'][3] + '\n\n'
    return prompt + 'Answer:'


def get_hellaswag(tokenizer=None, train_set=False, label=False, nsample=None, batch_size=1):
    # tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    # testset = load_dataset('piqa', split='validation', trust_remote_code=True)
    if train_set:
        testset = load_dataset('/mnt/data2/Mydatasets/hellaswag', split='train')
    else:
        testset = load_dataset('/mnt/data2/Mydatasets/hellaswag', split='validation')
    test_set = []
    i_tos = [i for i in range(len(testset['ind']))]
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
                testset[i]['label']
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


def get_hellaswag_calibration(tokenizer=None, train_set=False, label=False, nsample=None, batch_size=1, max_len=2048):

    testset = load_dataset('/mnt/data2/Mydatasets/hellaswag', split='train')
    test_set = []
    i_tos = [i for i in range(len(testset['ind']))]
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



def eval_hellaswag(model, tokenizer, test_set):
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
        if result == (int(lab)):
            ans += 1


    print()
    print()
    print(ans / len(test_set))
    return ans / len(test_set)

if __name__ == '__main__':
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained('/mnt/data/public/models/7B_hf/', use_fast=False)


"""



python lib/dataset/boolq.py



prompt = prompts["Topic without the ending answer"]
prompt.apply(example)

"""


