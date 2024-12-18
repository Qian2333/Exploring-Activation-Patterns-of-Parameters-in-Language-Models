
from datasets import load_dataset
from tqdm import tqdm
import random
import torch
import torch.nn as nn
#
# prompts = [
#     "Read the following passage and answer the question. \nThe passage: ",
#     "\nThe question: ",
#     "\nPlease answer true or false: "
# ]
# 61%
choices = ['no', 'yes']
prompts = [
    "",
    "\nQuestion: ",
    "?\nAnswer:"
]


def get_batch_prompt(dataset, i, j):
    batch = []
    for k in range(i, j):
        batch.append(
            prompts[0] + dataset[k]['question'] +
            prompts[1] + dataset[k]['passage'] +
            prompts[2]
        )
    return batch


def get_boolq(tokenizer=None, train_set=False, label=False, nsample=None, batch_size=1):
    # tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    if label:
            # max_len = max(max_len, train_set[-1][0].shape[0])
        testset = load_dataset('/mnt/data2/Mydatasets/boolq', split='validation')
        test_set = []
        i_tos = [i for i in range(len(testset['question']))]
        if nsample:
            random.shuffle(i_tos)
            i_tos = i_tos[:nsample]
        for i in tqdm(range(len(i_tos))):
            # j = min(i + batch_size, len(i_tos))
            test_set.append(
                [
                    tokenizer(
                        prompts[0] + testset['passage'][i] +
                        prompts[1] + testset['question'][i] +
                        prompts[2],
                        # padding='max_length',
                        # max_length=1400,
                        return_tensors='pt'
                    ).input_ids,
                    testset['answer'][i]
                ]
            )
            # max_len = max(max_len, test_set[-1][0].shape[0])
        if train_set:
            trainset = load_dataset('/mnt/data2/Mydatasets/boolq', split='train')
            train_set = []
            max_len = 0
            for i in tqdm(range(len(trainset['question']))):
                train_set.append([
                    tokenizer(
                        prompts[0] + testset[i]['passage'] +
                        prompts[1] + testset[i]['question'] +
                        prompts[2],
                        # padding='max_length',
                        # max_length=1400,
                        return_tensors='pt'
                    ).input_ids, trainset['answer'][i]
                ])
            return train_set, test_set
        print('boolq loaded')
        return test_set
        # max_len = max(max_len, train_set[-1][0].shape[0])
    testset = load_dataset('/mnt/data2/Mydatasets/boolq', split='validation')
    test_set = []
    i_tos = [i for i in range(len(testset['question']))]
    if nsample:
        random.shuffle(i_tos)
        i_tos = i_tos[:nsample]
    for i in tqdm(i_tos):
        test_set.append(
                tokenizer(
                    prompts[0] + testset[i]['passage'] +
                    prompts[1] + testset[i]['question'] +
                    prompts[2],
                    # padding='max_length',
                    # max_length=1400,
                    return_tensors='pt'
                ).input_ids
        )
        # max_len = max(max_len, test_set[-1][0].shape[0])
    if train_set:
        trainset = load_dataset('/mnt/data2/Mydatasets/boolq', split='train')
        train_set = []
        max_len = 0
        i_tos = [i for i in range(len(trainset['question']))]
        if nsample:
            random.shuffle(i_tos)
            i_tos = i_tos[:nsample]
        for i in tqdm(i_tos):
            train_set.append(
                tokenizer(
                    prompts[0] + trainset['passage'][i] +
                    prompts[1] + trainset['question'][i] +
                    prompts[2],
                    # padding='max_length',
                    # max_length=1400,
                    return_tensors='pt'
                ).input_ids
            )
        return train_set, test_set
    print('boolq loaded')
    return test_set


def get_boolq_calibration(tokenizer=None, train_set=False, label=False, nsample=None, batch_size=1, max_len=2048):
    # tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    trainset = load_dataset('/mnt/data2/Mydatasets/boolq', split='train')
    train_set = []
    i_tos = [i for i in range(len(trainset['question']))]
    if nsample:
        random.shuffle(i_tos)
        i_tos = i_tos[:nsample]
        j = nsample
    for i in tqdm(i_tos):
        prompt1 = prompts[0] + trainset[i]['passage'] +\
            prompts[1] + trainset[i]['question'] +\
            prompts[2]
        num_count = 0
        # while num_count < 20:
        #     prompt1 += '\n###\n' + prompts[0] + trainset[j]['passage'] +\
        #         prompts[1] + trainset[j]['question'] +\
        #         prompts[2]
        #     j += 1
        #     num_count += 1
        tokens = tokenizer(
            prompt1,
            # padding='max_length',
            # max_length=1400,
            return_tensors='pt'
        ).input_ids
        while tokens.shape[1] < max_len:
            prompt1 += '\n###\n' + prompts[0] + trainset[j]['passage'] +\
                prompts[1] + trainset[j]['question'] +\
                prompts[2]
            tokens = tokenizer(
                prompt1,
                # padding='max_length',
                # max_length=1400,
                return_tensors='pt'
            ).input_ids
            # prompt1 += str(trainset[j]['answer'])
            j += 1
        train_set.append(tokens[:, :max_len])
    return train_set, None


def eval_ppl_boolq(model, tokenizer, nsamples=256):
        # Get input IDs
    test_set, _ = get_boolq_calibration(tokenizer=tokenizer, train_set=False, nsample=nsamples, max_len=1024)

    # List to store negative log likelihoods
    nlls = []
    print(f"nsamples {nsamples}")

    # Loop through each batch
    for i, sentence in enumerate(test_set):
        if i % 50 == 0:
            print(f"sample {i}")

        # Calculate end index
        j = min(i, nsamples)

        # Prepare inputs and move to device
        inputs = sentence.cuda(0)

        # Forward pass through the model
        lm_logits = model(inputs).logits

        # Shift logits and labels for next token prediction
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = inputs[:, 1:]

        # Compute loss
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1)).item()

        # Calculate negative log likelihood
        neg_log_likelihood = loss 

        # Append to list of negative log likelihoods
        nlls.append(neg_log_likelihood)

    # Compute perplexity
    ppl = torch.exp(torch.mean(torch.tensor(nlls)) / nsamples)

    # Empty CUDA cache to save memory
    torch.cuda.empty_cache()

    return ppl.item()




def eval_boolq(model, tokenizer, test_set):
    ans = 0
    from tqdm import tqdm

    choice_tokens = []
    for choice in choices:
        choice_tokens.append(tokenizer(choice).input_ids[-1])
        
    for item in tqdm(test_set):
        # inp, lab = item
        [inp, lab] = item
        result = model.forward(input_ids=inp.cuda(0)).logits[0, -1, :]

        result = torch.max(torch.tensor([
            result[choice_tokens[0]], result[choice_tokens[1]]
        ]), dim=-1)[1].item()
        if result == (int(lab)):
            ans += 1
    print('boolq')
    print()
    print(ans / len(test_set))
    return ans / len(test_set)



if __name__ == '__main__':
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('/mnt/data/public/models/7B_hf/', use_fast=False)

    get_boolq(tokenizer)


"""



python lib/dataset/boolq.py



 
"""


