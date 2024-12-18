from .code_eval import read_problems
from .boolq import get_boolq, eval_boolq, get_boolq_calibration
from .mmlu import get_mmlu, eval_mmlu, get_mmlu_calibration
from .piqa import get_piqa, eval_piqa, get_piqa_calibration
from .siqa import get_siqa, eval_siqa, get_siqa_calibration
from .hellaswag import get_hellaswag, eval_hellaswag, get_hellaswag_calibration
from .wikic4 import get_c4, get_wikitext2
from .gsm8k import get_gsm8k, eval_gsm8k
from .stsb import get_sick, get_stsb

import random


def get_humaneval(tokenizer, nsample=None):
    problems = read_problems()
    test_set = []
    task_ids = list(problems.keys())
    if nsample:
        random.shuffle(task_ids)
        task_ids = task_ids[:nsample]
    for task_id in task_ids:
        test_set.append(
            tokenizer(
                problems[task_id]['prompt'],
                return_tensors='pt'
            ).input_ids
        )
    return test_set


# Load and process c4 dataset
def get_c4_nlabel(nsamples, seed, seqlen, tokenizer):
    from .wikic4 import load_dataset
    # Load train and validation datasets
    traindata = load_dataset('allenai/c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train')

    # Generate samples from training set
    # random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] > seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        trainloader.append(inp)

    return trainloader


# Load and process wikitext2 dataset
def get_wikitext2_nlabel(nsamples, seed, seqlen, tokenizer):
    from .wikic4 import load_dataset
    # Load train and test datasets
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')

    # Encode datasets
    trainenc = tokenizer(" ".join(traindata['text']), return_tensors='pt')

    # Generate samples from training set
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append(inp)
    return trainloader


def get_dataset(name, tokenizer, nsample=None, label=False, shot=8, max_len=2048):
    if name == 'gsm8k':
        return get_gsm8k(tokenizer, label=label, nsample=nsample, shot=shot)
    if name == 'humaneval':
        return get_humaneval(tokenizer, nsample=nsample)
    if name == 'boolq':
        return get_boolq(tokenizer, label=label, nsample=nsample)
        # dataset, _ = get_boolq_calibration(tokenizer, label=label, nsample=nsample, max_len=max_len)
        # return dataset 
    if name == 'mmlu':
        return get_mmlu(tokenizer, label=label, nsample=nsample)
    if name == 'piqa':
        return get_piqa(tokenizer, label=label, nsample=nsample)
    if name == 'siqa':
        return get_siqa(tokenizer, label=label, nsample=nsample)
    if name == 'hellaswag':
        return get_hellaswag(tokenizer, label=label, nsample=nsample)
    if name == 'c4':
        return get_c4_nlabel(nsample, 0, 256, tokenizer)
    if name == 'wikitext2':
        return get_wikitext2_nlabel(nsample, 0, 256, tokenizer)
    if name == 'stsb':
        return get_stsb(tokenizer=tokenizer, nsample=nsample)
    if name == 'sick':
        return get_sick(tokenizer=tokenizer, nsample=nsample)

# Function to select the appropriate loader based on dataset name
def get_loaders(name, nsamples=128, seed=0, seqlen=2048, tokenizer=None):
    if 'wikitext2' in name:
        return get_wikitext2(nsamples, seed, seqlen, tokenizer)
    if "c4" in name:
        return get_c4(nsamples, seed, seqlen, tokenizer)
    if name == 'humaneval':
        return get_humaneval(tokenizer, nsample=nsamples), None
    if name == 'boolq':
        return get_boolq_calibration(tokenizer, train_set=True, nsample=nsamples, max_len=seqlen)
    if name == 'mmlu':
        return get_mmlu_calibration(tokenizer, train_set=True, nsample=nsamples, max_len=seqlen), None
    if name == 'piqa':
        return get_piqa_calibration(tokenizer, train_set=True, nsample=nsamples, max_len=seqlen), None
    if name == 'siqa':
        return get_siqa_calibration(tokenizer, train_set=True, nsample=nsamples, max_len=seqlen), None
    if name == 'hellaswag':
        return get_hellaswag_calibration(tokenizer, train_set=True, nsample=nsamples, max_len=seqlen), None

def eval_one_shot(model, task=['boolq', 'piqa'], tokenizer=None, nsample=None):
    result = {}
    if 'boolq' in task:
        result['boolq'] = eval_boolq(model, tokenizer, get_boolq(tokenizer, label=True, nsample=nsample))
    if 'mmlu' in task:
        result['mmlu'] = eval_mmlu(model, tokenizer, get_mmlu(tokenizer, label=True, nsample=nsample, n_shot=5))
    if 'piqa' in task:
        result['piqa'] = eval_piqa(model, tokenizer, get_piqa(tokenizer, label=True, nsample=nsample))
    if 'siqa' in task:
        result['siqa'] = eval_siqa(model, tokenizer, get_siqa(tokenizer, label=True, nsample=nsample))
    if 'hellaswag' in task:
        result['hellaswag'] = eval_hellaswag(model, tokenizer, get_hellaswag(tokenizer, label=True, nsample=nsample))
    print(result)
    return result


"""

python lib/dataset/__init__.py

"""
