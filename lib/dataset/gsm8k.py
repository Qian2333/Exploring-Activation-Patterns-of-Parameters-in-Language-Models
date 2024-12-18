from datasets import load_dataset
from tqdm import tqdm
import random
import torch
import re

ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
ANSWER_TRIGGER = "The answer is"
INVALID_ANS = "[invalid]"


def extract_answer_from_output(completion):
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    else:
        return INVALID_ANS


def is_correct(model_answer, answer):
    gt_answer = extract_answer_from_output(answer)
    assert gt_answer != INVALID_ANS
    return model_answer == gt_answer




def create_demo_text(n_shot=8, cot_flag=True):
    question, chain, answer = [], [], []
    question.append(
        "There are 15 trees in the grove. "
        "Grove workers will plant trees in the grove today. "
        "After they are done, there will be 21 trees. "
        "How many trees did the grove workers plant today?"
    )
    chain.append(
        "There are 15 trees originally. "
        "Then there were 21 trees after some more were planted. "
        "So there must have been 21 - 15 = 6."
    )
    answer.append("6")

    question.append(
        "If there are 3 cars in the parking lot and 2 more cars arrive, "
        "how many cars are in the parking lot?"
    )
    chain.append("There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5.")
    answer.append("5")

    question.append(
        "Leah had 32 chocolates and her sister had 42. If they ate 35, "
        "how many pieces do they have left in total?"
    )
    chain.append(
        "Originally, Leah had 32 chocolates. "
        "Her sister had 42. So in total they had 32 + 42 = 74. "
        "After eating 35, they had 74 - 35 = 39."
    )
    answer.append("39")

    question.append(
        "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason "
        "has 12 lollipops. How many lollipops did Jason give to Denny?"
    )
    chain.append(
        "Jason started with 20 lollipops. Then he had 12 after giving some "
        "to Denny. So he gave Denny 20 - 12 = 8."
    )
    answer.append("8")

    question.append(
        "Shawn has five toys. For Christmas, he got two toys each from his "
        "mom and dad. How many toys does he have now?"
    )
    chain.append(
        "Shawn started with 5 toys. If he got 2 toys each from his mom and "
        "dad, then that is 4 more toys. 5 + 4 = 9."
    )
    answer.append("9")

    question.append(
        "There were nine computers in the server room. Five more computers "
        "were installed each day, from monday to thursday. "
        "How many computers are now in the server room?"
    )
    chain.append(
        "There were originally 9 computers. For each of 4 days, 5 more "
        "computers were added. So 5 * 4 = 20 computers were added. "
        "9 + 20 is 29."
    )
    answer.append("29")

    question.append(
        "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On "
        "wednesday, he lost 2 more. "
        "How many golf balls did he have at the end of wednesday?"
    )
    chain.append(
        "Michael started with 58 golf balls. After losing 23 on tuesday, "
        "he had 58 - 23 = 35. After losing 2 more, "
        "he had 35 - 2 = 33 golf balls."
    )
    answer.append("33")

    question.append(
        "Olivia has $23. She bought five bagels for $3 each. "
        "How much money does she have left?"
    )
    chain.append(
        "Olivia had 23 dollars. "
        "5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. "
        "So she has 23 - 15 dollars left. 23 - 15 is 8."
    )
    answer.append("8")

    # randomize order of the examples ...
    index_list = list(range(len(question)))
    random.shuffle(index_list)

    # Concatenate demonstration examples ...
    demo_text = ""
    for i in index_list[:n_shot]:
        if cot_flag:
            demo_text += (
                "Q: "
                + question[i]
                + "\nA: "
                + chain[i]
                + " "
                + ANSWER_TRIGGER
                + " "
                + answer[i]
                + ".\n\n"
            )
        else:
            demo_text += (
                "Question: "
                + question[i]
                + "\nAnswer: "
                + ANSWER_TRIGGER
                + " "
                + answer[i]
                + ".\n\n"
            )
    return demo_text


def gen_prompt(dataset, i, shot=5):
    prompt = create_demo_text(n_shot=shot) \
                + "Q: "\
                + dataset[i]['question']\
                + "\nA: "

    return prompt


def get_gsm8k(tokenizer=None, train_set=False, label=False, nsample=None, shot=8, batch_size=1):
    # tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    # testset = load_dataset('piqa', split='validation', trust_remote_code=True)
    if train_set:
        testset = load_dataset('/mnt/data2/Mydatasets/gsm8k', 'main', split='train')
    else:
        testset = load_dataset('/mnt/data2/Mydatasets/gsm8k', 'main', split='test')
    test_set = []
    i_tos = [i for i in range(len(testset['answer']))]
    if nsample:
        random.shuffle(i_tos)
        i_tos = i_tos[:nsample]
    if label:
        for i in i_tos:
            test_set.append([
                tokenizer(
                    gen_prompt(testset, i, shot=shot),
                    # padding='max_length',
                    # max_length=1400,
                    return_tensors='pt'
                ).input_ids,
                testset['answer'][i],
                testset['question'][i]
            ])
        return test_set
    for i in i_tos:
        test_set.append(
            tokenizer(
                gen_prompt(testset, i, shot=shot),
                # padding='max_length',
                # max_length=1400,
                return_tensors='pt'
            ).input_ids
        )
    return test_set


def clean_answer(model_pred):
    model_pred = model_pred.lower()
    preds = model_pred.split(ANSWER_TRIGGER.lower())
    answer_flag = True if len(preds) > 1 else False
    if answer_flag:
        # Pick first answer with flag
        pred = preds[1]
    else:
        # Pick last number without flag
        pred = preds[-1]

    pred = pred.replace(",", "")
    pred = [s for s in re.findall(r"-?\d+\.?\d*", pred)]

    if len(pred) == 0:
        return INVALID_ANS

    if answer_flag:
        # choose the first element in list
        pred = pred[0]
    else:
        # choose the last element in list
        pred = pred[-1]

    # (For arithmetic tasks) if a word ends with period, it will be omitted ...
    if pred[-1] == ".":
        pred = pred[:-1]

    return pred



def generate(model, tokenizer, input_ids, generate_kwargs=dict(max_new_tokens=256, top_p=0.95, temperature=0.8)):
    input_ids = input_ids.cuda()

    output_ids = model.generate(
        input_ids=input_ids, **generate_kwargs
    )
    response = []
    for i in range(output_ids.shape[0]):
        response.append(
            tokenizer.decode(
                output_ids[i][input_ids.shape[1] :],
                skip_special_tokens=True,
                ignore_tokenization_space=True,
            )
        )

    if len(response) > 1:
        return response
    return response[0]


def eval_gsm8k(model, test_set):
    answers = []
    for item in tqdm(test_set):
        [input_ids, label, ques] = item
        model_completion = generate(model, tokenizer, input_ids)
        model_answer = clean_answer(model_completion)
        is_cor = is_correct(model_answer, label)
        answers.append(is_cor)
        print(sum(answers) / len(answers))

    print()
    print()
    print(sum(answers) / len(answers))
    return sum(answers) / len(answers)

if __name__ == '__main__':
    from transformers import AutoTokenizer, AutoModelForCausalLM

    cache_dir = '/data/MyModels/llama3_8b'
    # cache_dir = '/data/MyModels/llama2_70b_hf'
    tokenizer = AutoTokenizer.from_pretrained(cache_dir, use_fast=False)

    model = AutoModelForCausalLM.from_pretrained(
		cache_dir,
		torch_dtype=torch.float16,
		low_cpu_mem_usage=True,
		device_map="auto"
	)

    model.seqlen = model.config.max_position_embeddings

    answers = []
    testset = get_gsm8k(tokenizer=tokenizer, train_set=False, label=True, nsample=100)
    for item in tqdm(testset):
        [input_ids, label, ques] = item
        model_completion = generate(model, tokenizer, input_ids)
        # print(ques)
        # print(label)
        # print('model response')
        # print(model_completion)
        model_answer = clean_answer(model_completion)
        # print(model_answer)
        # exit(0)
        is_cor = is_correct(model_answer, label)
        answers.append(is_cor)
        print(sum(answers) / len(answers))





"""



python lib/dataset/boolq.py


huggingface-cli download --repo-type dataset --resume-download gsm8k --local-dir /data/Mydatasets/gsm8k


"""


