import argparse
import os
import numpy as np
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"
from lib.util.logger import Logger, str_pad
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from importlib.metadata import version
from lib.util.toolbag import setup_seed
from lib.util.prune_ori import prune_wanda, prune_magnitude, prune_sparsegpt, prune_ablate, check_sparsity, \
    find_layers, prune_grad, prune_wanda_pro
from lib.util.eval import eval_ppl, eval_zero_shot

print('torch', version('torch'))
print('transformers', version('transformers'))
print('accelerate', version('accelerate'))
print('# of gpus: ', torch.cuda.device_count())

def get_llm(model_name=None, cache_dir="/mnt/data/public/models/7B_hf/"):
    if cache_dir == '/mnt/data2/wyd/MyModels/qwen7b/':
        model = AutoModelForCausalLM.from_pretrained(
            cache_dir,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="auto",
            trust_remote_code=True
        )
        model.seqlen = 2048
    else:
        model = AutoModelForCausalLM.from_pretrained(
            cache_dir,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="auto"
        )

        model.seqlen = model.config.max_position_embeddings
    print(model.seqlen)
    return model

model_file = {
    'llama2_13b': '/mnt/data2/wyd/MyModels/llama2_13b_hf/',
    'llama2_7b': '/mnt/data2/wyd/MyModels/llama2_7b_hf/' ,
    'qwen_7b': '/mnt/data2/wyd/MyModels/qwen7b/',
    'llama3_8b': '/mnt/data2/wyd/MyModels/llama3_8b'
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='see the cache dir', help='LLaMA model')
    parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')
    parser.add_argument('--nsamples', type=int, default=128, help='Number of calibration samples.')
    parser.add_argument('--sparsity_ratio', type=float, default=0, help='Sparsity level')
    parser.add_argument("--sparsity_type", type=str, choices=["unstructured", "4:8", "2:4"])
    parser.add_argument("--prune_method", type=str, choices=["magnitude", "wanda", "sparsegpt",
                        "ablate_mag_seq", "ablate_wanda_seq", "ablate_mag_iter", "ablate_wanda_iter",
                        "wandap", "search", "grad", "dense"])
    parser.add_argument("--cache_dir", default="llm_weights", type=str )
    parser.add_argument('--use_variant', action="store_true", help="whether to use the wanda variant described in the appendix")
    parser.add_argument('--save', type=str, default="logs/prune", help='Path to save results.')
    parser.add_argument('--save_model', type=str, default=None, help='Path to save the pruned model.')

    parser.add_argument("--jz_set", type=str, default="c4", help="dataset for calibration")
    parser.add_argument("--eval_zero_shot", action="store_true")
    args = parser.parse_args()

    # Setting seeds for reproducibility
    setup_seed(args.seed)
    logger1 = Logger(name='llama_wanda_log', tim=False)

    # Handling n:m sparsity
    prune_n, prune_m = 0, 0
    if args.sparsity_type != "unstructured":
        assert args.sparsity_ratio == 0.5, "sparsity ratio must be 0.5 for structured N:M sparsity"
        prune_n, prune_m = map(int, args.sparsity_type.split(":"))

    print(f"loading llm model {args.model}")
    model = get_llm(args.model, args.cache_dir)
    model.eval()

    # tokenizer = AutoTokenizer.from_pretrained(args.cache_dir, use_fast=False)
    if args.cache_dir == '/mnt/data2/wyd/MyModels/qwen7b/':
        tokenizer = AutoTokenizer.from_pretrained(
            args.cache_dir,
            use_fast=True,
            trust_remote_code=True
        )
    else:
        # tokenizer = AutoTokenizer.from_pretrained('/mnt/data2/wyd/MyModels/llama3_8b', use_fast=True)
        tokenizer = AutoTokenizer.from_pretrained(
            args.cache_dir,
            use_fast=False
        )

    device = torch.device("cuda:0")
    if "30b" in args.model or "65b" in args.model: # for 30b and 65b we use device_map to load onto multiple A6000 GPUs, thus the processing here.
        device = model.hf_device_map["lm_head"]
    print("use device ", device)

    if args.sparsity_ratio != 0:
        print("pruning starts")
        if args.prune_method == "wanda":
            prune_wanda(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        elif args.prune_method == "wandap":
            prune_wanda_pro(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        elif args.prune_method == "magnitude":
            prune_magnitude(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        elif args.prune_method == "sparsegpt":
            prune_sparsegpt(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        elif args.prune_method == "grad":
            prune_grad(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        elif "ablate" in args.prune_method:
            prune_ablate(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)

    ################################################################
    print("*"*30)
    sparsity_ratio = check_sparsity(model)
    print(f"sparsity sanity check {sparsity_ratio:.4f}")
    print("*"*30)
    ################################################################

    # if args.save_model:
    #     model.save_pretrained(args.save_model)
    #     tokenizer.save_pretrained(args.save_model)

    ppl_test_wiki = eval_ppl(args, model, tokenizer, device)
    print(f"wikitext perplexity {ppl_test_wiki}")

    ppl_test = 'todo'
    # from lib.dataset.boolq import eval_ppl_boolq
    # ppl_test = eval_ppl_boolq(model, tokenizer)
    # print(f"boolq perplexity {ppl_test}")

    from lib.dataset import eval_one_shot
    # result = eval_one_shot(model, task=['boolq', 'piqa', 'siqa', 'hellaswag', 'mmlu'],
    #                        tokenizer=tokenizer, nsample=256)
    result = eval_one_shot(model, task=['boolq', 'piqa', 'siqa', 'hellaswag', 'mmlu'],
                           tokenizer=tokenizer, nsample=256)
    logger1.info('pruning method: ' + args.prune_method + '  Calibration set:' + args.jz_set \
                 + '\nmodel cache: ' + args.cache_dir \
                 + '\nwiki_ppl: ' + str(ppl_test_wiki) + '| boolq_ppl: ' + str(ppl_test) + '\nresult: ' + str(result) + '\n')
    logger1.info('----------------------------------------------------------------------------------')


if __name__ == '__main__':
    main()


"""

llama2

dense 79.42% 

wanda 6.9207844734191895 {'boolq': 0.7510703363914373, 'mmlu': 0.33376877857609405}

wandap 
4*0.4+12*0.5+16*0.52492 6.918911    75.47%
0.3 + 0.13/3*3+12*0.5+16*0.52492  6.943212  76.21%
0.35+0.383+0.417+0.45 6.8
[0.3, 0.4, 0.45, 0.45] + 0.13/3*3+12*0.5+16*0.52492  6.952818393707275  {'boolq': 0.7590214067278287, 'mmlu': 0.3187459177008491}

{'boolq': 0.7571865443425077, 'piqa': 0.4967355821545158}




huggingface-cli download --repo-type dataset --resume-download boolq --local-dir ~/wyd/Mydatasets/boolq

"""