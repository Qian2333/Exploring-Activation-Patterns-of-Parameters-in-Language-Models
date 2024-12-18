import time
import heapq
import torch
import torch.nn as nn
from .sparsegpt import SparseGPT
from .layerwrapper import WrappedGPT
# from lib.dataset.wikic4 import get_loaders
from lib.dataset import get_loaders
import torch.optim as optim

from .ablate import AblateGPT

def find_layers(module, layers=[nn.Linear], name=''):
    """
    Recursively find the layers of a certain type in a module.

    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        name (str): Name of the module.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res

def check_sparsity(model):
    use_cache = model.config.use_cache
    model.config.use_cache = False

    layers = model.model.layers
    count = 0
    total_params = 0
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        sub_count = 0
        sub_params = 0
        for name in subset:
            W = subset[name].weight.data
            count += (W==0).sum().item()
            total_params += W.numel()

            sub_count += (W==0).sum().item()
            sub_params += W.numel()

        print(f"layer {i} sparsity {float(sub_count)/sub_params:.6f}")

    model.config.use_cache = use_cache
    return float(count)/total_params


def prepare_calibration_input(model, dataloader, device, nsample=128):
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    # dev = model.hf_device_map["model.embed_tokens"]
    if "model.embed_tokens" in model.hf_device_map:
        device = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((nsample, model.seqlen, model.config.hidden_size), dtype=dtype, device=device)
    inps.requires_grad = False
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch.to(device))
        except ValueError:
            pass
    layers[0] = layers[0].module

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']
    model.config.use_cache = use_cache

    return inps, outs, attention_mask, position_ids


def prepare_grad_input(model, dataloader, device):
    # use_cache = model.config.use_cache
    # model.config.use_cache = False
    # layers = model.model.layers
    dtype = next(iter(model.parameters())).dtype

    # dev = model.hf_device_map["model.embed_tokens"]
    if "model.embed_tokens" in model.hf_device_map:
        device = model.hf_device_map["model.embed_tokens"]

    inps = torch.zeros((128, model.seqlen), dtype=dtype, device=device)
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise KeyError
    model = Catcher(model)
    for batch in dataloader:
        try:
            model(batch[0].to(device))
        except KeyError:
            pass

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    model = model.module
    return inps, outs, attention_mask, position_ids


def return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before):
    thres_cumsum = sum_before * alpha
    sort_mask = tmp_metric <= thres_cumsum.reshape((-1,1))
    thres = torch.gather(sort_res[0], dim=1, index=sort_mask.sum(dim=1, keepdims=True)-1)
    W_mask = (W_metric <= thres)
    cur_sparsity = (W_mask==True).sum() / W_mask.numel()
    return W_mask, cur_sparsity


def prune_magnitude(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    layers = model.model.layers

    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        for name in subset:
            W = subset[name].weight.data
            W_metric = torch.abs(W)
            if prune_n != 0:
                W_mask = (torch.zeros_like(W)==1)
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:,ii:(ii+prune_m)].float()
                        W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
            else:
                thresh = torch.sort(W_metric.flatten().cuda())[0][int(W.numel()*args.sparsity_ratio)].cpu()
                W_mask = (W_metric<=thresh)

            W[W_mask] = 0


def get_gradient_tensors_by_modules(net: torch.nn.Module, opt, y, modules):
    import copy
    opt.zero_grad()
    y.backward(retain_graph=True)
    # y.backward()
    ans_ls = {}

    for name, par in net.named_parameters():
        # print(name)
        if name[:-7] not in modules:
            continue
        # print(name)
        # continue
        if par.grad is None:
            continue
        if par.data is None:
            continue
        tmp = copy.deepcopy(par.grad * par.data)
        ans_ls[name[:-7]] = tmp.cuda(1)
    # exit(0)
    return ans_ls


def prune_grad(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    use_cache = model.config.use_cache
    model.config.use_cache = False
    optimizer1 = optim.SGD(model.parameters(), lr=1e-5)

    print("loading calibdation data")
    dataloader, _ = get_loaders("c4", nsamples=args.nsamples, seed=args.seed, seqlen=model.seqlen, tokenizer=tokenizer)
    print("dataset loading complete")
    inps, outs, attention_mask, position_ids = prepare_grad_input(model, dataloader, device)

    # print(find_layers(model))
    # # print(position_ids[0])
    # exit(0)
    modules = find_layers(model)
    module_metric = {}
    # print(modules)
    for i in range(args.nsamples):
        # print(inps[i].unsqueeze(0).shape)
        # exit(0)
        outputs1 = model(inps[i].unsqueeze(0)[:, :512].to(torch.int))
        tr1 = get_gradient_tensors_by_modules(model, optimizer1, torch.max(outputs1.logits, dim=-1)[0][0, -1], modules)
        for key in tr1.keys():
            if key not in module_metric:
                module_metric[key] = torch.abs(tr1[key])
            else:
                module_metric[key] += torch.abs(tr1[key])
    # exit(0)
    print('gradient calculated')

    for name in modules:
        if name not in module_metric:
            continue

        print(f"pruning {name}")
        W_metric = module_metric[name].to(model.device)

        W_mask = (torch.zeros_like(W_metric) == 1)

        sort_res = torch.sort(W_metric, dim=-1, stable=True)

        # print(sort_res)
        # print(W_metric.shape)
        indices = sort_res[1][:, :int(W_metric.shape[1] * args.sparsity_ratio)]
        W_mask.scatter_(1, indices, True)

        modules[name].weight.data[W_mask] = 0  ## set weights to zero

    # layers = model.model.layers
    # for i in range(len(layers)):
    #     layer = layers[i]
    #     subset = find_layers(layer)
    #
    #     wrapped_layers = {}
    #     for name in subset:
    #         wrapped_layers[name] = WrappedGPT(subset[name])
    #
    #     def add_batch(name):
    #         def tmp(_, inp, out):
    #             wrapped_layers[name].add_batch(inp[0].data, out.data)
    #         return tmp
    #
    #     handles = []
    #     for name in wrapped_layers:
    #         handles.append(subset[name].register_forward_hook(add_batch(name)))
    #     for j in range(args.nsamples):
    #         with torch.no_grad():
    #             outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
    #     for h in handles:
    #         h.remove()
    #
    #     for name in subset:
    #         print(f"pruning layer {i} name {name}")
    #         W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))
    #
    #         W_mask = (torch.zeros_like(W_metric) == 1)  ## initialize a mask to be all False
    #         sort_res = torch.sort(W_metric, dim=-1, stable=True)
    #
    #         # unstructured pruning
    #         indices = sort_res[1][:,:int(W_metric.shape[1]*args.sparsity_ratio)]
    #         W_mask.scatter_(1, indices, True)
    #
    #         subset[name].weight.data[W_mask] = 0  ## set weights to zero
    model.config.use_cache = use_cache
    torch.cuda.empty_cache()


def prune_wanda_pro(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    use_cache = model.config.use_cache
    model.config.use_cache = False

    print("loading calibdation data")
    dataloader, _ = get_loaders(
        args.jz_set, 
        nsamples=args.nsamples,
        seed=args.seed,
        seqlen=model.seqlen,
        tokenizer=tokenizer
    )
    # for batch in dataloader:
    #     print(batch.shape)
    #     exit(0)
    print("dataset loading complete")
    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device, args.nsamples)

    layers = model.model.layers

    if args.model == 'llama2_7b':
        sparsitys = [0.5698316177002389, 0.5366455371052089, 0.4801830903953581, 0.46609832638533166, 0.4642484734306026, 0.4663229151253212, 0.4691464178839526, 0.4690636397602521, 0.4688300186796718, 0.46909109112168523, 0.4713240259398702, 0.4779113263536811, 0.48013479483463084, 0.4793389111181827, 0.47987771311883, 0.47705897386617346, 0.47754161427489683, 0.47912675012624306, 0.4847081444093412, 0.4887681433019765, 0.48853670621310913, 0.4994581736685586, 0.5027891643259319, 0.5120923465200711, 0.5152169592529826, 0.5238505055675504, 0.5270248713793841, 0.536286754991283, 0.5422504941585843, 0.5575063421082826, 0.567225143813238, 0.5725110130695745]
    elif args.model == 'llama2_13b':
        sparsitys = [0.5496839541459442, 0.5462058660686806, 0.5254088296933362, 0.5103784145741852, 0.49690074214576985, 0.49029434235152725, 0.47165090770824786, 0.46026578361601755, 0.456084858765523, 0.44757057577440074, 0.4480078221033666, 0.45414606048567296, 0.4562776680787374, 0.4733658841683403, 0.47195693034493413, 0.4733112882448456, 0.47846227595361546, 0.483178835602938, 0.48966128518843755, 0.4914407942503428, 0.4910600034791848, 0.49206278768506095, 0.5038792150056969, 0.508173991490923, 0.5072850417178432, 0.5111552154465175, 0.512554440051483, 0.5159336132733139, 0.5150889614997116, 0.5162577594276856, 0.5176487680783858, 0.5181452342124112, 0.5214010539769939, 0.5213494245102243, 0.5258986958648709, 0.5264208396364602, 0.527415000585438, 0.530218575549759, 0.5319754289982663, 0.5318228302449084]
    elif args.model == 'llama3_8b':
        sparsitys = [0.5251140131898693, 0.46538353507267793, 0.44301298888365864, 0.4084288551055355, 0.4229464157203041, 0.44224612198265156, 0.44953005189737827, 0.4689167567291532, 0.4824337629036096, 0.49144871926595723, 0.5037643428477786, 0.503443532867654, 0.5081049848762448, 0.5122274474816492, 0.5035327768105571, 0.49460698788271, 0.49526388784282227, 0.48707527021165736, 0.49838599878116585, 0.5044694810421985, 0.5074621652134816, 0.5097244924704744, 0.5223810781598084, 0.52608493588962, 0.532347660096728, 0.5343461132740388, 0.5367476556793974, 0.5405655853072835, 0.5416060831514332, 0.5464125704199954, 0.5471474347781415, 0.5448382941643661]
    elif args.model == 'llama_7b':
        sparsitys = [0.5612472387659471, 0.5393453101725671, 0.49066573527532736, 0.4771049746072197, 0.46893333360890443, 0.4652506616358705, 0.46372459099719565, 0.46377408110366325, 0.4619061813989389, 0.46412044167244865, 0.4647548161499886, 0.4665784978274029, 0.47014135219996894, 0.4753157164964991, 0.47691544369561223, 0.4789296248157857, 0.4775751409724201, 0.4781697084613067, 0.48553466381203575, 0.490726925884562, 0.49190327026311426, 0.5042673557321183, 0.5020715067942109, 0.5187214210269638, 0.5213564474684745, 0.5246224440028717, 0.5331718543621345, 0.5400316894424102, 0.5507586024869777, 0.5618454513754834, 0.5625239384388445, 0.5680115790527304]

    for i in range(len(layers)):

        sparsity_ratio_i = sparsitys[i]
        layer = layers[i]
        subset = find_layers(layer)

        if f"model.layers.{i}" in model.hf_device_map:   ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()

        for name in subset:
            print(f"pruning layer {i} name {name}")
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))

            W_mask = (torch.zeros_like(W_metric) == 1)  ## initialize a mask to be all False
            
            sort_res = torch.sort(W_metric, dim=-1, stable=True)

            # unstructured pruning
            indices = sort_res[1][:,:int(W_metric.shape[1]*sparsity_ratio_i)]
            W_mask.scatter_(1, indices, True)

            subset[name].weight.data[W_mask] = 0  ## set weights to zero

        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()


def prune_wanda(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    use_cache = model.config.use_cache 
    model.config.use_cache = False 

    print("loading calibdation data")
    dataloader, _ = get_loaders(
        args.jz_set, 
        nsamples=args.nsamples,
        seed=args.seed,
        seqlen=model.seqlen,
        tokenizer=tokenizer
    )
    # dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)
    print("dataset loading complete")
    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device)

    # use_cache = model.config.use_cache
    # model.config.use_cache = False

    # print("loading calibdation data")
    # dataloader, _ = get_loaders(args.jz_set, nsamples=args.nsamples, seed=args.seed, seqlen=model.seqlen, tokenizer=tokenizer)
    
    # print("dataset loading complete")
    # with torch.no_grad():
    #     inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device, args.nsamples)

    layers = model.model.layers
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        if f"model.layers.{i}" in model.hf_device_map:   ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()

        for name in subset:
            print(f"pruning layer {i} name {name}")
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))

            W_mask = (torch.zeros_like(W_metric) == 1)  ## initialize a mask to be all False
            sort_res = torch.sort(W_metric, dim=-1, stable=True)

            indices = sort_res[1][:,:int(W_metric.shape[1]*args.sparsity_ratio)]
            W_mask.scatter_(1, indices, True)

            subset[name].weight.data[W_mask] = 0  ## set weights to zero

        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()


@torch.no_grad()
def prune_sparsegpt(args, model, tokenizer, dev, prune_n=0, prune_m=0):
    ## SparseGPT code available at: https://github.com/IST-DASLab/sparsegpt/tree/f5c25005a61f96a0933ca2f95705a963585aafaa
    print('Starting ...')
    dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    if "model.embed_tokens" in model.hf_device_map:
        dev = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    print('Ready.')

    for i in range(len(layers)):
        layer = layers[i]
        if f"model.layers.{i}" in model.hf_device_map:
            dev = model.hf_device_map[f"model.layers.{i}"]
            print(f"layer {i} device {dev}")
            inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)

        subset = find_layers(layer)

        gpts = {}
        for name in subset:
            gpts[name] = SparseGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()

        for name in gpts:
            print(i, name)
            print('Pruning ...')

            gpts[name].fasterprune(args.sparsity_ratio, prune_n=prune_n, prune_m=prune_m, percdamp=0.01, blocksize=128)
            gpts[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

        layers[i] = layer
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()



@torch.no_grad()
def prune_ablate(args, model, tokenizer, dev, prune_n=0, prune_m=0):
    ## SparseGPT code available at: https://github.com/IST-DASLab/sparsegpt/tree/f5c25005a61f96a0933ca2f95705a963585aafaa
    print('Starting ...')
    dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    if "model.embed_tokens" in model.hf_device_map:
        dev = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    print('Ready.')

    for i in range(len(layers)):
        layer = layers[i]
        if f"model.layers.{i}" in model.hf_device_map:
            dev = model.hf_device_map[f"model.layers.{i}"]
            print(f"layer {i} device {dev}")
            inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)

        subset = find_layers(layer)

        gpts = {}
        for name in subset:
            gpts[name] = AblateGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()

        for name in gpts:
            print(i, name)
            print('Pruning ...')

            if args.prune_method == "ablate_wanda_seq":
                prune_mask = gpts[name].get_wanda_mask(args.sparsity_ratio, prune_n, prune_m)
            elif args.prune_method == "ablate_mag_seq":
                prune_mask = gpts[name].get_mag_mask(args.sparsity_ratio, prune_n, prune_m)
            elif "iter" in args.prune_method:
                prune_mask = None

            gpts[name].fasterprune(args, args.sparsity_ratio, mask=prune_mask, prune_n=prune_n, prune_m=prune_m, percdamp=0.01, blocksize=128)
            gpts[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

        layers[i] = layer
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()