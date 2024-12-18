import torch
import numpy as np
import random
import torch.nn as nn
import torch.nn.init as init
import copy


def init_model(net):
    for par in list(net.parameters()):
        init.normal(par)
    return net


def get_gradient_tensors_by_modules(net, opt, y, device=torch.device("cuda:0")):
    opt.zero_grad()
    y.backward(retain_graph=True)
    # y.backward()
    ans_ls = []

    for name, par in net.named_parameters():
        # if name[:23] == 'bert.model.embed_tokens':
        #     continue
        if len(par.data.shape) < 2:
            continue
        if max(par.data.shape) > 15000:
            continue
        # print(name, par.data.shape)
        # print(name)
        # continue
        if par.grad is None:
            continue
        if par.data is None:
            continue
        tmp = copy.deepcopy(par.grad.view(-1) * par.data.view(-1))
        ans_ls.append(tmp)
    # exit(0)
    return ans_ls


def get_grad_x_score(net, opt, y, device=torch.device("cuda:0")):
    opt.zero_grad()
    y.backward(retain_graph=True)
    # y.backward()
    ans_ls = []

    for name, par in net.named_parameters():
        if len(par.data.shape) < 2:
            continue
        if max(par.data.shape) > 12000:
            continue
        print(name, par.data.shape)
        # continue
        if par.grad is None:
            continue
        if par.data is None:
            continue
        tmp = copy.deepcopy(par.grad.view(-1) * par.data)
        ans_ls.append(tmp.to(device))
    # exit(0)
    return ans_ls


def prepare_calibration_input1(model, input_ids, device, sqelen, nsample):
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    # dev = model.hf_device_map["model.embed_tokens"]
    if "model.embed_tokens" in model.hf_device_map:
        device = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((nsample, sqelen, model.config.hidden_size), dtype=dtype, device=device)
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
    # print(model)
    # exit(0)
    try:
        model(input_ids.to(device))
    except ValueError:
        pass
    layers[0] = layers[0].module

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']
    model.config.use_cache = use_cache

    return inps, outs, attention_mask, position_ids



from lib.util.prune_ori import get_loaders, prepare_calibration_input, find_layers, WrappedGPT, return_given_alpha


def cal_prune_wanda_score1(args, model, tokenizer, input_ids, device=torch.device("cuda:0")):
    use_cache = model.config.use_cache
    model.config.use_cache = False
    ans_ls = []

    # print("loading calibdation data")
    #
    # print("dataset loading complete")
    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input1(model,
                                                                              input_ids,
                                                                              device,
                                                                              len(input_ids[0]),
                                                                              nsample=1)

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
        for j in range(1):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()

        for name in subset:
            # print(f"pruning layer {i} name {name}")
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))

            # W_mask = (torch.zeros_like(W_metric) == 1)  ## initialize a mask to be all False
            # sort_res = torch.sort(W_metric, dim=-1, stable=True)
            #
            # indices = sort_res[1][:,:int(W_metric.shape[1]*args.sparsity_ratio)]
            # W_mask.scatter_(1, indices, True)
            ans_ls.append(copy.deepcopy(W_metric).cuda(1))

        for j in range(1):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        inps, outs = outs, inps
    return ans_ls


def cal_prune_wanda_score2(args, model, input_ids1, input_ids2, device=torch.device("cuda:0")):
    use_cache = model.config.use_cache
    model.config.use_cache = False
    ans_ls = []

    # print("loading calibdation data")
    #
    # print("dataset loading complete")
    with torch.no_grad():
        inps1, outs1, attention_mask1, position_ids1 = prepare_calibration_input1(model,
                                                                                  input_ids1,
                                                                                  device,
                                                                                  len(input_ids1[0]),
                                                                                  nsample=1)
        inps2, outs2, attention_mask2, position_ids2 = prepare_calibration_input1(model,
                                                                                  input_ids2,
                                                                                  device,
                                                                                  len(input_ids2[0]),
                                                                                  nsample=1)

    layers = model.model.layers
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        if f"model.layers.{i}" in model.hf_device_map:   ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps1, outs1, attention_mask1, position_ids1 = inps1.to(dev), outs1.to(dev), \
                                                           attention_mask1.to(dev), position_ids1.to(dev)

        wrapped_layers1 = {}
        for name in subset:
            wrapped_layers1[name] = WrappedGPT(subset[name])

        def add_batch(model, name):
            def tmp(_, inp, out):
                model[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in wrapped_layers1:
            handles.append(subset[name].register_forward_hook(add_batch(wrapped_layers1, name)))

        for j in range(1):
            with torch.no_grad():
                outs1[j] = layer(inps1[j].unsqueeze(0), attention_mask=attention_mask1, position_ids=position_ids1)[0]


        for h in handles:
            h.remove()

        Ws1 = {}
        for name in subset:
            W_metric1 = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers1[name].scaler_row.reshape((1,-1)))

            Ws1[name] = copy.deepcopy(W_metric1).cuda(1).reshape(-1).to(torch.float64)
            Ws1[name] = Ws1[name] > (torch.sum(Ws1[name]) / len(Ws1[name]))

            # print(Ws1[name])
            # print(torch.sum(Ws1[name]))
            # exit(0)

        for j in range(1):
            with torch.no_grad():
                outs1[j] = layer(inps1[j].unsqueeze(0), attention_mask=attention_mask1, position_ids=position_ids1)[0]
        inps1, outs1 = outs1, inps1





        if f"model.layers.{i}" in model.hf_device_map:   ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps2, outs2, attention_mask2, position_ids2 = inps2.to(dev), outs2.to(dev), \
                                                           attention_mask2.to(dev), position_ids2.to(dev)

        wrapped_layers2 = {}
        for name in subset:
            wrapped_layers2[name] = WrappedGPT(subset[name])

        handles = []
        for name in wrapped_layers2:
            handles.append(subset[name].register_forward_hook(add_batch(wrapped_layers2, name)))

        for j in range(1):
            with torch.no_grad():
                outs2[j] = layer(inps2[j].unsqueeze(0), attention_mask=attention_mask2, position_ids=position_ids2)[0]

        for h in handles:
            h.remove()

        Ws2 = {}
        for name in subset:
            W_metric2 = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers2[name].scaler_row.reshape((1,-1)))

            Ws2[name] = copy.deepcopy(W_metric2).cuda(1).reshape(-1).to(torch.float64)
            Ws2[name] = Ws2[name] > (torch.sum(Ws2[name]) / len(Ws2[name]))

        for j in range(1):
            with torch.no_grad():
                outs2[j] = layer(inps2[j].unsqueeze(0), attention_mask=attention_mask2, position_ids=position_ids2)[0]
        inps2, outs2 = outs2, inps2

        for name in Ws1:
            W_1 = Ws1[name]
            W_2 = Ws2[name]

            ans_ls.append(torch.sum(W_1 * W_1) / torch.sum(W_1 * W_1) / torch.sum(W_2 * W_2))
    return ans_ls
