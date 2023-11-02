import json

import torch

from model.llm_model import ModelArgs, Transformer
from data.tokenizer import Tokenizer
from utils.apply_delta import apply_model_delta_online
from pathlib import Path

def _load_and_redistribute_checkpoint(llama_model_path, model_name):

    with open(Path(llama_model_path) / model_name / 'params.json') as f:
        params = json.load(f)
    tokenizer = Tokenizer(model_path=str(Path(llama_model_path) / 'tokenizer.model'))
    print('Using model path: %s, model_name: %s' % (llama_model_path, model_name))
    if model_name=='7B' or model_name=='llama-2-7b-chat':
        checkpoint = torch.load(llama_model_path + model_name + '/consolidated.00.pth', map_location="cpu")
        return checkpoint, tokenizer, params


    checkpoints = (Path(llama_model_path) / model_name).glob('*.pth')
    checkpoints = sorted(checkpoints)


    loaded = []
    for x in checkpoints:
        print('loading from', x)
        loaded.append(torch.load(x, map_location='cpu'))

    full_state_dict = {}
    split_dims = {}

    def add_weight_with_split_dim(name, dim):
        if dim < 0:  # bcast without split
            full_state_dict[name] = loaded[0][name].clone()
        else:
            full_state_dict[name] = torch.cat([x[name] for x in loaded], dim=dim)
        for x in loaded:
            del x[name]
        split_dims[name] = dim

    add_weight_with_split_dim('tok_embeddings.weight', 1)
    add_weight_with_split_dim('norm.weight', -1)
    add_weight_with_split_dim('output.weight', 0)
    for i in range(params['n_layers']):
        print('gathering layer %d of %d' % (i, params['n_layers']))
        layer_prefix = f'layers.{i}.'
        bcast_names = [
            'attention_norm.weight',
            'ffn_norm.weight',
        ]
        column_parallel_names = [
            'attention.wq.weight',
            'attention.wk.weight',
            'attention.wv.weight',
            'feed_forward.w1.weight',
            'feed_forward.w3.weight',
        ]
        row_parallel_names = [
            'attention.wo.weight',
            'feed_forward.w2.weight',
        ]
        for key in bcast_names:
            add_weight_with_split_dim(layer_prefix + key, -1)
        for key in column_parallel_names:
            add_weight_with_split_dim(layer_prefix + key, 0)
        for key in row_parallel_names:
            add_weight_with_split_dim(layer_prefix + key, 1)

    checkpoint=full_state_dict


    return checkpoint, tokenizer, params


def Llama(args, **kwargs):

    llama_model_path = args.llama_model_path
    model_name = args.llm_model

    checkpoint, tokenizer, params = _load_and_redistribute_checkpoint(llama_model_path, model_name)

    model_args: ModelArgs = ModelArgs(
        max_seq_len=args.max_seq_len,
        max_batch_size=32,
        grad_checkpoint=args.grad_checkpoint,
        do_pretrain=args.do_pretrain,
        **params
    )

    model_args.vocab_size = tokenizer.n_words
    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    llama = Transformer(model_args).to(torch.bfloat16)
    # llama = Transformer(model_args)
    llama.load_state_dict(checkpoint, strict=False)

    if args.use_vicuna:
        apply_model_delta_online(llama,'../data/weights/vicuna_'+args.llm_model)

    total=0.
    trainable_names=[]

    for name, param in llama.named_parameters():
        if "adapter" not in name:
            param.requires_grad = False
        elif args.do_pretrain and ("txt_adapter" in name or "adapter_attn" in name):
            param.requires_grad = False
        # elif not args.do_pretrain and "vis_adapter" in name:
        #     param.requires_grad = False
        else:
            param.requires_grad = True
            param.data = param.data.float()
            total += param.nelement()
            trainable_names.append(name)

    # print(trainable_names)
    torch.set_default_tensor_type(torch.FloatTensor)
    print('Number of trainable params: %.2fM' % (total / 1e6))
    return llama