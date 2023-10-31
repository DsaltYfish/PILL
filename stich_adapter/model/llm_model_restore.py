# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from typing import Optional, Tuple
from dataclasses import dataclass
import math
import random

import torch
from torch import nn
import torch.nn.functional as F
from transformers.activations import get_activation
# from transformers.modeling_utils import apply_chunking_to_forward
from model.blip2_qformer import Blip2Qformer

from torch.nn import Embedding, Linear
import torch
import pdb
@dataclass
class ModelArgs:
    dim: int = 512
    n_layers: int = 8
    n_heads: int = 8
    vocab_size: int = 32000  # defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    norm_eps: float = 1e-5

    max_batch_size: int = 32
    max_seq_len: int = 2048

    adapter_activater: str = 'gelu'
    down_sample_size: int = 32
    learned_quaries: int = 32
    grad_checkpoint: bool = True

    reverse_prob: float = 0.1


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        if x is None:
            return x
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class ParallelAdapter(nn.Module):
    """Conventional Adapter layer, in which the weights of up and down sampler modules
    are parameters and are optimized."""

    def __init__(self, args: ModelArgs, if_add, mode):
        super().__init__()
        self.config = args
        self.input_dim = args.dim
        self.down_sample_size = args.down_sample_size
        self.activation = get_activation(args.adapter_activater.lower())
        self.down_sampler = nn.Linear(self.input_dim, self.down_sample_size, False)
        self.up_sampler = nn.Linear(self.down_sample_size, self.input_dim, False)

        nn.init.normal_(self.down_sampler.weight, 0, 0.02)
        nn.init.normal_(self.up_sampler.weight, 0, 0.02)

    def forward(self, x):
        z = self.down_sampler(x)
        # z = self.activation(z)
        z = self.up_sampler(z)
        return z / math.sqrt(self.down_sample_size)


def gumbel_sigmoid_sub(x, tau=1e-12, training=True):
    if not training:
        return (x / tau).sigmoid()

    y = x.sigmoid()

    g1 = -torch.empty_like(x).exponential_().log()
    y_hard = ((x + g1 - g1) / tau).sigmoid()

    y_hard = (y_hard - y).detach() + y
    return y_hard
        

class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.n_local_heads = args.n_heads
        self.head_dim = args.dim // args.n_heads

        self.wq = Linear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False
        )
        self.wk = Linear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False
        )
        self.wv = Linear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False
        )
        self.wo = Linear(
            args.n_heads * self.head_dim,
            args.dim,
            bias=False
        )

        self.cache_enabled = False
        self.cache_k, self.cache_v = None, None

    def enable_cache(self):
        self.cache_enabled = True

    def disable_cache(self):
        self.cache_enabled = False
        self.cache_k, self.cache_v = None, None

    def forward(self, 
                x: torch.Tensor, 
                start_pos: int, 
                freqs_cis: torch.Tensor, 
                mask: Optional[torch.Tensor], 
                ):

        bsz, seqlen, _ = x.shape

        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)


        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_heads, self.head_dim)    

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        keys = xk
        values = xv
        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        if self.cache_enabled:
            if self.cache_k is None:
                assert start_pos == 0
                self.cache_k, self.cache_v = keys, values
            else:
                assert self.cache_k.size(2) >= start_pos
                self.cache_k = torch.cat([self.cache_k[:, :, :start_pos], keys], dim=2)
                self.cache_v = torch.cat([self.cache_v[:, :, :start_pos], values], dim=2)
                keys, values = self.cache_k, self.cache_v



        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)

        if mask is not None:
            scores = scores + mask
            
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            
        output_l = torch.matmul(scores, values)
        output_l = output_l.transpose(
            1, 2
        ).contiguous().view(bsz, seqlen, -1)

        return self.wo(output_l)


class FeedForward(nn.Module):
    def __init__(
        self,
        args: ModelArgs
    ):
        super().__init__()
        hidden_dim = int(8 * args.dim / 3)
        hidden_dim = args.multiple_of * ((hidden_dim + args.multiple_of - 1) // args.multiple_of)

        self.w1 = Linear(
            args.dim, hidden_dim, bias=False
        )
        self.w2 = Linear(
            hidden_dim, args.dim, bias=False
        )
        self.w3 = Linear(
            args.dim, hidden_dim, bias=False
        )

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(args)
        self.layer_id = layer_id #后续的CSP Adapter
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)
        # self.adapter_attn_v = ParallelAdapter(args, if_add=True, mode='vis')
        # self.adapter_attn_l = ParallelAdapter(args, if_add=True, mode='txt')
        self.adapter_ffn_v = ParallelAdapter(args, if_add=False, mode='vis')
        self.adapter_ffn_l = ParallelAdapter(args, if_add=False, mode='txt')
        # self.reverse_path = ReversePath(args)
        # self.adapter_ffn = ParallelAdapter(args, if_add=False, mode='txt')
        self.adapter_attn = ParallelAdapter(args, if_add=True, mode='txt')

    def forward(self, x: torch.Tensor, 
                start_pos: int, 
                freqs_cis: torch.Tensor, 
                mask: Optional[torch.Tensor], 
                image_position,
                ):
        x_norm = self.attention_norm(x)
        # x_a = torch.where(image_position, self.adapter_attn_v(x_norm), self.adapter_attn_l(x_norm))
        # loss = F.mse_loss(x_v, x_l)

        x_a = self.adapter_attn(x_norm)
        h = self.attention.forward(x_a+x_norm, start_pos, freqs_cis, mask) + x
        # h += x
        h_norm = self.ffn_norm(h)
        h_v, h_l = self.adapter_ffn_v(h_norm), self.adapter_ffn_l(h_norm)
        # h_l = self.adapter_ffn_l(h_norm)

        h_a = torch.where(image_position, h_v, h_l)
        # h_a = self.adapter_ffn(h_norm)
        # h_a = h_l
        h_f = self.feed_forward.forward(h_norm) + h + h_a

        return h_f
        # return h_f + h


class AdapterMLP(nn.Module):
    """ Pytorch Implemention of RepAdapter for 1d tensor"""

    def __init__(
            self,
            args,
            in_features=768,
            hidden_dim=32,
            out_features=4096
    ):
        super().__init__()
        self.conv_A = nn.Linear(in_features,hidden_dim,False)
        self.conv_B = nn.Linear(hidden_dim, out_features,False)
        # self.activation = get_activation(args.adapter_activater.lower())

        nn.init.xavier_uniform_(self.conv_A.weight)
        nn.init.xavier_uniform_(self.conv_B.weight)

    def forward(self, x):
        x=self.conv_B(self.conv_A(x))
        return x


class Transformer(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers
        self.tok_embeddings = Embedding(
            params.vocab_size, params.dim
        )

        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=0)

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))
          

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = Linear(
            params.dim, params.vocab_size, bias=False
        )

        self.freqs_cis = precompute_freqs_cis(
            self.params.dim // self.params.n_heads, self.params.max_seq_len * 2
        )

        self.grad_checkpoint = params.grad_checkpoint

        self.visual_feature_extractor = Blip2Qformer()

        self.adapter_proj_up = nn.Linear(
            self.visual_feature_extractor.Qformer.config.hidden_size, params.dim, False
        )

        self.adapter_proj_down = nn.Linear(
            params.dim, self.visual_feature_extractor.Qformer.config.hidden_size, False
        )

        # self.adapter_proj_norm = RMSNorm(self.visual_feature_extractor.Qformer.config.hidden_size, eps=params.norm_eps)

        # nn.init.normal_(self.adapter_proj.weight, 0, 0.02)

        # self.adapter_proj = AdapterMLP(params)

    def insert_image_embeds(self,examples,labels, image_embeds,prefix_img,prefix_nonimg,img_indicators):
        _bsz, seqlen,dim = examples.shape
        new_examples=[]
        new_labels=[]
        image_positions=[]
        for i, (example,label) in enumerate(zip(examples,labels)):
            image_position = torch.zeros([seqlen, dim], dtype=torch.bool)
            new_example=torch.cat([example[:1],prefix_img,image_embeds[i],example[1:]],0)
            new_label=torch.cat([label[:1],
                                torch.zeros(prefix_img.shape[0]+image_embeds.shape[1]).to(examples.device).type_as(labels),
                                label[1:]])
            new_example = new_example[:seqlen]
            new_label = new_label[:seqlen]
            image_position[1+prefix_img.shape[0]:1+prefix_img.shape[0]+image_embeds.shape[1]] = True
            new_examples.append(new_example.unsqueeze(0))
            new_labels.append(new_label.unsqueeze(0))
            image_positions.append(image_position.unsqueeze(0))
        new_examples = torch.cat(new_examples, 0)
        new_labels = torch.cat(new_labels, 0)
        image_positions = torch.cat(image_positions, 0)
        image_positions = image_positions.to(new_examples.device)
        return new_examples,new_labels,image_positions

    # def insert_image_embeds(self,examples,labels,image_embeds,prefix_img,prefix_nonimg,img_indicators):
    #     _bsz, seqlen,dim = examples.shape
    #     new_examples=[]
    #     new_labels=[]
    #     image_positions=[]
    #     for i, (example,label) in enumerate(zip(examples,labels)):
    #         image_position = torch.zeros([seqlen, dim], dtype=torch.bool)
    #         if img_indicators[i]>0.:
    #             new_example=torch.cat([example[:1],prefix_img,image_embeds[i],example[1:]],0)
    #             new_label=torch.cat([label[:1],
    #                                  torch.zeros(prefix_img.shape[0]+image_embeds.shape[1]).to(examples.device).type_as(labels),
    #                                  label[1:]])
    #             new_example = new_example[:seqlen]
    #             new_label = new_label[:seqlen]
    #             image_position[1+prefix_img.shape[0]:1+prefix_img.shape[0]+image_embeds.shape[1]] = True
    #         else:
    #             new_example=torch.cat([example[:1],prefix_nonimg,example[1:]],0)
    #             new_label=torch.cat([label[:1],
    #                                  torch.zeros(prefix_nonimg.shape[0]).to(examples.device).type_as(labels),
    #                                  label[1:]])
    #             new_example = new_example[:seqlen]
    #             new_label = new_label[:seqlen]
    #         new_examples.append(new_example.unsqueeze(0))
    #         new_labels.append(new_label.unsqueeze(0))
    #         image_positions.append(image_position.unsqueeze(0))
    #     new_examples = torch.cat(new_examples, 0)
    #     new_labels = torch.cat(new_labels, 0)
    #     image_positions = torch.cat(image_positions, 0)
    #     image_positions = image_positions.to(new_examples.device)
    #     return new_examples,new_labels,image_positions

    def forward(self, examples, labels,example_mask, images=None, prefix_img=None, prefix_nonimg=None,img_indicators=None):

        # print(images.dtype)
        _bsz, seqlen = examples.shape

        image_embeds_origin = self.visual_feature_extractor.forward_image(images)
        # image_embeds = self.adapter_proj_norm(image_embeds_origin)
        image_embeds = self.adapter_proj_up(image_embeds_origin)

        examples = self.tok_embeddings(examples)
        prefix_img=self.tok_embeddings(prefix_img.unsqueeze(0)).squeeze(0)
        prefix_nonimg=self.tok_embeddings(prefix_nonimg.unsqueeze(0)).squeeze(0)

        h,labels,image_positions=self.insert_image_embeds(examples,labels, image_embeds,prefix_img,prefix_nonimg,img_indicators)


        freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = freqs_cis[:seqlen]
        mask = None
        mask = torch.full((1, 1, seqlen, seqlen), float("-inf"), device=h.device)
        mask = torch.triu(mask, diagonal=0 + 1).type_as(h)

        start_pos = 0
        ga_loss = torch.tensor([0.]).to(h.device)
        attn = None
        for idx, layer in enumerate(self.layers):
            if self.grad_checkpoint and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                h = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer),
                    h,
                    start_pos,
                    freqs_cis,
                    mask,
                    image_positions,
                    attn
                )
            else:
                h = layer(h, start_pos, freqs_cis, mask, image_positions)

        h = self.norm(h)
        image_embeds_restore = h[:, 4:36, :]
        image_embeds_restore = self.adapter_proj_down(image_embeds_restore)
        restore_loss = F.mse_loss(image_embeds_restore, image_embeds_origin)

        output = self.output(h)
        output = output[:, :-1, :].reshape(-1, self.vocab_size)
        labels = labels[:, 1:].flatten()

        c_loss = self.criterion(output, labels)
        
        return c_loss, 0.1 * restore_loss


    @torch.inference_mode()
    def forward_inference(self, tokens: torch.Tensor, start_pos: int, image_positions):
        _bsz, seqlen,_ = tokens.shape
        # h = self.tok_embeddings(tokens)
        h=tokens
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]
        if seqlen == 1:
            mask = None
        elif start_pos == 0:
            mask = torch.full((1, 1, seqlen, seqlen), float("-inf"), device=tokens.device)
            mask = torch.triu(mask, diagonal=1).type_as(h)
        else:
            raise NotImplementedError()

        for i, layer in enumerate(self.layers):
            h = layer(h, start_pos, freqs_cis,mask, image_positions)

        h = self.norm(h)
        output = self.output(h[:, -1, :])
        return output.float()

    def enable_cache(self):
        for layer in self.layers:
            layer.attention.enable_cache()

    def disable_cache(self):
        for layer in self.layers:
            layer.attention.disable_cache()