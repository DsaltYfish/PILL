# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from typing import Optional, Tuple
from dataclasses import dataclass
import math
import random

import torch
from torch import nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
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
    do_pretrain: bool = False

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


class Adapter(nn.Module):
    """Conventional Adapter layer, in which the weights of up and down sampler modules
    are parameters and are optimized."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.config = args
        self.input_dim = args.dim
        self.down_sample_size = args.down_sample_size
        self.activation = get_activation(args.adapter_activater.lower())
        self.down_sampler = nn.Linear(self.input_dim, self.down_sample_size, False)
        self.up_sampler = nn.Linear(self.down_sample_size, self.input_dim, False)

        # nn.init.normal_(self.down_sampler.weight, 0, 0.02)
        nn.init.normal_(self.up_sampler.weight, 0, 0.02)

    def forward(self, x):
        z = self.down_sampler(x)
        z = self.activation(z)
        z = self.up_sampler(z)

        return z


class GLUAdapter(nn.Module):
    """Conventional Adapter layer, in which the weights of up and down sampler modules
    are parameters and are optimized."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.config = args
        self.input_dim = args.dim
        self.down_sample_size = args.down_sample_size
        self.activation = get_activation(args.adapter_activater.lower())
        self.down_sampler = nn.Linear(self.input_dim, self.down_sample_size, False)
        self.down_sampler_gate = nn.Linear(self.input_dim, self.down_sample_size, False)
        self.up_sampler = nn.Linear(self.down_sample_size, self.input_dim, False)
        # self.up_sampler_gate = nn.Linear(self.down_sample_size, self.input_dim, False)

        nn.init.normal_(self.down_sampler.weight, 0, 0.02)
        nn.init.zeros_(self.up_sampler.weight)

    def forward(self, x):
        z = F.silu(self.down_sampler_gate(x))*self.down_sampler(x)
        # z = self.activation(z)
        z = self.up_sampler(z)
        # z2 = self.up_sampler_gate(z).sigmoid()
        return z


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.args = args

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

        self.vis_adapter_gate = nn.Linear(args.dim, args.n_heads, False)
        nn.init.zeros_(self.vis_adapter_gate.weight)
        # self.attn_adapter_q = nn.Linear(args.dim, 32, False)
        # self.attn_adapter_k = nn.Linear(args.dim, 32, False)

        # self.attn_adapter_bias = nn.Parameter(torch.empty([1]))

        # nn.init.normal_(self.attn_adapter_q.weight, 0, 0.02)
        # nn.init.normal_(self.attn_adapter_k.weight, 0, 0.02)
        # nn.init.constant_(self.attn_adapter_bias.data, 0)

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
                image_position, 
                ):

        bsz, seqlen, _ = x.shape
        # xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        # xq, xk, xv, xg = self.wq(x), self.wk(x), self.wv(x), self.vis_adapter_gate(x.detach()).tanh() #是否要在这里截断梯度？

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_heads, self.head_dim)

        if not self.args.do_pretrain:
            xg = self.vis_adapter_gate(x).tanh()
            xg = xg.view(bsz, seqlen, self.n_local_heads, -1)
            image_position = image_position.view(bsz, seqlen, self.n_local_heads, self.head_dim)
            xv = torch.where(image_position, xv * xg, xv)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        keys = xk
        values = xv
        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        # xq_v = xq[:, :, :32]
        # xq_t = xq[:, :, 32:]

        # xk_v = keys[:, :, :32]
        # xk_t = keys[:, :, 32:]

        if self.cache_enabled:
            if self.cache_k is None:
                assert start_pos == 0
                self.cache_k, self.cache_v = keys, values
            else:
                assert self.cache_k.size(2) >= start_pos
                self.cache_k = torch.cat([self.cache_k[:, :, :start_pos], keys], dim=2)
                self.cache_v = torch.cat([self.cache_v[:, :, :start_pos], values], dim=2)
                keys, values = self.cache_k, self.cache_v

        # scores_v = torch.matmul(xq_v, keys.transpose(2, 3)) / math.sqrt(self.head_dim)

        # scores_tv = torch.matmul(xq_t, xk_v.transpose(2, 3)) / math.sqrt(self.head_dim)
        # scores_tt = torch.matmul(xq_t, xk_t.transpose(2, 3)) / math.sqrt(self.head_dim)

        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)

        if mask is not None:
            scores = scores + mask
            # scores_new = scores.clone()
            # scores_top2 = scores[:, :, 36:, 4:36].topk(2).values
            # modal_imp_attn = 1 - torch.tanh((scores_top2[:, :, :, 0] - scores_top2[:, :, :, 1]))[:, :, :, None]
            # # modal_imp_attn = 1 - torch.tanh((scores[:, :, 36:, 4:36].max(-1).values - scores[:, :, 36:, 4:36].mean(-1)))[:, :, :, None]
            # scores_new[:, :, 36:, 4:36] -= modal_imp_attn * 100. # 如果让模型学习到所有vis token的注意力都很高，这样减掉会不会就没用了
            # scores_new[:, :, 36:, 4:36] = modal_score
            # modal_score = scores[:, :, 36:, 4:36]
            # modal_imp_attn = 1 - torch.tanh((scores[:, :, 36:, 4:36].std(-1) / scores[:, :, 36:, 4:36].mean(-1)) ** 2)[:, :, :, None]
            # modal_score = modal_score - modal_imp_attn * modal_score.max(-1, keepdim=True)[0]
            # scores[:, :, 36:, 4:36] = modal_score
            # scores[:, :, 36:, 4:36] = scores[:, :, 36:, 4:36] + adaptive_mask[:, None, 36:, 4:36]
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        # scores = F.softmax(scores_new.float(), dim=-1).type_as(xq)
        # scores_new = scores.clone()
        # scores_canvas = torch.ones_like(scores)
        # scores_new[:, :, 36:, 4:36] *= torch.tanh((scores[:, :, 36:, 4:36].std(-1) / scores[:, :, 36:, 4:36].mean(-1)) ** 2)[:, :, :, None]
        # scores_new[:, :, 36:, 4:36] *= 0
        # scores = scores * scores_canvas
        # scores = scores * scores_canvas[:, None]
            
        output_l = torch.matmul(scores, values)
        output_l = output_l.transpose(
            1, 2
        ).contiguous().view(bsz, seqlen, -1)

        # modal_attn = None

        # if not self.cache_enabled:
            # modal_attn = scores[:, :, 36:, 4:36].sum(1)

            # imp_attn = (modal_attn.mean([1, 2]) / modal_attn.std([1, 2])) ** 2
            # imp_attn_label = torch.zeros_like(imp_attn)
            # imp_loss = F.l1_loss(imp_attn, imp_attn_label)

            # if previous_attn is not None:
            #     sad_loss = F.mse_loss(previous_attn, modal_attn.detach())
            #     loss = sad_loss + 0.1 * imp_loss
            #     loss = sad_loss

        #     vis_token_attn = modal_attn.amax((2, 3))
        #     vis_token_label = torch.ones_like(vis_token_attn)
        #     vis_token_loss = F.mse_loss(vis_token_attn, vis_token_label)

        #     loss = txt_token_loss + vis_token_loss

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
        # self.vis_adapter_ffn = GLUAdapter(args)
        self.txt_adapter_ffn = GLUAdapter(args)
        # self.reverse_path = ReversePath(args)
        # self.adapter_ffn = ParallelAdapter(args, if_add=False, mode='txt')
        self.adapter_attn = GLUAdapter(args)

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
        h = self.attention.forward(x_a+x_norm, start_pos, freqs_cis, mask, image_position) + x
        # h = self.attention.forward(self.attention_norm(x), start_pos, freqs_cis, mask, image_position) + x
        # h += x
        h_norm = self.ffn_norm(h)
        # h_a = torch.where(image_position, self.adapter_ffn_v(h_norm)+h_norm, h_norm)
        # h_v, h_l = self.vis_adapter_ffn(h_norm), self.txt_adapter_ffn(h_norm)
        h_l = self.txt_adapter_ffn(h_norm)
        # h_a = torch.where(image_position, h_v, h_l)

        h_a = h_l
        h_f = self.feed_forward.forward(h_norm + h_a) + h

        return h_f
        # return h_f + h


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

        self.vis_adapter_proj = nn.Linear(
            self.visual_feature_extractor.Qformer.config.hidden_size, params.dim, False
        )

        # self.adapter_learnable_token = nn.Parameter(torch.zeros([32, params.dim]))

        # self.adapter_proj_norm = RMSNorm(self.visual_feature_extractor.Qformer.config.hidden_size, eps=params.norm_eps)

        nn.init.normal_(self.vis_adapter_proj.weight, 0, 0.02)

        # self.adapter_proj = AdapterMLP(params)

    # def insert_image_embeds(self,examples,labels, image_embeds,prefix_img,prefix_nonimg,img_indicators,img_position):
    #     _bsz, seqlen,dim = examples.shape
    #     new_examples=[]
    #     new_labels=[]
    #     image_positions=[]
    #     for i, (example,label) in enumerate(zip(examples,labels)):
    #         image_position = torch.zeros([seqlen, dim], dtype=torch.bool)
    #         new_example=torch.cat([example[:img_position],prefix_img,image_embeds[i],example[img_position:]],0)
    #         new_label=torch.cat([label[:img_position],
    #                             torch.zeros(prefix_img.shape[0]+image_embeds.shape[1]).to(examples.device).type_as(labels),
    #                             label[img_position:]])
    #         new_example = new_example[:seqlen]
    #         new_label = new_label[:seqlen]
    #         image_position[img_position+prefix_img.shape[0]:img_position+prefix_img.shape[0]+image_embeds.shape[1]] = True
    #         new_examples.append(new_example.unsqueeze(0))
    #         new_labels.append(new_label.unsqueeze(0))
    #         image_positions.append(image_position.unsqueeze(0))
    #     new_examples = torch.cat(new_examples, 0)
    #     new_labels = torch.cat(new_labels, 0)
    #     image_positions = torch.cat(image_positions, 0)
    #     image_positions = image_positions.to(new_examples.device)
    #     return new_examples,new_labels,image_positions

    def insert_image_embeds(self,examples,labels,image_embeds,prefix_img,prefix_nonimg,img_indicators,img_position):
        _bsz, seqlen,dim = examples.shape
        new_examples=[]
        new_labels=[]
        image_positions=[]
        for i, (example,label) in enumerate(zip(examples,labels)):
            image_position = torch.zeros([seqlen, dim], dtype=torch.bool)
            if img_indicators[i]>0.:
                new_example=torch.cat([example[:img_position],prefix_img,image_embeds[i],example[img_position:]],0)
                new_label=torch.cat([label[:img_position],
                                     torch.zeros(prefix_img.shape[0]+image_embeds.shape[1]).to(examples.device).type_as(labels),
                                     label[img_position:]])
                new_example = new_example[:seqlen]
                new_label = new_label[:seqlen]
                image_position[img_position+prefix_img.shape[0]:img_position+prefix_img.shape[0]+image_embeds.shape[1]] = True
            else:
                new_example=torch.cat([example[:img_position],prefix_nonimg,example[img_position:]],0)
                new_label=torch.cat([label[:img_position],
                                     torch.zeros(prefix_nonimg.shape[0]).to(examples.device).type_as(labels),
                                     label[img_position:]])
                new_example = new_example[:seqlen]
                new_label = new_label[:seqlen]
            new_examples.append(new_example.unsqueeze(0))
            new_labels.append(new_label.unsqueeze(0))
            image_positions.append(image_position.unsqueeze(0))
        new_examples = torch.cat(new_examples, 0)
        new_labels = torch.cat(new_labels, 0)
        image_positions = torch.cat(image_positions, 0)
        image_positions = image_positions.to(new_examples.device)
        return new_examples,new_labels,image_positions

    def forward(self, examples, labels,example_mask, images=None, prefix_img=None, prefix_nonimg=None,img_indicators=None, img_position=None):

        # print(images.dtype)
        _bsz, seqlen = examples.shape

        image_embeds=self.visual_feature_extractor.forward_image(images)
        # indices = torch.randperm(32).to(image_embeds.device)
        # image_embeds_shuffled = image_embeds.index_select(1, indices)
        # image_embeds=self.adapter_proj_norm(image_embeds)

        image_embeds=self.vis_adapter_proj(image_embeds)

        # image_embeds = self.adapter_proj_norm(image_embeds)

        # print(img_indicators)
        # if isinstance(img_indicators,list):
        #     img_indicators = torch.Tensor(img_indicators).to(image_embeds.device).long()
        # modality_embed=self.adapter_modality_embedding(img_indicators.unsqueeze(1))


        examples = self.tok_embeddings(examples)
        prefix_img=self.tok_embeddings(prefix_img.unsqueeze(0)).squeeze(0)
        prefix_nonimg=self.tok_embeddings(prefix_nonimg.unsqueeze(0)).squeeze(0)


        # h,labels=self.insert_image_embeds(examples,labels,image_embeds,prefix_img,prefix_nonimg,img_indicators)
        h,labels,image_positions=self.insert_image_embeds(examples,labels, image_embeds,prefix_img,prefix_nonimg,img_indicators,img_position)


        freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = freqs_cis[:seqlen]
        mask = None
        mask = torch.full((1, 1, seqlen, seqlen), float("-inf"), device=h.device)
        mask = torch.triu(mask, diagonal=0 + 1).type_as(h)
        # mask[:, :, img_position+prefix_img.shape[0]:img_position+prefix_img.shape[0]+image_embeds.shape[1], :prefix_img.shape[0]+img_position] = float("-inf")

        start_pos = 0
        ga_loss = torch.tensor([0.]).to(h.device)
        attn = None
        for idx, layer in enumerate(self.layers[:-1]):
            h = layer(h, start_pos, freqs_cis, mask, image_positions)

        if self.params.do_pretrain:
            image_positions = ~image_positions
        h = self.layers[-1](h, start_pos, freqs_cis, mask, image_positions)

        h = self.norm(h)
        output = self.output(h)
        output = output[:, :-1, :].reshape(-1, self.vocab_size)
        labels = labels[:, 1:].flatten()

        c_loss = self.criterion(output, labels)
        
        return c_loss, ga_loss * 0.01


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