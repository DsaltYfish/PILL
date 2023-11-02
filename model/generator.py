# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from typing import List

import torch

from tokenizer import Tokenizer
from model.llm_model import Transformer
from  torch.cuda.amp import autocast

class Generator:
    def __init__(self, model: Transformer, tokenizer: Tokenizer):
        self.model = model
        self.tokenizer = tokenizer


    @torch.inference_mode()
    def generate(
        self,
        prompts: List[str],
        images: torch.Tensor,
        max_gen_len: int,
        temperature: float = 0.8,
        top_p: float = 0.95,
    ) -> List[str]:
        self.model.enable_cache()
        bsz = len(prompts)
        params = self.model.params
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)
        self.model.eval()

        images=images.cuda()
        B, num_seq_images, C, H, W = images.shape
        images = images.reshape(B * num_seq_images, C, H, W)
        self.model = self.model.cuda()

        image_embeds=self.model.visual_feature_extractor.forward_image(images).bfloat16()
        image_embeds=self.model.vis_adapter_proj(image_embeds)

        image_embeds=image_embeds.reshape(B, num_seq_images, 32, self.model.params.dim)

        prompt_tokens=[]
        image_positions=[]
        for i,x in enumerate(prompts):
            token_idx=self.tokenizer.encode(x, bos=True, eos=False)

            prompt_tokens.append(token_idx)


        min_prompt_size = min([len(t) for t in prompt_tokens])
        max_prompt_size = max([len(t) for t in prompt_tokens])

        total_len = min(params.max_seq_len, max_gen_len + max_prompt_size) # 512

        tokens = torch.full((bsz, total_len), 0).cuda().long()
        input_text_mask=torch.zeros_like(tokens).bool()

        for k, t in enumerate(prompt_tokens):
            t=t[:total_len]
            tokens[k, : len(t)] = torch.tensor(t).long()
            input_text_mask[k,:len(t)]=True

        
        image_positions = torch.where(tokens==32000, True, False)
        tok = torch.where(image_positions, 0, tokens)
        examples = self.model.tok_embeddings(tok)

        token_embeds=[]

        for b, (example, image_embed, image_position) in enumerate(zip(examples, image_embeds, image_positions)):
            indices = torch.where(image_position)[0].tolist()
            new_example = example.clone()
            for image_id, indice in enumerate(indices[::32]):
                new_example[indice:indice+32] = image_embed[image_id]
            
            token_embeds.append(new_example.unsqueeze(0))

        token_embeds = torch.cat(token_embeds, 0)

        start_pos = min_prompt_size
        prev_pos = 0
        for cur_pos in range(start_pos, total_len):

            if prev_pos==0:
                h=torch.cat([token_embeds[:,prev_pos:cur_pos]],1)
            else:
                h=token_embeds[:,prev_pos:cur_pos]
            image_position = image_positions[:, prev_pos:cur_pos]
            logits = self.model.forward_inference(h, prev_pos, image_position)
            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits, dim=-1)
            next_token = next_token.reshape(-1)
            # only replace token if prompt has already been generated

            next_token_embeds = torch.where(
                input_text_mask[:, cur_pos,None], token_embeds[:, cur_pos], self.model.tok_embeddings(next_token)
            )
            token_embeds[:,cur_pos]=next_token_embeds

            next_token = torch.where(
                input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )
            tokens[:, cur_pos] = next_token

            prev_pos = cur_pos

        decoded = []
        for i, t in enumerate(tokens.tolist()):
            # cut to max gen len
            t = t[: len(prompt_tokens[i]) + max_gen_len]
            # cut to eos tok if any
            try:
                t = t[: t.index(self.tokenizer.eos_id)]
            except ValueError:
                pass
            decoded.append(self.tokenizer.decode(t))

        self.model.disable_cache()
        return decoded


def sample_top_p(probs, p):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token
