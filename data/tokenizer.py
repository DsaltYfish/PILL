# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from sentencepiece import SentencePieceProcessor
from logging import getLogger
from typing import List
import os
import torch
from sentencepiece import sentencepiece_model_pb2 as sp_pb2_model
from transformers import LlamaTokenizer

logger = getLogger()


class Tokenizer:
    def __init__(self, model_path: str, max_length: int=10):
        # reload tokenizer
        assert os.path.isfile(model_path), model_path
        # self.sp_model = SentencePieceProcessor(model_file=model_path)
        self.sp_model = LlamaTokenizer.from_pretrained(model_path)
        logger.info(f"Reloaded SentencePiece model from {model_path}")

        # BOS / EOS token IDs
        self.n_words: int = self.sp_model.vocab_size # 32000
        # self.bos_id: int = self.sp_model.bos_id # 1
        # self.eos_id: int = self.sp_model.eos_id # 2
        # self.pad_id: int = self.sp_model.pad_id # 0
        self.bos_id: int = 1
        self.eos_id: int = 2
        self.pad_id: int = 0
        self.max_length = max_length

        special_tokens = ['<image_placeholder>']
        self.sp_model.add_tokens(special_tokens)
        
        # self.cls_id: int = self.sp_model.cls_id()
        # logger.info(
        #     f"#words: {self.n_words} - BOS ID: {self.bos_id} - EOS ID: {self.eos_id} - PAD ID: {self.pad_id}"
        # )
        # print(f"#words: {self.n_words} - BOS ID: {self.bos_id} - EOS ID: {self.eos_id} - PAD ID: {self.pad_id}")
        # assert self.sp_model.vocab_size() == self.sp_model.get_piece_size()

    def encode(self, s: str, bos: bool, eos: bool) -> List[int]:
        assert type(s) is str
        t = self.sp_model.encode(s, add_special_tokens=False)
        if bos:
            t = [1] + t
        if eos:
            t = t + [2]

        return t
    
    def tokenize(self, text):
        t = torch.tensor(self.encode(text, bos=True, eos=False), dtype=torch.int64)
        padding = self.max_length - t.shape[0]
        print(padding)
        if padding > 0:
            t = torch.cat((t, torch.zeros(padding, dtype=torch.int64)))
        elif padding < 0:
            t = t[:self.max_length]

        # attn_mask = t.ge(0)
        # t[~attn_mask] = 0
        # attn_mask = attn_mask.float()
        # example: 
        # t = tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0])  
        # attn_mask = tensor([ True,  True,  True,  True,  True,  True,  True,  True,  True,  True, False, False])
        return t

    def decode(self, t: List[int]) -> str:
        return self.sp_model.decode(t)


if __name__ == "__main__":
    tokenizer = Tokenizer("/data/zfy/llama1/weight/pyllama_data/tokenizer.model", 512)
    # tokens = tokenizer.encode("霸王chaji", True, False)
    tokens = tokenizer.tokenize("Please ensure that you are answering the question based on the information presented in the image 0: <image0>图.Question:What is the airline name that appears on the side of the airplane? Answer: Delta\n\nThe image 1: <image1>图 is the primary source of information for answering the questions. Please refer to it carefully when answering question: Which is the train shown in the image? Answer: INTERCITY\n\nQuestions is related to image 2: <image2>图. Please analyze the image and provide the correct answer for the question: What is the first name of the person who owns the copyright? Miguel\n\nPlease refer to image 3: <image3>图 when answering the following questions: What is the name of the ski trail? Answer: Charlie Banks Trail or CHARLIE BANKS\n\nPay close attention to the details in image 4: <image4>图, as they may provide important information for answering the questions. Question:What type of transit is this? Answer: mass\n\nUse the image 5: <image5>图 as a visual aid to help you understand the context and answer the questions accurately. Question:what line is this bus running? Answer: Linea 102\n\nMake sure your answers are based on the information presented in the image 6: <image6>图, and any OCR text associated with it. Question:What brand is the bread? Answer: An Luc Pele\n\nFor each question, use the image 7: <image7>图 as a reference to answer the question: What is the number of the plane? VT-JCJ\n\nThe image 8: <image8>图 is the primary source of information for answering the questions. Please refer to it carefully when answering question: What is the picture called? Answer:")
    print(tokens)
    print(tokenizer.decode(tokens)) 
    
    