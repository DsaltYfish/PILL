# coding=utf-8
# Copyright 2022 Gen Luo. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import  json, re,random
import torch.utils.data as Data
from torchvision.transforms import transforms
import os
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from PIL import Image
from data.base_prompt import *
import torch
from data.tokenizer import Tokenizer
import copy

class ScienceQADataSet(Data.Dataset):
    def __init__(self, args,split,model_path,max_words=512,max_image_feats=1):
        super(ScienceQADataSet, self).__init__()
        self.args = args
        # --------------------------
        # ---- Raw data loading ---
        # --------------------------
        self.problems = json.load(open(os.path.join(args.data_root, 'problems.json')))
        pid_splits = json.load(open(os.path.join(args.data_root, 'pid_splits.json')))
        captions = json.load(open(args.caption_file))["captions"]
        self.image_path=os.path.join(args.data_root,split)
        self.tokenizer = Tokenizer(model_path=model_path + '/tokenizer.model')
        self.max_words = max_words
        self.max_image_feats=max_image_feats
        self.split=split
        for qid in self.problems:
            self.problems[qid]['caption'] = captions[qid] if qid in captions else ""

        self.qids = pid_splits['%s' % (split)]
        # self.qids = self.qids[:100]

        print(f"number of problems in split {split}: {len(self.qids)}\n")
        self.system_message = "[INST]<<SYS>>\nYou are a language vision assistant. Based on the image, question, context, and options, please give an appropriate response, including your choice and reason.\n<</SYS>>\n\n"
        # self.system_message = "[INST]"

        self.transforms=transforms.Compose([transforms.Resize((224, 224), interpolation=Image.BICUBIC),transforms.ToTensor(), transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)])

    def tokenize(self,prompt,answer):
        example=prompt+answer
        # print(prompt)
        prompt=torch.tensor(self.tokenizer.encode(prompt, bos=True, eos=False), dtype=torch.int64)
        example = torch.tensor(self.tokenizer.encode(example, bos=True, eos=True), dtype=torch.int64)

        example = example[:self.max_words]
        label = copy.deepcopy(example)
        label[:len(prompt)] = 0
        
        return example, label


    def __getitem__(self, idx):

        prompt_question,prompt_answer= build_prompt(self.problems,self.qids[idx],self.args)
        prompt_question = self.system_message + "Image:N/A.\n" + prompt_question + "[/INST]"
        # answer,choices,qid=self.problems[self.qids[idx]]["answer"], self.problems[self.qids[idx]]["choices"],self.qids[idx]

        images_lst = []

        if self.problems[self.qids[idx]]['image'] is not None:
            image = Image.open(os.path.join(self.image_path, self.qids[idx], 'image.png')).convert('RGB')
            image = self.transforms(image)
            
            prompt_question = prompt_question.replace("Image:N/A", "Image:<image>"+"<image_placeholder>"*32)
        else:
            image = torch.Tensor(torch.zeros(3,224,224).float())
        images_lst.append(image)

        images = torch.stack(images_lst)

        example, label=self.tokenize(prompt_question,prompt_answer)

        image_positions = torch.where(example==32000, True, False)

        samples = {'example': example,'label': label,  'images': images, 'image_positions': image_positions}

        return samples, len(example), len(images_lst)

    def __len__(self):
        return len(self.qids)

    def shuffle_list(self, list):
        random.shuffle(list)

    def decode(self, t):
        return self.tokenizer.decode(t)