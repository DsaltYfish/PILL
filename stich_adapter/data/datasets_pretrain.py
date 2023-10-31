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

class CCPretrainDataSet(Data.Dataset):
    def __init__(self, args,model_path,max_words=512):
        super(CCPretrainDataSet, self).__init__()
        self.args = args
        # --------------------------
        # ---- Raw data loading ---
        # --------------------------
        self.data = json.load(open(os.path.join(args.data_root, 'chat.json')))
        self.data_path = args.data_root

        self.tokenizer = Tokenizer(model_path=model_path + '/tokenizer.model')
        self.max_words = max_words

        print(f"number of problems: {len(self.data)}\n")

        # self.system_message = "[INST]<<SYS>>\nYou are a helpful vision language assistant. Based on the image and instruction, please provide an appropriate response. \n<</SYS>>\n\n"

        self.system_message = "[INST]"

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

        question=self.data[idx]['conversations'][0]['value']
        answer=self.data[idx]['conversations'][1]['value']

        question=question.replace("<image>", "<image>"+"<image_placeholder>"*32)

        prompt_question = self.system_message + question + "[/INST]"
        images_lst = []

        if self.data[idx]['image'] is not None:
            image = Image.open(os.path.join(self.data_path, 'images', self.data[idx]['image'])).convert('RGB') 
            image = self.transforms(image)
            images_lst.append(image)

        images = torch.stack(images_lst)

        example, label=self.tokenize(prompt_question,answer)

        image_positions = torch.where(example==32000, True, False)

        samples = {'example': example,'label': label,  'images': images, 'image_positions': image_positions}

        return samples, len(example), len(images_lst)

    def __len__(self):
        return len(self.data)

    def shuffle_list(self, list):
        random.shuffle(list)