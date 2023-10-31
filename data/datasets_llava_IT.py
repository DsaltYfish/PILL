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
import torch
from data.tokenizer import Tokenizer
# from data import data_collate
import copy
from moviepy.editor import VideoFileClip
class InstrcutDataSet(Data.Dataset):
    def __init__(self, args,model_path,max_words=512):
        super(InstrcutDataSet, self).__init__()
        self.args = args
        # --------------------------
        # ---- Raw data loading ---
        # --------------------------
        # self.data = json.load(open(os.path.join(args.data_root, 'llava_instruct_150k.json')))
        self.data = json.load(open(os.path.join(args.data_root, 'llava_v1_5_mix665k.json')))

        self.tokenizer = Tokenizer(model_path=model_path + '/tokenizer.model')
        self.max_words = max_words

        print(f"number of problems: {len(self.data)}\n")

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

        example, label=self.tokenize(prompt_question,answer)

        images_lst = []

        if 'image' in self.data[idx]:
            image = Image.open(os.path.join('/data/zfy/dataset', self.data[idx]['image'])).convert('RGB')
            image = self.transforms(image)

        else:
            image = torch.Tensor(torch.zeros(3,224,224).float())

        images_lst.append(image)

        for i in range(2,len(self.data[idx]['conversations']),2):

            question="[INST]"+self.data[idx]['conversations'][i]['value']+"[/INST]"
            answer=self.data[idx]['conversations'][i+1]['value']

            dialog_example, dialog_label=self.tokenize(question,answer)

            example = torch.cat([example, dialog_example])
            label = torch.cat([label, dialog_label])
        
        images = torch.stack(images_lst)

        example = example[:self.max_words]
        label = label[:self.max_words]

        image_positions = torch.where(example==32000, True, False)

        samples = {'example': example,'label': label,  'images': images, 'image_positions': image_positions}

        return samples, len(example), len(images_lst)

    def __len__(self):
        return len(self.data)

    def shuffle_list(self, list):
        random.shuffle(list)