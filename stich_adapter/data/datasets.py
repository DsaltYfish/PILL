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

        self.transforms=transforms.Compose([transforms.Resize((224, 224), interpolation=Image.BICUBIC),transforms.ToTensor(), transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)])

    def tokenize(self,prompt,answer):
        example=prompt+answer
        # print(prompt)
        prompt=torch.tensor(self.tokenizer.encode(prompt, bos=True, eos=False), dtype=torch.int64)
        example = torch.tensor(self.tokenizer.encode(example, bos=True, eos=True), dtype=torch.int64)
        padding = self.max_words - example.shape[0]
        if padding > 0:
            example = torch.cat((example, torch.zeros(padding, dtype=torch.int64) - 1))
        elif padding < 0:
            example = example[:self.max_words]
        labels = copy.deepcopy(example)
        labels[:len(prompt)] = -1
        example_mask = example.ge(0)
        label_mask = labels.ge(0)
        example[~example_mask] = 0
        labels[~label_mask] = 0
        example_mask = example_mask.float()
        label_mask = label_mask.float()
        return example, labels, example_mask,label_mask


    def __getitem__(self, idx):

        prompt_question,prompt_answer= build_prompt(self.problems,self.qids[idx],self.args)
        prompt_question = self.system_message + "\n" + prompt_question + "[/INST]"
        answer,choices,qid=self.problems[self.qids[idx]]["answer"], self.problems[self.qids[idx]]["choices"],self.qids[idx]

        if self.problems[self.qids[idx]]['image'] is not None:
            image = Image.open(os.path.join(self.image_path, self.qids[idx], 'image.png')).convert('RGB')
            image = self.transforms(image)
            # image_mask=torch.cat([torch.Tensor([float('-inf')]*self.max_image_feats),torch.zeros(self.max_words)])
            indicator=1
        else:
            image=torch.Tensor(torch.zeros(3,224,224).float())
            # image_mask=torch.zeros(self.max_words+self.max_image_feats)
            indicator=0

        example, labels, example_mask, label_mask=self.tokenize(prompt_question,prompt_answer)

        return example, labels, example_mask, image,indicator

    def __len__(self):
        return len(self.qids)

    def shuffle_list(self, list):
        random.shuffle(list)

    def decode(self, t):
        return self.tokenizer.decode(t)



class InstrcutDataSet(Data.Dataset):
    def __init__(self, args,model_path,max_words=512):
        super(InstrcutDataSet, self).__init__()
        self.args = args
        # --------------------------
        # ---- Raw data loading ---
        # --------------------------
        self.data = json.load(open(os.path.join(args.data_root, 'llava_instruct_150k.json')))

        self.tokenizer = Tokenizer(model_path=model_path + '/tokenizer.model')
        self.max_words = max_words

        print(f"number of problems: {len(self.data)}\n")

        self.system_message = "[INST]<<SYS>>\nYou are a helpful vision language assistant. Based on the image and instruction, please provide an appropriate response. \n<</SYS>>\n\n"

        self.transforms=transforms.Compose([transforms.Resize((224, 224), interpolation=Image.BICUBIC),transforms.ToTensor(), transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)])

    def tokenize(self,prompt,answer,max_words=512):
        example=prompt+answer
        # print(prompt)
        prompt=torch.tensor(self.tokenizer.encode(prompt, bos=True, eos=False), dtype=torch.int64)
        example = torch.tensor(self.tokenizer.encode(example, bos=True, eos=True), dtype=torch.int64)
        # padding = max_words - example.shape[0]
        # if padding > 0:
        #     example = torch.cat((example, torch.zeros(padding, dtype=torch.int64) - 1))
        # elif padding < 0:
        #     example = example[:self.max_words]
        labels = copy.deepcopy(example)
        labels[:len(prompt)] = -1
        example_mask = example.ge(0)
        label_mask = labels.ge(0)
        example[~example_mask] = 0
        labels[~label_mask] = 0
        return example, labels, example_mask


    def __getitem__(self, idx):

        question=self.data[idx]['conversations'][0]['value']
        answer=self.data[idx]['conversations'][1]['value']

        question=question.replace("<image>\n", "").replace("\n<image>", "")

        prompt_question = self.system_message + question + "[/INST]"

        example, labels, example_mask=self.tokenize(prompt_question,answer)

        if self.data[idx]['image'] is not None:
            image = Image.open(os.path.join('/data/zfy/dataset/coco/train2014', 'COCO_train2014_'+self.data[idx]['image'])).convert('RGB')
            image = self.transforms(image)
            indicator=1
        else:
            image=torch.Tensor(torch.zeros(3,224,224).float())
            indicator=0

        for i in range(2,len(self.data[idx]['conversations']),2):

            question="[INST]"+self.data[idx]['conversations'][i]['value']+"[/INST]"
            answer=self.data[idx]['conversations'][i+1]['value']

            dialog_example, dialog_labels, dialog_example_mask=self.tokenize(question,answer)

            example = torch.cat([example, dialog_example])
            labels = torch.cat([labels, dialog_labels])
            example_mask = torch.cat([example_mask, dialog_example_mask])
            
        padding = self.max_words - example.shape[0]
        if padding > 0:
            example = torch.cat((example, torch.zeros(padding, dtype=torch.int64)))
            labels = torch.cat((labels, torch.zeros(padding, dtype=torch.int64)))
            example_mask = torch.cat((example_mask, torch.zeros(padding, dtype=torch.bool)))
        elif padding < 0:
            example = example[:self.max_words]
            labels = labels[:self.max_words]
            example_mask = example_mask[:self.max_words]

        return example, labels, example_mask, image, indicator

    def __len__(self):
        return len(self.data)

    def shuffle_list(self, list):
        random.shuffle(list)


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

    def tokenize(self,prompt,answer,max_words=512):
        example=prompt+answer
        # print(prompt)
        prompt=torch.tensor(self.tokenizer.encode(prompt, bos=True, eos=False), dtype=torch.int64)
        example = torch.tensor(self.tokenizer.encode(example, bos=True, eos=True), dtype=torch.int64)
        padding = max_words - example.shape[0]
        if padding > 0:
            example = torch.cat((example, torch.zeros(padding, dtype=torch.int64) - 1))
        elif padding < 0:
            example = example[:self.max_words]
        labels = copy.deepcopy(example)
        labels[:len(prompt)] = -1
        example_mask = example.ge(0)
        label_mask = labels.ge(0)
        example[~example_mask] = 0
        labels[~label_mask] = 0
        example_mask = example_mask.float()
        label_mask = label_mask.float()
        return example, labels, example_mask,label_mask


    def __getitem__(self, idx):

        question=self.data[idx]['conversations'][0]['value']
        answer=self.data[idx]['conversations'][1]['value']

        question=question.replace("<image>\n", "").replace("\n<image>", "")

        prompt_question = self.system_message + question + "[/INST]"

        if self.data[idx]['image'] is not None:
            image = Image.open(os.path.join(self.data_path, 'images', self.data[idx]['image'])).convert('RGB') 
            image = self.transforms(image)
            indicator=1
        else:
            image=torch.Tensor(torch.zeros(3,224,224).float())
            indicator=0

        example, labels, example_mask, label_mask=self.tokenize(prompt_question,answer, self.max_words)

        return example, labels, example_mask, image, indicator

    def __len__(self):
        return len(self.data)

    def shuffle_list(self, list):
        random.shuffle(list)


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    class Cfg():
        def __init__(self):
            super(Cfg, self).__init__()
            self.options = ["A", "B", "C", "D", "E"]
            self.use_caption = True
            self.prompt_format = 'CQM-A'
            self.data_root = './data'
            self.output_root = './output'
            self.caption_file = './data/captions.json'
    cfg=Cfg()
    dataset=ScienceQADataSet(cfg,'val','./data/weights')
    data_loader = DataLoader(dataset,
                             batch_size=1,
                             shuffle=False,
                             pin_memory=True)
    max_question_len=0
    max_answer_len=0
    #406 max question
    for prompt_questions,question_mask,images,image_masks,prompt_answers,answers,qids in data_loader:
        print(prompt_questions)
        print(answers)
    #     if len(prompt_questions[0].split())>max_question_len:
    #         max_question_len=len(prompt_questions[0].split())
    #     if len(prompt_answers[0].split())>max_answer_len:
    #         max_answer_len=len(prompt_answers[0].split())
    # print(max_question_len,max_answer_len)






