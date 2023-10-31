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
    def __init__(self, args,model_path,max_words=512, num_frames=8):
        super(InstrcutDataSet, self).__init__()
        self.args = args
        # --------------------------
        # ---- Raw data loading ---
        # --------------------------

        self.json_files = []
        for root, dirs, files in os.walk(args.json_folder_path):
            for file in files:
                if file == "train.jsonl":
                    self.json_files.append(os.path.join(root, file))

        self.shuffle_list(self.json_files)
        self.data = []
        for file_path in self.json_files:
            # i = 0
            with open(file_path, "r", encoding="utf-8") as jsonl_file:
                for line in jsonl_file:
                    # i += 1
                    # if i == 100:
                    #     break
                    json_data = json.loads(line)
                    self.data.append(json_data)

        self.tokenizer = Tokenizer(model_path=model_path + '/tokenizer.model')
        self.max_words = max_words

        print(f"number of problems: {len(self.data)}\n")

        self.system_message = "[INST]"
        self.num_frames = num_frames

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

        question=self.data[idx]['input_text']
        answer=self.data[idx]['output_text']

        question=question.replace("图", "<image_placeholder>"*32)

        prompt_question = self.system_message + question + "[/INST]"

        example, label=self.tokenize(prompt_question,answer)

        images_lst = []

        image_files = self.data[idx]['input_image']

        if image_files is not None:

            if isinstance(image_files, list):
                for image_file in image_files:
                    image_file = image_file.replace("./data/", "/data/zfy/dataset/")
                    image = Image.open(image_file).convert('RGB')
                    image = self.transforms(image)
                    images_lst.append(image)

            else:
                image_file = image_files.replace("./data/", "/data/zfy/dataset/")
                postfix = image_file.split('.')[-1]
                if postfix == 'mp4' or postfix == 'webm' or postfix == 'avi':
                    images_lst = self.extract_frames(image_file,self.num_frames)
                else:
                    image = Image.open(image_file).convert('RGB')
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

    def extract_frames(self, video_path, num_frames):
        clip = VideoFileClip(video_path)
        duration = clip.duration
        frame_times = [duration * i / (num_frames + 1) for i in range(1, num_frames + 1)]
        
        frames = []
        for t in frame_times:
            frame = clip.get_frame(t)
            image = Image.fromarray(frame)
            image = self.transforms(image)
            frames.append(image)
        
        clip.close()
        
        return frames
