# PILL: Plug into LLM with Adapter Expert and Attention Gate

## Install

#### Install Package
    git clone https://github.com/DsaltYfish/PILL.git
    cd PILL

    conda create -n pill python=3.9 -y
    conda activate pill

    pip install -r requirements.txt

#### weight Preparation
PILL is based on Llama2 Chat 7B and Blip2 FlanT5xxl. Download the corresponding [LLM](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf/tree/main) from the following huggingface space via clone the repository using git-lfs.

#### Data Preparation
- For ScienceQA dataset, please prepare the dataset from the [official repo](https://github.com/lupantech/ScienceQA).
- For pretraining dataset CC595K, please prepare the dataset from the [LLaVA](https://github.com/haotian-liu/LLaVA/blob/main/docs/Data.md) or from [HuggingFace](https://huggingface.co/datasets/liuhaotian/LLaVA-CC3M-Pretrain-595K)
- For instruction tuning dataset mix665k, please prepare the annotation of the instruction tuning data from the [LLaVA](https://github.com/haotian-liu/LLaVA/blob/main/docs/Data.md) or from [HuggingFace](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/blob/main/llava_v1_5_mix665k.json), and download the images from constituting datasets:[COCO](http://images.cocodataset.org/zips/train2017.zip), [GQA](https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip), [OCR-VQA](https://drive.google.com/drive/folders/1_GYPY5UkUy7HIcR0zq3ZCFgeZN7BAfm_?usp=sharing)(please save all files as .img), [TextVQA](https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip), [VisualGenome_part1](https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip) and [VisualGenome_part2](https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip)

After downloading all of them, organize the data as follows in ./dataset

    ├── coco
    │   └── train2017
    ├── gqa
    │   └── images
    ├── llava
    │   ├── chat.json
    │   └── images
    ├── llava1.5
    │   ├── images
    │   └── llava_v1_5_mix665k.json
    ├── ocr_vqa
    │   └── images
    ├── scienceQA
    │   ├── captions.json
    │   ├── pid_splits.json
    │   ├── problems.json
    │   ├── test
    │   └── train
    ├── textvqa
    │   └── train_images
    └── vg
        ├── VG_100K
        └── VG_100K_2
    

## Train
we provide Pretrained, scienceQA and Instruction Tuning weight in
https://drive.google.com/drive/folders/17tMI_-NbhySuetocRNadvuO8ElJO2c_b?usp=sharing

#### Pretrain   
    torchrun --nproc_per_node=1 pretrain_cc595k.py \
                             --epoch=3 \
                             --accum_iter=8 \
                             --batch_size=16 \
                             --blr=1e-3 \
                             --data_root=./dataset/llava \
                             --output_dir=./output_dir/pretrain \
                             --log_dir=./output_dir/pretrain

#### ScienceQA
    torchrun --nproc_per_node=1 train_instruct.py \
                            --epoch=20 \
                            --accum_iter=8 \
                            --batch_size=4 \
                            --blr=2e-3 \
                            --data_root=./dataset/scienceQA \
                            --adapter_model=./output_dir/pretrain/checkpoint-2.pth \
                            --output_dir=./output_dir/sqa \
                            --log_dir=./output_dir/sqa

#### Instruction Tuning
    torchrun --nproc_per_node=1 train_instruct.py \
                            --epoch=5 \
                            --accum_iter=16 \
                            --batch_size=2 \
                            --blr=1e-3 \
                            --data_root=./dataset/llava1.5 \
                            --adapter_model=./output_dir/pretrain/checkpoint-2.pth \
                            --output_dir=./output_dir/finetune \
                            --log_dir=./output_dir/finetune


## Acknowledgement

This repo borrows some data and codes from [LLaMA](https://github.com/facebookresearch/llama), [LLaVA](https://github.com/haotian-liu/LLaVA), [MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4) and [LaVIN](https://github.com/luogen1996/LaVIN). Thanks for their great works.