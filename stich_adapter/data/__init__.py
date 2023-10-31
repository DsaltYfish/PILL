# from data.datasets_instruct import InstrcutDataSet
from data.datasets_llava_IT import InstrcutDataSet
from data.datasets_pretrain import CCPretrainDataSet
from data.datasets_sqa import ScienceQADataSet
import torch

def data_collate(batch):
    samples = [item[0] for item in batch]

    max_example_len = max([item[1] for item in batch])
    max_image_len = max([item[2] for item in batch])

    examples = []
    labels = []
    images_w_pad = []
    image_positions_w_pad = []
    
    for sample in samples:
        example = sample['example']
        label = sample['label']
        images = sample['images']
        image_positions = sample['image_positions']
        
        txt_pad = max_example_len - example.shape[0]
        img_pad = max_image_len - images.shape[0]
        if txt_pad > 0:
            example = torch.cat((example, torch.zeros(txt_pad, dtype=torch.int64)))
            label = torch.cat((label, torch.zeros(txt_pad, dtype=torch.int64)))
            image_positions = torch.cat((image_positions, torch.zeros(txt_pad, dtype=torch.bool)))

        if img_pad > 0:
            images = torch.cat((images, torch.zeros([img_pad, 3, 224, 224]).float()))

        examples.append(example)
        labels.append(label)
        images_w_pad.append(images)
        image_positions_w_pad.append(image_positions)

    return torch.stack(examples), torch.stack(labels), torch.stack(images_w_pad), torch.stack(image_positions_w_pad)