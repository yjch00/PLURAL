# Copyright 2022 The OFA-Sys Team. 
# All rights reserved.
# This source code is licensed under the Apache 2.0 license 
# found in the LICENSE file in the root directory.

from io import BytesIO

import logging
import warnings
import string

import numpy as np
import torch
import base64
from torchvision import transforms

from PIL import Image, ImageFile

from data import data_utils
from data.ofa_dataset import OFADataset

ImageFile.LOAD_TRUNCATED_IMAGES = True
ImageFile.MAX_IMAGE_PIXELS = None
Image.MAX_IMAGE_PIXELS = None

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


def collate(samples, pad_idx, eos_idx):
    if len(samples) == 0:
        return {}

    def merge(key):
        return data_utils.collate_tokens(
            [s[key] for s in samples],
            pad_idx,
            eos_idx=eos_idx,
        )

    id = np.array([s["id"] for s in samples])
    src_tokens = merge("source")
    src_lengths = torch.LongTensor([s["source"].ne(pad_idx).long().sum() for s in samples])

    patch_images = torch.stack([sample['patch_image'] for sample in samples], dim=0)
    #print('222',samples[0])
    if samples[0].get("patch_image_2", None) is not None:        
        patch_images_2 = torch.stack([sample['patch_image_2'] for sample in samples], dim=0)
    else:
        patch_images_2 = None

    patch_masks = torch.cat([sample['patch_mask'] for sample in samples])

    prev_output_tokens = None
    target = None
    if samples[0].get("target", None) is not None:
        target = merge("target")
        tgt_lengths = torch.LongTensor([s["target"].ne(pad_idx).long().sum() for s in samples])
        ntokens = tgt_lengths.sum().item()

        if samples[0].get("prev_output_tokens", None) is not None:
            prev_output_tokens = merge("prev_output_tokens")
    else:
        ntokens = src_lengths.sum().item()

    batch = {
        "id": id,
        "nsentences": len(samples),
        "ntokens": ntokens,
        "net_input": {
            "src_tokens": src_tokens,
            "src_lengths": src_lengths,
            "patch_images": patch_images,
            "patch_images_2": patch_images_2,
            "patch_masks": patch_masks,
            "prev_output_tokens": prev_output_tokens
        },
        "target": target,
    }

    return batch


class CaptionDataset(OFADataset):
    def __init__(
        self,
        split,
        dataset,
        bpe,
        src_dict,
        tgt_dict=None,
        max_src_length=128,
        max_tgt_length=30,
        patch_image_size=224,
        imagenet_default_mean_and_std=False,
        scst=False,
        diff_dataset=None
    ):
        super().__init__(split, dataset, bpe, src_dict, tgt_dict)
        self.max_src_length = max_src_length
        self.max_tgt_length = max_tgt_length
        self.patch_image_size = patch_image_size
        self.scst = scst
        self.diff_dataset = diff_dataset

        self.transtab = str.maketrans({key: None for key in string.punctuation})

        if imagenet_default_mean_and_std:
            mean = IMAGENET_DEFAULT_MEAN
            std = IMAGENET_DEFAULT_STD
        else:
            mean = [0.5, 0.5, 0.5]
            std = [0.5, 0.5, 0.5]

        self.patch_resize_transform = transforms.Compose([
            lambda image: image.convert("RGB"),
            transforms.Resize((patch_image_size, patch_image_size), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

        if type(bpe).__name__ == 'GPT2BPE':
            self.prompt = " what does the image describe?"
        elif type(bpe).__name__ == 'BertBPE':
            self.prompt = "图片描述了什么内容?"

    def process_data(self, index, dataset):
        uniq_id, image, caption, question, prev_image, _, dataset_name, _ = dataset[index%len(dataset)]

        image = Image.open(BytesIO(base64.urlsafe_b64decode(image)))
        patch_image = self.patch_resize_transform(image)
        patch_mask = torch.tensor([True])
        
        if self.split == 'train' and not self.scst:
            caption = caption.translate(self.transtab).strip()
            caption_token_list = caption.strip().split()
            tgt_caption = ' '.join(caption_token_list[:self.max_tgt_length])
        else:
            caption = ' '.join(caption.strip().split())
            caption_list = [cap.translate(self.transtab).strip() for cap in caption.strip().split('&&')]
            tgt_caption = '&&'.join(caption_list)
        
        


        tgt_item = self.encode_text(" {}".format(tgt_caption))
        target_item = torch.cat([tgt_item, self.eos_item])
        prev_output_item = torch.cat([self.bos_item, tgt_item])

        if dataset_name == 'mimic_f':
            src_item = self.encode_text(self.prompt)
            
            
        elif dataset_name == 'mimic_i':
            src_item = self.encode_text('what is the summary of the image?')     
              
        else:
            

            if self.split == 'train' and not self.scst:
                question = question.translate(self.transtab).strip()
                question_token_list = question.strip().split()
                tgt_question = ' '.join(question_token_list[:self.max_src_length])
            else:
                question = ' '.join(question.strip().split())
                question_list = [cap.translate(self.transtab).strip() for cap in question.strip().split('&&')]
                tgt_question = '&&'.join(question_list)
            src_item = self.encode_text(tgt_question)

        src_item = torch.cat([self.bos_item, src_item, self.eos_item])
        
        if len(prev_image) == 0:
            example = {
                "id": uniq_id,
                "source": src_item,
                "patch_image": patch_image,
                "patch_mask": patch_mask,
                "target": target_item,
                "prev_output_tokens": prev_output_item
            }
        else:
            prev_image = Image.open(BytesIO(base64.urlsafe_b64decode(prev_image)))
            prev_patch_image = self.patch_resize_transform(prev_image)
            example = {
                "id": uniq_id,
                "source": src_item,
                "patch_image": patch_image,
                "patch_image_2": prev_patch_image,
                "patch_mask": patch_mask,
                "target": target_item,
                "prev_output_tokens": prev_output_item
            }
        return [example]


    def __getitem__(self, index):
        example = self.process_data(index, self.dataset)

        diff_example = []
        if self.diff_dataset is not None:
            diff_example += self.process_data(index, self.diff_dataset)
        return example, diff_example


    def collater(self, samples, pad_to_length=None):
        """Merge samples of different tasks to form two mini-batches.
        Args:
            samples (List[Tuple]): samples to collate
        Returns:
            Tuple[dict]: two mini-batch containing the data of different tasks
        """

        samples_v1 = []   # containing image-text pairs
        samples_v2 = []   # containing detection data, text data and image data

        for sample_tuple in samples:
            samples_v1 += sample_tuple[0]
            samples_v2 += sample_tuple[1]
        if samples_v2 != []:
            res_v1 = collate(samples_v1, pad_idx=self.pad, eos_idx=self.eos)
            res_v2 = collate(samples_v2, pad_idx=self.pad, eos_idx=self.eos)
            return [res_v1, res_v2]
        else:
            res_v1 = collate(samples_v1, pad_idx=self.pad, eos_idx=self.eos)
            return res_v1
