import cv2
import numpy
from matplotlib import pyplot as plt
import glob
import json
import os
from mpl_toolkits.axes_grid1 import ImageGrid
import csv
import json
import os
from PIL import Image
from io import BytesIO
import base64
import pandas as pd
from pycocoevalcap_custom.eval import COCOEvalCap
from pycocotools.coco import COCO
import os
import re



test_data = '/data/yjcho/PLURAL/data/mimic_diff_vqa_test.tsv'
model_path = '/data/yjcho/PLURAL/checkpoints/stage3_finetune.pt'


name = test_data.split('/')[-1][:-4]+'_'+model_path.split('/')[-3]+'_'+model_path.split('/')[-1][:-3]
print(name)

annotation_file = f'gts_{name}.json'
results_file = f'res_{name}.json'

# =====================================================================================

fp = open(f'{test_data}', "r")
input_args = ["", "--task=refcoco", "--beam=10", f"--path={model_path}", "--bpe-dir=utils/BPE", "--no-repeat-ngram-size=3", "--patch-image-size=384"]

fp.seek(0)
images = []
prev_images = []
gts = []
questions = []
dataset = []
fp.seek(0)
while(True):
    column_l = fp.readline().rstrip("\n").split("\t")
    if len(column_l) == 1: break
    images.append(column_l[1])
    gts.append(column_l[2])
    questions.append(column_l[3])
    dataset.append(column_l[6])
    prev_images.append(column_l[4])
    
print(len(prev_images))


import torch
import torch.nn.functional as F
import numpy as np
from fairseq import checkpoint_utils, options, tasks, utils
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from tasks.mm_tasks.refcoco import RefcocoTask

from models.ofa import OFAModel
from PIL import Image, ImageOps

tasks.register_task('refcoco', RefcocoTask)

# turn on cuda if GPU is available
use_cuda = torch.cuda.is_available()
# use fp16 only when GPU is available
use_fp16 = False

# specify some options for evaluation
parser = options.get_generation_parser()
# tiny model 일때만 patch 284
args = options.parse_args_and_arch(parser, input_args)
cfg = convert_namespace_to_omegaconf(args)

# Load pretrained ckpt & config
task = tasks.setup_task(cfg.task)
models, cfg = checkpoint_utils.load_model_ensemble(
    utils.split_paths(cfg.common_eval.path),
    task=task
)
# Move models to GPU
for model in models:
    model.eval()
    if use_fp16:
        model.half()
    if use_cuda and not cfg.distributed_training.pipeline_model_parallel:
        model.cuda()
    model.prepare_for_inference_(cfg)

# Initialize generator
generator = task.build_generator(models, cfg.generation)

# Image transform
from torchvision import transforms
mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]

patch_resize_transform = transforms.Compose([
    lambda image: image.convert("RGB"),
    transforms.Resize((task.cfg.patch_image_size, task.cfg.patch_image_size), interpolation=Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
    transforms.RandomHorizontalFlip(p=0)
])

# Text preprocess
bos_item = torch.LongTensor([task.src_dict.bos()])
eos_item = torch.LongTensor([task.src_dict.eos()])
pad_idx = task.src_dict.pad()


def get_symbols_to_strip_from_output(generator):
    if hasattr(generator, "symbols_to_strip_from_output"):
        return generator.symbols_to_strip_from_output
    else:
        return {generator.bos, generator.eos}


def decode_fn(x, tgt_dict, bpe, generator, tokenizer=None):
    x = tgt_dict.string(x.int().cpu(), extra_symbols_to_ignore=get_symbols_to_strip_from_output(generator))
    token_result = []
    bin_result = []
    img_result = []
    for token in x.strip().split():
        if token.startswith('<bin_'):
            bin_result.append(token)
        elif token.startswith('<code_'):
            img_result.append(token)
        else:
            if bpe is not None:
                token = bpe.decode('{}'.format(token))
            if tokenizer is not None:
                token = tokenizer.decode(token)
            if token.startswith(' ') or len(token_result) == 0:
                token_result.append(token.strip())
            else:
                token_result[-1] += token

    return ' '.join(token_result), ' '.join(bin_result), ' '.join(img_result)


def coord2bin(coords, w_resize_ratio, h_resize_ratio):
    coord_list = [float(coord) for coord in coords.strip().split()]
    bin_list = []
    bin_list += ["<bin_{}>".format(int(round(coord_list[0] * w_resize_ratio / task.cfg.max_image_size * (task.cfg.num_bins - 1))))]
    bin_list += ["<bin_{}>".format(int(round(coord_list[1] * h_resize_ratio / task.cfg.max_image_size * (task.cfg.num_bins - 1))))]
    bin_list += ["<bin_{}>".format(int(round(coord_list[2] * w_resize_ratio / task.cfg.max_image_size * (task.cfg.num_bins - 1))))]
    bin_list += ["<bin_{}>".format(int(round(coord_list[3] * h_resize_ratio / task.cfg.max_image_size * (task.cfg.num_bins - 1))))]
    return ' '.join(bin_list)


def bin2coord(bins, w_resize_ratio, h_resize_ratio):
    bin_list = [int(bin[5:-1]) for bin in bins.strip().split()]
    coord_list = []
    coord_list += [bin_list[0] / (task.cfg.num_bins - 1) * task.cfg.max_image_size / w_resize_ratio]
    coord_list += [bin_list[1] / (task.cfg.num_bins - 1) * task.cfg.max_image_size / h_resize_ratio]
    coord_list += [bin_list[2] / (task.cfg.num_bins - 1) * task.cfg.max_image_size / w_resize_ratio]
    coord_list += [bin_list[3] / (task.cfg.num_bins - 1) * task.cfg.max_image_size / h_resize_ratio]
    return coord_list


def encode_text(text, length=None, append_bos=False, append_eos=False):
    line = [
      task.bpe.encode(' {}'.format(word.strip())) 
      if not word.startswith('<code_') and not word.startswith('<bin_') else word
      for word in text.strip().split()
    ]
    line = ' '.join(line)
    s = task.tgt_dict.encode_line(
        line=line,
        add_if_not_exist=False,
        append_eos=False
    ).long()
    if length is not None:
        s = s[:length]
    if append_bos:
        s = torch.cat([bos_item, s])
    if append_eos:
        s = torch.cat([s, eos_item])
    return s

def construct_sample1(image: Image, instruction: str):
    patch_image = patch_resize_transform(image).unsqueeze(0)
    
    patch_mask = torch.tensor([True])

    instruction = encode_text(' {}'.format(instruction.lower().strip()), append_bos=True, append_eos=True).unsqueeze(0)
    instruction_length = torch.LongTensor([s.ne(pad_idx).long().sum() for s in instruction])
    sample = {
        "id":np.array(['42']),
        "net_input": {
            "src_tokens": instruction,
            "src_lengths": instruction_length,
            "patch_images": patch_image,
            "patch_masks": patch_mask,
        }
    }
    return sample
def construct_sample2(image: Image, image2 : Image, instruction: str):
    patch_image = patch_resize_transform(image).unsqueeze(0)
    patch_image2 = patch_resize_transform(image2).unsqueeze(0)
    
    patch_mask = torch.tensor([True])

    instruction = encode_text(' {}'.format(instruction.lower().strip()), append_bos=True, append_eos=True).unsqueeze(0)
    instruction_length = torch.LongTensor([s.ne(pad_idx).long().sum() for s in instruction])
    sample = {
        "id":np.array(['42']),
        "net_input": {
            "src_tokens": instruction,
            "src_lengths": instruction_length,
            "patch_images": patch_image,
            "patch_images_2": patch_image2,
            "patch_masks": patch_mask,
        }
    }
    return sample

def construct_sample_wo_image(instruction: str):
    patch_mask = torch.tensor([True])
    instruction = encode_text(' {}'.format(instruction.lower().strip()), append_bos=True, append_eos=True).unsqueeze(0)
    instruction_length = torch.LongTensor([s.ne(pad_idx).long().sum() for s in instruction])
    sample = {
        "id":np.array(['42']),
        "net_input": {
            "src_tokens": instruction,
            "src_lengths": instruction_length,
            "patch_masks": patch_mask,
        }
    }
    return sample

# Function to turn FP32 to FP16
def apply_half(t):
    if t.dtype is torch.float32:
        return t.to(dtype=torch.half)
    return t

import time


infer_res = {"res":[], "gts": []}
start = time.time()

for kk in range(len(images)):
    #instruction = 'what does the image describe?'
    #instruction = 'what is the summary of the image?'
    
    instruction = questions[kk]
    #print(instruction)
    gt = gts[kk]
    #print(instruction)
    if len(prev_images[kk]) == 0:
        
        image = Image.open(BytesIO(base64.urlsafe_b64decode(images[kk])))

        
        
        w = image.size[0]
        h = image.size[1]

        

        # Construct input sample & preprocess for GPU if cuda available
        #sample = construct_sample(image, image2, instruction)
        sample = construct_sample1(image, instruction)
        
        sample = utils.move_to_cuda(sample) if use_cuda else sample
        sample = utils.apply_to_sample(apply_half, sample) if use_fp16 else sample

        # Generate result
        with torch.no_grad():
            hypos = task.inference_step(generator, models, sample)
            tokens, bins, imgs = decode_fn(hypos[0][0]["tokens"], task.tgt_dict, task.bpe, generator)
        print('{0} / {1}'.format(kk, len(images)), end = '\r')
        infer_res["res"].append(tokens)
        infer_res['gts'].append(gt)
    else:
    
        image = Image.open(BytesIO(base64.urlsafe_b64decode(images[kk])))
        image2 = Image.open(BytesIO(base64.urlsafe_b64decode(prev_images[kk])))
        
        w = image.size[0]
        h = image.size[1]


        # Construct input sample & preprocess for GPU if cuda available
        #sample = construct_sample(image, image2, instruction)
        sample = construct_sample2(image, image2, instruction)
        
        sample = utils.move_to_cuda(sample) if use_cuda else sample
        sample = utils.apply_to_sample(apply_half, sample) if use_fp16 else sample

        # Generate result
        with torch.no_grad():
            hypos = task.inference_step(generator, models, sample)
            tokens, bins, imgs = decode_fn(hypos[0][0]["tokens"], task.tgt_dict, task.bpe, generator)
        print('{0} / {1}'.format(kk, len(images)), end = '\r')
        infer_res["res"].append(tokens)
        infer_res['gts'].append(gt)
 
    
    #print(gt)
    #print(tokens)


# print(f"{end - start:.5f} sec")
print('res len:',len(infer_res['res']))
print('gts len:',len(infer_res['gts']))
res = []
gts_img = []
gts_ann = []

for i , cap in enumerate(zip(infer_res['res'], infer_res['gts'])):
    
    res.append({"image_id":str(i*2), 'caption':cap[0]})
    gts_img.append({"id":str(i*2), 'caption':cap[1]})
    gts_ann.append({"image_id":str(i*2), "id":str(i), 'caption':cap[1]})
    
gts = {'images':gts_img, 'annotations':gts_ann}    
  

json.dump(res,open(f'./eval_results/{results_file}','w'))
json.dump(gts,open(f'./eval_results/{annotation_file}','w'))
print('json file save!')
time.sleep(10)


# create coco object and coco_result object
coco = COCO(f'./eval_results/{annotation_file}')
coco_result = coco.loadRes(f'./eval_results/{results_file}')

# create coco_eval object by taking coco and coco_result
coco_eval = COCOEvalCap(coco, coco_result)

# evaluate on a subset of images by setting
# coco_eval.params['image_id'] = coco_result.getImgIds()
# please remove this line when evaluating the full validation set
coco_eval.params['image_id'] = coco_result.getImgIds()

# evaluate results
# SPICE will take a few minutes the first time, but speeds up due to caching
coco_eval.evaluate()
print('=========================================')
print(test_data)
print(model_path)
# print output evaluation scores
for metric, score in coco_eval.eval.items():
    
    print(f'{metric}: {score:.3f}')
print(name)
