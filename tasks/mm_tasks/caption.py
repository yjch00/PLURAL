# Copyright 2022 The OFA-Sys Team. 
# All rights reserved.
# This source code is licensed under the Apache 2.0 license 
# found in the LICENSE file in the root directory.

from dataclasses import dataclass, field
import json
import logging
from typing import Optional
from argparse import Namespace
from itertools import zip_longest
from collections import OrderedDict

import numpy as np
import sacrebleu
import string
from fairseq import metrics, utils
from fairseq.tasks import register_task

from tasks.ofa_task import OFATask, OFAConfig
from data.mm_data.caption_dataset import CaptionDataset
from data.file_dataset import FileDataset
from utils.cider.pyciderevalcap.ciderD.ciderD import CiderD


EVAL_BLEU_ORDER = 4

logger = logging.getLogger(__name__)


@dataclass
class CaptionConfig(OFAConfig):
    diff_data: Optional[str] = field(
        default=None,
        metadata={"help": "diff vqa data"},
    )
    eval_bleu: bool = field(
        default=False, metadata={"help": "evaluation with BLEU scores"}
    )
    eval_cider: bool = field(
        default=False, metadata={"help": "evaluation with CIDEr scores"}
    )
    eval_args: Optional[str] = field(
        default='{}',
        metadata={
            "help": 'generation args for BLUE or CIDEr scoring, e.g., \'{"beam": 4, "lenpen": 0.6}\', as JSON string'
        },
    )
    eval_print_samples: bool = field(
        default=False, metadata={"help": "print sample generations during validation"}
    )
    eval_cider_cached_tokens: Optional[str] = field(
        default=None,
        metadata={"help": "path to cached cPickle file used to calculate CIDEr scores"},
    )

    scst: bool = field(
        default=False, metadata={"help": "Self-critical sequence training"}
    )
    scst_args: str = field(
        default='{}',
        metadata={
            "help": 'generation args for Self-critical sequence training, as JSON string'
        },
    )


@register_task("caption", dataclass=CaptionConfig)
class CaptionTask(OFATask):
    def __init__(self, cfg: CaptionConfig, src_dict, tgt_dict):
        super().__init__(cfg, src_dict, tgt_dict)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        paths = self.cfg.data.split(',')
        assert len(paths) > 0

        if split == 'train':
            file_path = paths[(epoch - 1) % (len(paths) - 1)]
        else:
            file_path = paths[-1]
        dataset = FileDataset(file_path, self.cfg.selected_cols)

        self.diff_dataset = None
        if self.cfg.diff_data is not None:
            diff_paths = self.cfg.diff_data.split(',')
            if split == 'train':
                diff_file_path = diff_paths[(epoch - 1) % (len(diff_paths) - 1)]
            else:
                diff_file_path = diff_paths[-1]
            self.diff_dataset = FileDataset(diff_file_path, self.cfg.selected_cols)


        self.datasets[split] = CaptionDataset(
            split,
            dataset,
            self.bpe,
            self.src_dict,
            self.tgt_dict,
            max_src_length=self.cfg.max_src_length,
            max_tgt_length=self.cfg.max_tgt_length,
            patch_image_size=self.cfg.patch_image_size,
            imagenet_default_mean_and_std=self.cfg.imagenet_default_mean_and_std,
            scst=getattr(self.cfg, 'scst', False),
            diff_dataset=self.diff_dataset
        )

    def build_model(self, cfg):
        model = super().build_model(cfg)

        gen_args = json.loads(self.cfg.eval_args)
        self.sequence_generator = self.build_generator(
            [model], Namespace(**gen_args)
        )
        if self.cfg.eval_bleu or self.cfg.eval_cider:
            gen_args = json.loads(self.cfg.eval_args)
            self.sequence_generator = self.build_generator(
                [model], Namespace(**gen_args)
            )
            if self.cfg.eval_cider:
                self.CiderD_scorer = CiderD(df=self.cfg.eval_cider_cached_tokens)
        if self.cfg.scst:
            scst_args = json.loads(self.cfg.scst_args)
            self.scst_generator = self.build_generator(
                [model], Namespace(**scst_args)
            )

        return model


    def _calculate_bert_scores(self, gen_res, gt_res):
        scores= bertscore.compute(predictions=gen_res, references=gt_res, model_type="bert-base-uncased")
        return scores

    def _calculate_distillbert_scores(self, gen_res, gt_res):
        scores = bertscore.compute(predictions=gen_res, references=gt_res, model_type="distilbert-base-uncased")
        return scores

    # def _calculate_bleu_scores(self, gen_res, gt_res, max_order):
    #     bleu = evaluate.load("bleu")
    #     scores = bleu.compute(predictions=gen_res, references=gt_res, max_order=max_order)
    #     return scores

    # def _calculate_rouge_scores(self, gen_res, gt_res):
    #     rouge = evaluate.load('rouge')
    #     scores = rouge.compute(predictions=gen_res, references=gt_res)
    #     return scores

    # def _calculate_radgraph_scores(self, gen_res, gt_res):
    #     f1radgraph = F1RadGraph(reward_level="partial")
    #     score, _, hypothesis_annotation_lists, reference_annotation_lists = f1radgraph(hyps=gen_res, refs=[i[0] for i in gt_res])
    #     return score

    # def _calculate_chexbert_scores(self, gen_res, gt_res):
    #     f1chexbert = F1CheXbert(device="cuda")
    #     accuracy, accuracy_not_averaged, class_report, class_report_5 = f1chexbert(
    #         hyps=gen_res,
    #         refs=[i[0] for i in gt_res])
    #     score = class_report_5["micro avg"]["f1-score"]
    #     return score



    def _calculate_cider_scores(self, gen_res, gt_res):
        '''
        gen_res: generated captions, list of str
        gt_idx: list of int, of the same length as gen_res
        gt_res: ground truth captions, list of list of str.
            gen_res[i] corresponds to gt_res[gt_idx[i]]
            Each image can have multiple ground truth captions
        '''
        gen_res_size = len(gen_res)

        res = OrderedDict()
        for i in range(gen_res_size):
            res[i] = [gen_res[i].strip()]

        gts = OrderedDict()
        gt_res_ = [
            [gt_res[i][j].strip() for j in range(len(gt_res[i]))]
            for i in range(len(gt_res))
        ]
        for i in range(gen_res_size):
            gts[i] = gt_res_[i]

        res_ = [{'image_id': i, 'caption': res[i]} for i in range(len(res))]
        _, scores = self.CiderD_scorer.compute_score(gts, res_)
        return scores

    def valid_step(self, sample, model, criterion):
        loss, sample_size, logging_output = criterion(model, sample)

        # model.eval()

        # hyps, refs = self._inference(self.sequence_generator, sample, model)
        # scores = self._calculate_bert_scores(hyps, refs)
        # logging_output["bert_precision"] = sum(scores["precision"])/len(scores["precision"])
        # logging_output["bert_recall"] = sum(scores["recall"])/len(scores["recall"])
        # logging_output["bert_f1"] = sum(scores["f1"])/len(scores["f1"])

        # scores = self._calculate_distillbert_scores(hyps, refs)
        # logging_output["distill_bert_precision"] = sum(scores["precision"])/len(scores["precision"])
        # logging_output["distill_bert_recall"] = sum(scores["recall"])/len(scores["recall"])
        # logging_output["distill_bert_f1"] = sum(scores["f1"])/len(scores["f1"])


        # scores = self._calculate_bleu_scores(hyps, refs ,1)
        # logging_output["bleu1"] = scores["bleu"]
        # logging_output["bleu_precision1"] = sum(scores["precisions"])/len(scores["precisions"])
        # scores = self._calculate_bleu_scores(hyps, refs ,2)
        # logging_output["bleu2"] = scores["bleu"]
        # logging_output["bleu_precision2"] = sum(scores["precisions"])/len(scores["precisions"])
        # scores = self._calculate_bleu_scores(hyps, refs ,3)
        # logging_output["bleu3"] = scores["bleu"]
        # logging_output["bleu_precision3"] = sum(scores["precisions"])/len(scores["precisions"])
        # scores = self._calculate_bleu_scores(hyps, refs ,4)
        # logging_output["bleu4"] = scores["bleu"]
        # logging_output["bleu_precision4"] = sum(scores["precisions"])/len(scores["precisions"])


        # scores = self._calculate_rouge_scores(hyps, refs)
        # logging_output["rouge1"] = scores["rouge1"]
        # logging_output["rouge2"] = scores["rouge2"]
        # logging_output["rougeL"] = scores["rougeL"]
        # logging_output["rougeLsum"] = scores["rougeLsum"]

        # scores = self._calculate_radgraph_scores(hyps, refs)
        # logging_output["F1_radgraph"] = scores


        return loss, sample_size, logging_output

    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)
        def sum_logs(key):
            import torch
            result = sum(log.get(key, 0) for log in logging_outputs)
            if torch.is_tensor(result):
                result = result.cpu()
            return result

        # metrics.log_scalar("bert_precision", sum_logs("bert_precision"))
        # metrics.log_scalar("bert_recall", sum_logs("bert_recall"))
        # metrics.log_scalar("bert_f1", sum_logs("bert_f1"))

        # metrics.log_scalar("distill_bert_precision", sum_logs("distill_bert_precision"))
        # metrics.log_scalar("distill_bert_recall", sum_logs("distill_bert_recall"))
        # metrics.log_scalar("distill_bert_f1", sum_logs("distill_bert_f1"))

        # metrics.log_scalar("bleu1", sum_logs("bleu1"))
        # metrics.log_scalar("bleu_precision1", sum_logs("bleu_precision1"))
        # metrics.log_scalar("bleu2", sum_logs("bleu2"))
        # metrics.log_scalar("bleu_precision2", sum_logs("bleu_precision2"))
        # metrics.log_scalar("bleu3", sum_logs("bleu3"))
        # metrics.log_scalar("bleu_precision3", sum_logs("bleu_precision3"))
        # metrics.log_scalar("bleu4", sum_logs("bleu4"))
        # metrics.log_scalar("bleu_precision4", sum_logs("bleu_precision4"))

        # metrics.log_scalar("rouge1", sum_logs("rouge1"))
        # metrics.log_scalar("rouge2", sum_logs("rouge2"))
        # metrics.log_scalar("rougeL", sum_logs("rougeL"))
        # metrics.log_scalar("rougeLsum", sum_logs("rougeLsum"))

        # metrics.log_scalar("F1_radgraph", sum_logs("F1_radgraph"))


    def _inference(self, generator, sample, model):

        def decode(toks, escape_unk=False):
            s = self.tgt_dict.string(
                toks.int().cpu(),
                # The default unknown string in fairseq is `<unk>`, but
                # this is tokenized by sacrebleu as `< unk >`, inflating
                # BLEU scores. Instead, we use a somewhat more verbose
                # alternative that is unlikely to appear in the real
                # reference, but doesn't get split into multiple tokens.
                unk_string=("UNKNOWNTOKENINREF" if escape_unk else "UNKNOWNTOKENINHYP"),
            )
            if self.bpe:
                s = self.bpe.decode(s)
            return s


        if isinstance(sample, list):
            gen_out = self.inference_step(generator, [model], sample[0])
            gen_out2 = self.inference_step(generator, [model], sample[1])
            hyps, refs = [], []
            transtab = str.maketrans({key: None for key in string.punctuation})
            for i in range(len(gen_out)):
                decode_tokens = decode(gen_out[i][0]["tokens"])
                hyps.append(decode_tokens.translate(transtab).strip())
                refs.append(
                    [
                        sent.translate(transtab).strip()
                        for sent in decode(
                            utils.strip_pad(sample[0]["target"][i], self.tgt_dict.pad()),
                            escape_unk=True,  # don't count <unk> as matches to the hypo
                        ).split('&&')
                    ]
                )
            for i in range(len(gen_out2)):
                decode_tokens = decode(gen_out2[i][0]["tokens"])
                hyps.append(decode_tokens.translate(transtab).strip())
                refs.append(
                    [
                        sent.translate(transtab).strip()
                        for sent in decode(
                            utils.strip_pad(sample[1]["target"][i], self.tgt_dict.pad()),
                            escape_unk=True,  # don't count <unk> as matches to the hypo
                        ).split('&&')
                    ]
                )
        else:
            gen_out = self.inference_step(generator, [model], sample)
            hyps, refs = [], []
            transtab = str.maketrans({key: None for key in string.punctuation})
            for i in range(len(gen_out)):
                decode_tokens = decode(gen_out[i][0]["tokens"])
                hyps.append(decode_tokens.translate(transtab).strip())
                refs.append(
                    [
                        sent.translate(transtab).strip()
                        for sent in decode(
                            utils.strip_pad(sample["target"][i], self.tgt_dict.pad()),
                            escape_unk=True,  # don't count <unk> as matches to the hypo
                        ).split('&&')
                    ]
                )

        if self.cfg.eval_print_samples:
            logger.info("example hypothesis: " + hyps[0])
            logger.info("example reference: " + ' && '.join(refs[0]))

        return hyps, refs
