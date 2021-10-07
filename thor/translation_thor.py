# Copyright (c) Microsoft. All rights reserved.
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import logging
import math
import json
import torch
import torch.nn.functional as F

from argparse import Namespace
from dataclasses import dataclass, field
from omegaconf import II

from fairseq import metrics, models
from fairseq.data import encoders
from fairseq.optim.amp_optimizer import AMPOptimizer
from fairseq.tasks import register_task
from fairseq.tasks.translation import TranslationConfig, TranslationTask


logger = logging.getLogger(__name__)


def symmetric_KL_loss(p, q, pad_mask):
    """ symmetric KL-divergence 1/2*(KL(p||q)+KL(q||p)) """
    p, q, pad_mask = p.float(), q.float(), pad_mask.view(-1)
    dict_size = q.size(-1)
    non_pad_mask = ~pad_mask
    p = p.view(-1, dict_size)[non_pad_mask]
    q = q.view(-1, dict_size)[non_pad_mask]
    loss = (p - q) * (torch.log(p) - torch.log(q))
    return 0.5 * loss.sum()


@dataclass
class TranslationThorConfig(TranslationConfig):
    num_experts: int = field(
        default=2,
        metadata={"help": "number of experts"},
    )
    consistency_alpha: float = field(
        default=1.0,
        metadata={"help": "weight of the consistency loss"},
    )
    inference_level: int = field(
        default=0,
        metadata={"help": "0 for token level, 1 for sentence level"},
    )
    sentence_avg: bool = II("optimization.sentence_avg")


@register_task("translation_thor", dataclass=TranslationThorConfig)
class TranslationThorTask(TranslationTask):
    """
    Translation task for Switch Transformer models.

    Args:
        src_dict (~fairseq.data.Dictionary): dictionary for the source language
        tgt_dict (~fairseq.data.Dictionary): dictionary for the target language

    .. note::

        The translation task is compatible with :mod:`fairseq-train`,
        :mod:`fairseq-generate` and :mod:`fairseq-interactive`.

    The translation task provides the following additional command-line
    arguments:

    .. argparse::
        :ref: fairseq.tasks.translation_parser
        :prog:
    """

    cfg: TranslationThorConfig

    def __init__(self, cfg: TranslationThorConfig, src_dict, tgt_dict):
        super().__init__(cfg, src_dict, tgt_dict)

    def build_model(self, cfg):
        model = models.build_model(cfg, self)

        if self.cfg.eval_bleu:
            detok_args = json.loads(self.cfg.eval_bleu_detok_args)
            self.tokenizer = encoders.build_tokenizer(
                Namespace(tokenizer=self.cfg.eval_bleu_detok, **detok_args)
            )

            gen_args = json.loads(self.cfg.eval_bleu_args)
            self.sequence_generator = self.build_generator(
                [model], Namespace(**gen_args)
            )

        return model

    def _get_loss(self, sample, model, criterion, expert_num=None):
        assert hasattr(
            criterion, "compute_loss"
        ), "translation_thor task requires the criterion to implement the compute_loss() method"

        encoder_out = model.encoder(
            src_tokens=sample["net_input"]["src_tokens"],
            src_lengths=sample["net_input"]["src_lengths"],
            expert_num=expert_num,
        )
        net_output = model.decoder(
            prev_output_tokens=sample["net_input"]["prev_output_tokens"],
            encoder_out=encoder_out,
            src_lengths=sample["net_input"]["src_lengths"],
            expert_num=expert_num,
        )
        loss, nll_loss = criterion.compute_loss(model, net_output, sample, reduce=True)

        logits = net_output[0].float()
        logits = F.softmax(logits, dim=-1)

        sample_size = (
            sample["target"].size(0) if criterion.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
            "selection": net_output[1].get("selection", None),
        }

        return loss, logits, sample_size, logging_output

    def train_step(
        self, sample, model, criterion, optimizer, update_num, ignore_grad=False
    ):
        model.train()
        model.set_num_updates(update_num)

        with torch.autograd.profiler.record_function("forward"):
            with torch.cuda.amp.autocast(enabled=(isinstance(optimizer, AMPOptimizer))):
                expert_num = None
                loss1, logits1, sample_size, logging_output1 = self._get_loss(sample, model, criterion, expert_num)

        with torch.autograd.profiler.record_function("forward"):
            with torch.cuda.amp.autocast(enabled=(isinstance(optimizer, AMPOptimizer))):
                expert_num = None
                loss2, logits2, sample_size, logging_output2 = self._get_loss(sample, model, criterion, expert_num)

        pad_mask = sample["target"].eq(criterion.padding_idx)
        consistency_loss = symmetric_KL_loss(logits1, logits2, pad_mask)
        loss = loss1 + loss2 + consistency_loss * self.cfg.consistency_alpha

        logging_output = {
            "loss": torch.tensor([logging_output1["loss"], logging_output2["loss"]]),
            "nll_loss": torch.tensor([logging_output1["nll_loss"], logging_output2["nll_loss"]]),
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
            "consistency": consistency_loss.data,
        }

        if ignore_grad:
            loss *= 0

        with torch.autograd.profiler.record_function("backward"):
            optimizer.backward(loss)
        return loss, sample_size, logging_output

    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)

        # follows the reduce_metrics() function in label_smoothed_cross_entropy.py
        loss_sum = sum(log.get("consistency", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        metrics.log_scalar(
            "consistency", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
