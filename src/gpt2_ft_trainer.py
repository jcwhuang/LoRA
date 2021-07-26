#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import argparse
import time
import math
import os, sys

import torch
torch.set_printoptions(threshold=100000)

import numpy as np

from gpu import (
  add_gpu_params, 
  GpuArguments,
  parse_gpu, 
  distributed_opt, 
  distributed_gather, 
  distributed_sync, 
  cleanup
)

from optimizer import (
  create_adam_optimizer, 
  create_optimizer_scheduler, 
  OptimizerArguments,
  create_adam_optimizer_from_args
)

from data_utils import FT_Dataset_Trainer # BinCorpus, BinLMOrderedIterator
from model import GPT2Config, GPT2LMModel
from exp_utils import create_exp_dir
  

from transformers import Trainer, TrainingArguments, HfArgumentParser
from transformers.trainer_utils import PredictionOutput

import itertools

from torch.utils.data import DataLoader
from dataclasses import dataclass, field


@dataclass
class DataTrainingArguments:

  train_data: str = field(
          metadata={"help": 'location of training data corpus'})

  valid_data: str = field(
          metadata={"help": 'location of validation data corpus'})

  train_batch_size: int = field(
          default=8, 
          metadata={"help": 'training batch size'})

  valid_batch_size: int = field(
          default=4, 
          metadata={"help": 'validation batch size'})

  grad_acc: int = field(
          default=1,
          metadata={"help": 'gradient accumlate'})

  seq_len: int = field(
          default=512,
          metadata={"help": 'number of tokens to predict.'})

  model_card: str = field(
          default='gpt2.sm', 
           
          metadata={"help": 'model names.',
              "choices": ['gpt2.sm', 'gpt2.md', 'gpt2.lg']})

  init_checkpoint: str = field(
          default=None,
          metadata={"help": 'initial checkpoint.'})

  log_interval: int = field(
          default=100, 
          metadata={"help": 'log interval.'})

  eval_interval: int = field(
          default=2000, 
          metadata={"help": 'eval interval.'})

  save_interval: int = field(
          default=500, 
          metadata={"help": 'save interval.'})

  work_dir: str = field(
          default=os.getenv('PT_OUTPUT_DIR', 'gpt2_model'), 
          metadata={"help": 'working folder.'})

  lora_dim: int = field(
          default=0, 
          metadata={"help": 'lora attn dimension.'})

  lora_alpha: int = field(
          default=128, 
          metadata={"help": 'lora attn alpha.'})

  obj: str = field(
          default='clm', 
          metadata={"help": 'language model training objective.',
                    "choices": ['jlm', 'clm']})

  lora_dropout: float = field(
          default=0.0, 
          metadata={"help": 'dropout probability for lora layers.'})

  label_smooth: float = field(
          default=0.0, 
          metadata={"help": 'label smoothing.'})

  random_seed: int = field(
          default=1, 
          metadata={"help": 'random seed.'})

  prefix_len: int = field(
          default=0,
          metadata={"help": "prefix length"})

  infix_len: int = field(
          default=0,
          metadata={"help": "infix length"})

  roll_interval: int = field(
          default=-1,
          metadata={"help": "rolling interval"})

  roll_lr: float = field(
          default=0.00001,
          metadata={"help": "rolling learning rate"})

  roll_step: int = field(
          default=100,
          metadata={"help": "rolling step"})

  device: int = field(
          default=0, 
          metadata={"help": 'GPU device'})


def print_args(args):
  print('=' * 100)
  for k, v in args.__dict__.items():
    print('    - {} : {}'.format(k, v))
  print('=' * 100)


class AverageMeter(object):
  """Computes and stores the average and current value
     Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
  """
  def __init__(self):
    self.reset()

  def reset(self):
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0

  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count


class MyTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        inputs = {key: value.to(args.device) for key, value in inputs.items()}

        lm_logits = model(**inputs)
        _batch, _len = inputs['input_ids'].shape
        # copied from GPT2LMModel
        lm_labels, lm_mask = inputs['lm_labels'], inputs['lm_mask']
        if args.label_smooth > 0.0001:
            logprobs = torch.nn.functional.log_softmax(lm_logits.view(-1, lm_logits.size(-1)), dim=-1)
            nll_loss = -logprobs.gather(dim=-1, index=lm_labels.view(-1).unsqueeze(1))
            nll_loss = nll_loss.squeeze(1)
            smooth_loss = -logprobs.mean(dim=-1)
            loss = (1.0 - args.label_smooth) * nll_loss + args.label_smooth * smooth_loss
            loss = loss.view(_batch, _len)
        else:
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-1, reduce=False)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), lm_labels.view(-1)).view(_batch, _len)

        if lm_mask is None:
            lm_mask = torch.ones(loss.shape, dtype=loss.dtype, device=loss.device)
        loss = loss * lm_mask 

        loss = loss.sum() / (lm_mask.sum() + 0.0001)

        return (loss, lm_logits) if return_outputs else loss

    def prediction_loop(self, dataloader, description, prediction_loss_only=None, ignore_keys=None, metric_key_prefix="eval"):
        self.model.eval()
        total_loss = 0.
        start_time = time.time()
      
        avg_lm_loss = AverageMeter()
      
        with torch.no_grad():
          for idx, data in enumerate(dataloader):
            _loss = self.compute_loss(self.model, data)
            loss = _loss.mean() 
            
            avg_lm_loss.update(loss.item())
      
            if idx % 100 == 0:
              print('eval samples:', idx, 'loss:', loss.float())
      
          total_time = time.time() - start_time
          print('average loss', avg_lm_loss.avg)
        metrics = {f"{metric_key_prefix}_avg_loss": avg_lm_loss.avg, f"{metric_key_prefix}_ppl": math.exp(avg_lm_loss.avg)}
        return PredictionOutput(predictions=None, label_ids=None, metrics=metrics)


def evaluate(model, valid_loader, args):
  model.eval()
  total_loss = 0.
  start_time = time.time()

  avg_lm_loss = AverageMeter()

  with torch.no_grad():
    for idx, data in enumerate(valid_loader):
      data = {key: value for key, value in data.items()}

      _input = data['input'].to(args.device)
      _target = data['target'].to(args.device)
      _msk = data['mask'].to(args.device)

      _lm_logits, _loss = model(_input, lm_labels=_target, lm_mask=_msk) 
      loss = _loss.mean() 
      
      avg_lm_loss.update(loss.item())

      if idx % 100 == 0:
        print('eval samples:', idx, 'loss:', loss.float())

    total_time = time.time() - start_time
    print('average loss', avg_lm_loss.avg)
  return avg_lm_loss.avg, math.exp(avg_lm_loss.avg)


def train_validate(model, optimizer, scheduler, train_loader, valid_loader, args, train_step=0, epoch=0):
  model.train()
  avg_lm_loss = AverageMeter()
  print('start to train the model................', epoch)
  log_start_time = time.time()
  best_val_ppl = None

  train_loader.sampler.set_epoch(epoch)

  for idx, data in enumerate(train_loader):
    data = {key: value for key, value in data.items()}

    _input = data['input'].to(args.device)
    _target = data['target'].to(args.device)
    _msk = data['mask'].to(args.device)
    _lm_logits, _lm_loss = model(_input, lm_labels=_target, lm_mask=_msk, label_smooth=args.label_smooth) 
    _lm_loss = _lm_loss.mean() 

    train_step += 1
    is_update = True if train_step % args.grad_acc == 0 else False
    avg_lm_loss.update(_lm_loss.item())
    optimizer_step(_lm_loss/(args.grad_acc), optimizer, model, scheduler, args, is_update=is_update)
    
    if train_step % args.log_interval == 0: 
      elapsed = time.time() - log_start_time

      log_str = '| epoch {:3d} step {:>8d} | {:>6d} batches | lr {:.3g}' \
                '| ms/batch {:5.2f} | loss {:5.2f} | avg loss {:5.2f} | ppl {:5.2f}'.format(
                epoch, train_step, idx + 1, optimizer.param_groups[0]['lr'], 
                elapsed * 1000 / args.log_interval, avg_lm_loss.val, avg_lm_loss.avg, math.exp(avg_lm_loss.avg)) 

      if args.rank == 0: 
        print(log_str)
      log_start_time = time.time()
      avg_lm_loss.reset()
    
    if train_step % args.save_interval == 0: 
      if args.rank == 0:
        model_path = os.path.join(args.work_dir, 'model.'+str(train_step)+'.pt')
        print('saving checkpoint', model_path)
        torch.save({'model_state_dict': model.state_dict()}, model_path)
      distributed_sync(args)

    # evaluation interval
    if train_step % args.eval_interval == 0:
      eval_start_time = time.time()

      valid_loss, valid_ppl = evaluate(model, valid_loader, args)

      if best_val_ppl is None or valid_ppl < best_val_ppl:
        best_val_ppl = valid_ppl
        
      log_str = '| Eval {:3d} at step {:>8d} | time: {:5.2f}s ' \
                '| valid loss {:5.2f} | valid ppl {:5.2f} | best ppl {:5.2f} '.format(
                train_step // args.eval_interval, train_step,
                (time.time() - eval_start_time), valid_loss, valid_ppl, best_val_ppl)

      if args.rank == 0:
        print('-' * 100)
        print(log_str)
        print('-' * 100)

      model.train()
      distributed_sync(args)

    if train_step == args.max_step:
      break

  if args.rank == 0:
    model_path = os.path.join(args.work_dir, 'model.'+str(train_step)+'.pt')
    print('saving checkpoint', model_path)
    torch.save({'model_state_dict': model.state_dict()}, model_path) 
  distributed_sync(args)
  return train_step


if __name__ == '__main__':
  parser = HfArgumentParser((DataTrainingArguments, TrainingArguments))
  args, training_args = parser.parse_args_into_dataclasses()

  print_args(args)
  print_args(training_args)
  
  #if args.rank == 0:
  #  args.logging = create_exp_dir(args.work_dir)

  train_data =  FT_Dataset_Trainer(
    args.train_data, args.train_batch_size, args.seq_len, 
    joint_lm=args.obj=='jlm', prefix_len=args.prefix_len, infix_len=args.infix_len
  )   
  
  valid_data = FT_Dataset_Trainer(
    args.valid_data, args.valid_batch_size, args.seq_len,
    prefix_len=args.prefix_len, infix_len=args.infix_len
  )

  if args.model_card == 'gpt2.sm':
    config = GPT2Config(
      n_embd=768, n_layer=12, n_head=12, 
      lora_attn_dim=args.lora_dim, lora_attn_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
    )
  elif args.model_card == 'gpt2.md':
    config = GPT2Config(
      n_embd=1024, n_layer=24, n_head=16, 
      lora_attn_dim=args.lora_dim, lora_attn_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
    )
  elif args.model_card == 'gpt2.lg':
    config = GPT2Config(
      n_embd=1280, n_layer=36, n_head=20, 
      lora_attn_dim=args.lora_dim, lora_attn_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
    )

  lm_net = GPT2LMModel(config)
  if args.init_checkpoint is not None:
    print('loading model pretrained weight.')
    lm_net.load_weight(torch.load(args.init_checkpoint))  

  lm_net = lm_net.cuda()

  try:
    if training_args.do_train:
      trainer = MyTrainer(
              model=lm_net,
              args=training_args,
              train_dataset=train_data,
              eval_dataset=valid_data
              )
      train_result = trainer.train()
      metrics = train_result.metrics

      max_train_samples = (
                          data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
                                  )
      metrics["train_samples"] = min(max_train_samples, len(train_dataset))

      trainer.log_metrics("train", metrics)
      trainer.save_metrics("train", metrics)
      trainer.save_state()

  except KeyboardInterrupt:
    print('-' * 100)
    print('Exiting from training early')

  distributed_sync(args)
  print('cleanup dist ...')
  cleanup(args)

