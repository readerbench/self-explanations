from collections import OrderedDict

import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, classification_report
from torch import optim
from transformers import BertModel, RobertaModel, get_linear_schedule_with_warmup
from torch.nn import functional as F
from torch.nn import ModuleList
from torchmetrics import F1Score, Accuracy
from torch.nn.functional import normalize
from core.data_processing.se_dataset import SelfExplanations


class BERTMTL(pl.LightningModule):
  def __init__(self, task_names, pretrained_bert_model, rb_feats=0, task_sample_weights=None, task_imp_weights=[], lr=1e-3,
               num_epochs=15, use_filtering=False, use_grad_norm=True):
    super().__init__()
    self.automatic_optimization = False
    if "roberta" in pretrained_bert_model:
      print(f"Training RoBERTa lr={lr}")
      self.bert = RobertaModel.from_pretrained(pretrained_bert_model, return_dict=False)
    else:
      print(f"Training BERT lr={lr}")
      self.bert = BertModel.from_pretrained(pretrained_bert_model, return_dict=False)
    self.use_grad_norm = use_grad_norm
    self.drop = nn.Dropout(p=0.2)
    self.tmp1 = nn.Linear(self.bert.config.hidden_size, 100)
    self.task_names = task_names
    task_classes = [SelfExplanations.MTL_CLASS_DICT[x] for x in self.task_names]
    self.num_tasks = len(task_names)
    if self.use_grad_norm:
      if len(task_imp_weights) == 0:
        self.task_imp_weights = torch.nn.Parameter(torch.ones(self.num_tasks).float())
        self.task_normalizer = self.num_tasks
      else:
        self.task_imp_weights = torch.nn.Parameter(torch.Tensor(task_imp_weights))
        self.task_normalizer = sum(task_imp_weights)
    else:
      self.task_imp_weights = [1 for _ in range(self.num_tasks)] if len(task_imp_weights) < self.num_tasks else task_imp_weights
    self.iteration_stat_aggregator = OrderedDict()
    self.train_iter_counter = 0
    self.val_iter_counter = 0
    self.task_sample_weights = task_sample_weights
    self.loss_f = None
    self.lr = lr
    self.num_epochs = num_epochs
    self.rb_feats = rb_feats
    self.use_filtering = use_filtering
    self.filtering = None
    if use_filtering:
      self.filtering = nn.Linear(8, 200)
    if self.rb_feats > 0:
      self.rb_feats_in = nn.Linear(self.rb_feats, 100)
      self.out = ModuleList([nn.Linear(200, task_classes[i]) for i in range(self.num_tasks)])
    else:
      self.out = ModuleList([nn.Linear(100, task_classes[i]) for i in range(self.num_tasks)])
    self.reset_iteration_stat_aggregator()
    self.initial_task_loss = None

  def reset_iteration_stat_aggregator(self):
    self.iteration_stat_aggregator[f"train_loss"] = 0
    self.iteration_stat_aggregator[f"test_loss"] = 0
    for key in self.task_names:
      self.iteration_stat_aggregator[f"{key}_train_loss"] = 0
      self.iteration_stat_aggregator[f"{key}_test_loss"] = 0
    self.train_iter_counter = 0
    self.val_iter_counter = 0

  def forward(self, input_ids, attention_mask, rb_feats_data=None, filter_data=None):
    _, pooled_output = self.bert(
      input_ids=input_ids,
      attention_mask=attention_mask
    )

    x = torch.tanh(self.tmp1(pooled_output))
    x = self.drop(x)

    if self.rb_feats > 0:
      feats = torch.tanh(self.rb_feats_in(rb_feats_data))
      x = torch.cat([feats, x], dim=1)
    if self.filtering:
      x = x * F.sigmoid(self.filtering(filter_data))

    x = [F.softmax(f(x)) for f in self.out]

    return x

  def process_batch_get_losses(self, batch):
    input_ids = batch['input_ids']
    attention_mask = batch['attention_mask']
    targets = batch['targets']
    if self.filtering:
      filter_data = batch['filter_data'].to(torch.float32)
      if self.rb_feats > 0:
        rb_feats_data = batch['rb_feats'].to(torch.float32)
        outputs = self(input_ids, attention_mask, rb_feats_data, filter_data)
      else:
        outputs = self(input_ids, attention_mask, filter_data=filter_data)
    else:
      if self.rb_feats > 0:
        rb_feats_data = batch['rb_feats'].to(torch.float32)
        outputs = self(input_ids, attention_mask, rb_feats_data)
      else:
        outputs = self(input_ids, attention_mask)

    if self.loss_f is None:
      if self.task_sample_weights is not None:
        self.loss_f = [nn.CrossEntropyLoss(weight=self.task_sample_weights[i].to(input_ids.device)) for i in range(self.num_tasks)]
      else:
        self.loss_f = [nn.CrossEntropyLoss() for _ in range(self.num_tasks)]
    partial_losses = [0 for _ in range(self.num_tasks)]

    transp_targets = targets.transpose(1, 0)
    for task_id in range(self.num_tasks):
      task_mask = transp_targets[task_id] != 9
      partial_losses[task_id] += self.loss_f[task_id](outputs[task_id][task_mask], transp_targets[task_id][task_mask])
    loss = sum([partial_losses[i] * self.task_imp_weights[i] for i in range(len(partial_losses))])

    return loss, partial_losses, transp_targets, outputs

  def training_step(self, batch, batch_idx):
    if self.use_grad_norm:
      return self.gradnorm_training_step(batch)
    return self.classic_training_step(batch)

  def classic_training_step(self, batch):
    loss, partial_losses, _, _ = self.process_batch_get_losses(batch)

    # Logging to TensorBoard by default
    self.iteration_stat_aggregator["train_loss"] += loss
    for i, task in enumerate(self.task_names):
      self.iteration_stat_aggregator[f"{task}_train_loss"] += partial_losses[i]
    self.train_iter_counter += 1

    return loss

  def gradnorm_training_step(self, batch):
    opt = self.optimizers()
    loss, partial_losses, _, _ = self.process_batch_get_losses(batch)

    if self.initial_task_loss is None:
      self.initial_task_loss = torch.Tensor([l.data.cpu() for l in partial_losses])

    opt.zero_grad()
    self.manual_backward(loss, retain_graph=True)

    self.task_imp_weights.grad.zero_()
    # get the gradient norms for each of the tasks
    norms = []
    for i in range(len(partial_losses)):
      # get the gradient of this task loss with respect to the shared parameters
      gygw = torch.autograd.grad(partial_losses[i], self.tmp1.parameters(), retain_graph=True)
      # compute the norm
      norms.append(torch.norm(torch.mul(self.task_imp_weights[i], gygw[0])))
    norms = torch.stack(norms)

    # compute the inverse training rate r_i(t)
    loss_ratio = torch.Tensor([l.data.detach() / self.initial_task_loss[i] for i, l in enumerate(partial_losses)])

    inverse_train_rate = loss_ratio / loss_ratio.mean()
    mean_norm = norms.mean().item()

    # compute the GradNorm loss
    alpha = 0.12
    constant_term = torch.tensor(mean_norm * (inverse_train_rate ** alpha), requires_grad=False).type_as(self.task_imp_weights)
    grad_norm_loss = torch.sum(torch.abs(norms - constant_term))
    self.task_imp_weights.grad = torch.autograd.grad(grad_norm_loss, self.task_imp_weights)[0]

    # Logging to TensorBoard by default
    self.iteration_stat_aggregator["train_loss"] += loss.item()
    for i, task in enumerate(self.task_names):
      self.iteration_stat_aggregator[f"{task}_train_loss"] += partial_losses[i].item()
    self.train_iter_counter += 1
    opt.step()

    self.task_imp_weights.data.copy_(normalize(self.task_imp_weights.data, p=1, dim=0) * self.task_normalizer)
    tw = self.task_imp_weights.data.cpu().numpy()

    for i in range(len(tw)):
      self.log(f"task_{i}_weight", tw[i])
    return loss

  def on_train_epoch_end(self):
    print(f"Reached epoch {self.current_epoch} end.")
    sch = self.lr_schedulers()
    sch.step()
    self.log("LR", sch.get_lr()[0])

  def validation_step(self, batch, batch_idx):
    loss, partial_losses, transp_targets, outputs = self.process_batch_get_losses(batch)

    out_targets = transp_targets.cpu()
    out_outputs = torch.Tensor([[y.argmax() for y in x] for x in outputs]).cpu()
    # Logging to TensorBoard by default
    self.iteration_stat_aggregator["test_loss"] += loss
    for i, task in enumerate(self.task_names):
      self.iteration_stat_aggregator[f"{task}_test_loss"] += partial_losses[i]
    self.val_iter_counter += 1
    return (out_targets, out_outputs)

  def test_step(self, batch, batch_idx):
    _, _, transp_targets, outputs = self.process_batch_get_losses(batch)

    out_targets = transp_targets.cpu()
    out_outputs = torch.Tensor([[y.argmax() for y in x] for x in outputs]).cpu()
    return (out_targets.reshape(-1).tolist(),
            out_outputs.reshape(-1).tolist(),
            batch["text_s"],
            batch["text_p"],
            batch["filter_data"].reshape(-1).tolist())

  def test_epoch_end(self, validation_step_outputs):
    for i, l in enumerate(validation_step_outputs):
      out_line = ""
      out_line += l[2][0] + "\t"
      out_line += l[3][0] + "\t"
      out_line += "\t".join(map(str, l[0])) + "\t"
      out_line += "\t".join(map(str, l[1])) + "\t"
      out_line += "\t".join(map(str, l[4])) + "\t"
      print(f"{i}\t{out_line}")
    targets = [x[0] for x in validation_step_outputs]
    outputs = [x[1] for x in validation_step_outputs]

    task_targets = [[] for _ in range(self.num_tasks)]
    task_outputs = [[] for _ in range(self.num_tasks)]

    for i in range(self.num_tasks):
      print("=" * 5 + self.task_names[i] + "=" * 20)
      for j in range(len(targets)):
        task_targets[i].append(targets[j][i])
        task_outputs[i].append(outputs[j][i])

      task_targets[i] = torch.cat(task_targets[i])
      task_outputs[i] = torch.cat(task_outputs[i])
      task_mask = task_targets[i] != 9
      filtered_targets = task_targets[i][task_mask].int()
      filtered_outputs = task_outputs[i][task_mask].int()
      print(confusion_matrix(filtered_targets, filtered_outputs))
      print(classification_report(filtered_targets, filtered_outputs))

  def validation_epoch_end(self, validation_step_outputs):
    targets = [x[0] for x in validation_step_outputs]
    outputs = [x[1] for x in validation_step_outputs]

    task_targets = [[] for _ in range(self.num_tasks)]
    task_outputs = [[] for _ in range(self.num_tasks)]

    for i in range(self.num_tasks):
      for j in range(len(targets)):
        task_targets[i].append(targets[j][i])
        task_outputs[i].append(outputs[j][i])

      task_targets[i] = torch.cat(task_targets[i])
      task_outputs[i] = torch.cat(task_outputs[i])
      task_mask = task_targets[i] != 9
      filtered_targets = task_targets[i][task_mask].int()
      filtered_outputs = task_outputs[i][task_mask].int()
      f1 = F1Score(num_classes=SelfExplanations.MTL_CLASS_DICT[self.task_names[i]])
      acc = Accuracy()

      for key in self.iteration_stat_aggregator:
        if key.endswith("test_loss") and self.val_iter_counter > 0:
          self.log(key, self.iteration_stat_aggregator[key] / self.val_iter_counter)
        elif self.train_iter_counter > 0:
          self.log(key, self.iteration_stat_aggregator[key] / self.train_iter_counter)
      self.reset_iteration_stat_aggregator()
      self.log(f"acc_{self.task_names[i]}", acc(filtered_outputs, filtered_targets))
      self.log(f"f1_{self.task_names[i]}", f1(filtered_outputs, filtered_targets))
      print(confusion_matrix(filtered_targets, filtered_outputs))
      print(classification_report(filtered_targets, filtered_outputs))

  def configure_optimizers(self):
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
      # BERT params - with WD
      {'params': [p for n, p in self.named_parameters() if
                  not any(nd in n for nd in no_decay) and n.find("bert") != -1],
       'weight_decay': 0.0001, 'lr': 1e-5},
      # BERT params - no WD
      {'params': [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay) and n.find("bert") != -1],
       'weight_decay': 0.0, 'lr': 1e-5},
      # task-level weights
      {'params': [p for n, p in self.named_parameters() if n.find("task_imp_weights") != -1],
       'weight_decay': 0.0001, 'lr': 1e-2},
      # non-BERT params - with WD
      {'params': [p for n, p in self.named_parameters() if
                  not any(nd in n for nd in no_decay) and n.find("bert") == -1 and n.find("task_imp_weights") == -1],
       'weight_decay': 0.0001},
      # non-BERT params - no WD
      {'params': [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay) and n.find("bert") == -1],
       'weight_decay': 0.0}
    ]
    optimizer = optim.AdamW(optimizer_grouped_parameters, lr=self.lr)

    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.num_epochs//5, num_training_steps=self.num_epochs)
    scheduler.step()
    return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch'
                }
            }

