from model_evaluator import ModelEvaluator

import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import torch.nn as nn
import numpy as np
import abc

class CustomModel(pl.LightningModule):

    def __init__(self, hparams, training_dataset, validation_dataset, labels, model_to_use):
        super().__init__()

        self.hparams = hparams
        self.training_dataset = training_dataset
        self.validation_dataset = validation_dataset
        self.labels = labels

        self.define_model(model_to_use)

    @abc.abstractmethod
    def define_model(self, model_to_use):
        pass

    @abc.abstractmethod
    def forward(self, ids, mask, token_type_ids):
        pass

    def loss_fn(self, outputs, targets):
      return torch.nn.BCEWithLogitsLoss()(outputs, targets)

    def general_step(self, batch, batch_idx, mode):
      ids = batch['ids']
      mask = batch['mask']
      token_type_ids = batch['token_type_ids']
      targets = batch['targets']

      outputs = self.forward(ids, mask, token_type_ids)

      return {'outputs': outputs, 'targets': targets}

    #******Training******
    #This method runs on each GPU
    def training_step(self, batch, batch_idx):
      return self.general_step(batch, batch_idx, "train")
    
    #This method aggregates the results of training_step in the different GPUs
    def training_step_end(self, aggregated_outputs):
      loss = self.loss_fn(aggregated_outputs["outputs"], aggregated_outputs["targets"])
      self.log('training_loss',loss)
      return {'loss':loss}
    
    #This method runs at the end of each epoch
    def training_epoch_end(self, results_of_each_batch):
      pass

    #******Validation******
    #This method runs in each GPU
    def validation_step(self, batch, batch_idx):
      return self.general_step(batch, batch_idx, "val")

    #This method aggregates the results of validation_step in the different GPUs
    def validation_step_end (self, aggregated_outputs):
      outputs = torch.sigmoid(aggregated_outputs['outputs']).cpu().detach().numpy().tolist()
      predictions = (np.array(outputs) >= 0.5).astype(int)

      targets = aggregated_outputs['targets'].cpu().detach().numpy()

      return {'predictions': predictions, 'targets': targets}

    #This method runs at the end of each epoch
    def validation_epoch_end(self, results_of_each_batch):
      predictions = np.empty([0,len(self.labels)])
      targets = np.empty([0,len(self.labels)])

      for result in results_of_each_batch:
        predictions = np.concatenate((predictions,result['predictions']))
        targets = np.concatenate((targets,result['targets']))
      
      total_accuracy, accuracy_per_label = self.evaluate_results(predictions, targets)
      self.log('total_accuracy', total_accuracy)
      self.log('accuracy_per_label',accuracy_per_label)
    
    #******Test******
    #This method runs in each GPU
    def test_step(self, batch, batch_idx):
      return self.general_step(batch, batch_idx, "val")
    
    #This method aggregates the results of validation_step in the different GPUs
    def test_step_end (self, aggregated_outputs):
      outputs = torch.sigmoid(aggregated_outputs['outputs']).cpu().detach().numpy().tolist()
      predictions = (np.array(outputs) >= 0.5).astype(int)

      targets = aggregated_outputs['targets'].cpu().detach().numpy()

      return {'predictions': predictions, 'targets': targets}
    
    #This method runs at the end of each epoch
    def test_epoch_end(self, results_of_each_batch):
      predictions = np.empty([0,len(self.labels)])
      targets = np.empty([0,len(self.labels)])

      for result in results_of_each_batch:
        predictions = np.concatenate((predictions,result['predictions']))
        targets = np.concatenate((targets,result['targets']))
      
      total_accuracy, accuracy_per_label = self.evaluate_results(predictions, targets)
      print(f"Total accuracy: {total_accuracy}")
      print(f"Accuracy per label: {accuracy_per_label}")

    def evaluate_results(self, predictions, targets):
      #binary relevance
      total_accuracy = ModelEvaluator.get_total_accuracy(targets, predictions)
      #one vs rest
      accuracy_per_label = ModelEvaluator.get_accuracy_per_label(self.labels,targets,predictions)
      return total_accuracy, accuracy_per_label

    def configure_optimizers(self):
      return torch.optim.Adam(params = self.parameters(), lr=self.hparams["learning_rate"])

    #******Dataloaders******
    def train_dataloader(self):
      return DataLoader(self.training_dataset, batch_size=self.hparams["train_batch_size"], shuffle= self.hparams["shuffle"])

    def val_dataloader(self):
      return DataLoader(self.validation_dataset, batch_size=self.hparams["validation_batch_size"], shuffle= False)