"""
Accompanying implementation for paper:
BACKTRACKING GRADIENT DESCENT METHOD FOR GENERAL C1 FUNCTIONS, WITH APPLICATIONS TO DEEP LEARNING
https://arxiv.org/pdf/1808.05160.pdf
Forked and inspired by https://github.com/davidtvs/pytorch-lr-finder
"""

from __future__ import print_function, with_statement, division
import torch
from tqdm.autonotebook import tqdm
#import matplotlib.pyplot as plt
import numpy as np
import copy

def change_lr(optim,lr_current):
    for g in optim.param_groups:
        g['lr'] = lr_current

class LRFinder(object):
    """Learning rate finder using backtracking line search, full automatic pipeline
        >>> lr_finder = LRFinder(model, optimizer, criterion, device)
        >>> lr_finder.backtrack(trainloader,applied_optmizer,alpha = 0.5, beta = 0.5, num_iter = 20)

    Cyclical Learning Rates for Training Neural Networks: https://arxiv.org/abs/1506.01186
    fastai/lr_find: https://github.com/fastai/fastai
    lrfinder: https://github.com/davidtvs/pytorch-lr-finder

    """

    def __init__(self, model, optimizer, criterion, device=None):
        self.model = model
        self.optimizer = optimizer
        self.lr_current = optimizer.param_groups[0]['lr']
        self.criterion = criterion
        self.best_loss = None
        self.params = list(self.model.parameters())
        self.grads = {}
        for i in range(len(self.params)):
            self.grads[i] = []
        self.lr_min = 1e-6 - 1e-8
        self.lr_max = 100  + 1e-8
        self.thres = 1e-7
            
        # Save the original state of the model and optimizer so they can be restored if
        # needed
        self.model_state = copy.deepcopy(model.state_dict())
        self.model_device = next(self.model.parameters()).device
        self.optimizer_state = optimizer.state_dict()

        # If device is None, use the same as the model
        if device:
            self.device = device
        else:
            self.device = self.model_device

    def reset(self):
        """Restores the model to their initial states."""
        self.model.load_state_dict(self.model_state)
        self.model.to(self.model_device)
        self.optimizer.load_state_dict(self.optimizer_state)
        
    def rollback(self, state):
        """Restores the model to their previous states."""
        self.model.load_state_dict(state)


    def backtrack(self,trainloader,alpha = 0.5, beta = 0.5, num_iter = 20, lr_justified = True):
        """Performs the learning rate backtracking line search.

        Arguments:
        train_loader (torch.utils.data.DataLoader): the training set data laoder.
        alpha, beta: hyper parameters for backtracking line search
        num_iter: number of mini-bactch iterations for backtracking
        """
        self.alpha = alpha
        self.beta = beta
        self.num_batches = len(trainloader.dataset)/trainloader.batch_size
        self.lr_justified = lr_justified
        
        self.ratio = 1/np.sqrt(self.num_batches) if self.lr_justified else 1

        # Reset test results
        self.model_state = copy.deepcopy(self.model.state_dict())
        self.history = {'lr':[],'loss':[]}
        self.best_loss = None
        for i in range(len(self.params)):
            self.grads[i] = []
        self.cond = True

        # Move the model to the proper device
        self.model.to(self.device)

        lr_current = self.optimizer.param_groups[0]['lr']
        # Create an iterator to get data batch by batch
        iterator = iter(trainloader)
        for iteration in tqdm(range(num_iter)):
            # Get a new set of inputs and labels
            try:
                inputs, labels = next(iterator)
            except StopIteration:
                iterator = iter(trainloader)
                inputs, labels = next(iterator)

            # Train on batch and retrieve loss
            self._train_batch(inputs, labels,lr_current)
        self.lr_BT = np.mean(self.history['lr']) # final ouput of learning rate finder
        if self.lr_justified:
            self.lr_BT = self.lr_BT*self.ratio # final ouput of learning rate finder
        print('Line Search Backtracking completed, final learning rate %.5f'%self.lr_BT)
        self.reset()
    
    def _train_batch(self, inputs, labels, lr_current):
        if len(self.history['lr']) > 0:
                self.lr_current = max([self.lr_min,np.mean(self.history['lr'])])
            
        self.state = copy.deepcopy(self.model.state_dict())
        self.model.train()

        # Move data to the correct device
        inputs, labels = inputs.to(self.device), labels.to(self.device)

        # Forwars and backward, compute gradients
        # Forward pass
        self.optimizer.zero_grad()
        loss = self.criterion(self.model(inputs), labels)

        # Backward pass: compute gradients
        loss.backward()
        self.loss = loss.item()
        self.norm_squared = np.sum([torch.norm(param.grad, p=2).item()**2 for param in self.params])
        
        # Backtracking loop
        self.track_batch(inputs, labels)
    
        if (self.check > 0 or np.isnan(self.check)) and self.cond:
            while (self.check > 0 or np.isnan(self.check)) and self.cond:
                self.lr_current = self.lr_current*self.beta                           
                self.track_batch(inputs, labels)

        elif self.check < 0 and self.cond:
            while self.check < 0 and self.cond:
                self.lr_current = self.lr_current/self.beta                           
                self.track_batch(inputs, labels)
            self.lr_current = self.lr_current*self.beta
            self.loss_new = self.loss_new_prev
        
        self.history['lr'].append(self.lr_current)
        self.history['loss'].append(self.loss_new)
        
        # official update with justified learning rate:
        self.rollback(state = self.state)
        if self.lr_justified:
            self.lr_current = self.lr_current* self.ratio
        change_lr(self.optimizer,self.lr_current)
        self.optimizer.step()
            
    def track_batch(self, inputs, labels):
        self.rollback(state = self.state)
        change_lr(self.optimizer,self.lr_current)
        self.optimizer.step()
        
        try: self.loss_new_prev = self.loss_new 
        except: self.loss_new_prev = self.loss
        
        self.loss_new = self.criterion(self.model(inputs), labels).item()
        self.check = self.loss_new - self.loss + self.alpha*self.lr_current*self.norm_squared
        
        # check condition
        self.cond = self.lr_current * self.norm_squared > self.thres and (self.lr_current >=  self.lr_min and self.lr_current <= self.lr_max) 
        
    def _validate(self, dataloader):
        # Set model to evaluation mode and disable gradient computation
        running_loss = 0
        self.model.eval()
        with torch.no_grad():
            for batch_idx, (inputs, labels) in enumerate(dataloader):
                # Move data to the correct device
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # Forward pass and loss computation
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                running_loss += loss.item() * inputs.size(0)

        return running_loss / len(dataloader.dataset)
        
