import torch
from recbole.trainer import Trainer
from recbole.utils import early_stopping
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.optim as optim
from torch.optim.swa_utils import AveragedModel, SWALR
from tqdm import tqdm

class SAMTrainer(Trainer):
    def __init__(self, config, model):
        super(SAMTrainer, self).__init__(config, model)
        self.enable_sam = config['enable_sam'] if 'enable_sam' in config else False
        self.sam_rho = config['sam_rho'] if 'sam_rho' in config else 0.05
        self.enable_swa = config['enable_swa'] if 'enable_swa' in config else False
        self.swa_start_epoch = config['swa_start_epoch'] if 'swa_start_epoch' in config else 5
        self.swa_lr = config['swa_lr'] if 'swa_lr' in config else 0.05
        
        if self.enable_swa:
            self.swa_model = AveragedModel(self.model)
            self.swa_scheduler = SWALR(self.optimizer, swa_lr=self.swa_lr)
            
    def _train_epoch(self, train_data, epoch_idx, loss_func=None, show_progress=False):
        self.model.train()
        loss_func = loss_func or self.model.calculate_loss
        total_loss = 0.
        iter_data = (
            tqdm(
                train_data,
                total=len(train_data),
                desc="Train epoch {}".format(epoch_idx),
                dynamic_ncols=True,
            )
            if show_progress
            else train_data
        )
        
        for batch_idx, interaction in enumerate(iter_data):
            interaction = interaction.to(self.device)
            self.optimizer.zero_grad()
            
            # Standard forward pass
            loss = loss_func(interaction)
            loss.backward()
            
            if self.enable_sam:
                # SAM Step 1: Ascent
                # Save gradients
                grads = []
                for p in self.model.parameters():
                    if p.grad is not None:
                        grads.append(p.grad.clone())
                        
                # Clip gradients
                if self.clip_grad_norm:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)

                # Save old parameters
                self._save_state()
                
                # Move to neighbor (Ascent Step)
                self._ascent_step()
                
                # Zero grad for second pass
                self.optimizer.zero_grad()
                
                # Second forward pass
                loss_adv = loss_func(interaction)
                loss_adv.backward()
                
                # Restore original parameters
                self._restore_state()
                
                # Add gradients from first pass (optional, sometimes SAM just uses second pass grad at original point)
                # Standard SAM: update at w using grad at w+e
                # So we just step using the current grad (which is from w+e) but apply it to w (which we restored)
                
                self.optimizer.step()
                
            else:
                if self.clip_grad_norm:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
                self.optimizer.step()
                
            total_loss += loss.item()
            
        return total_loss / len(train_data)

    def _save_state(self):
        self.state = {}
        for name, param in self.model.named_parameters():
             if param.grad is not None:
                self.state[name] = param.data.clone()

    def _ascent_step(self):
        # e_w = rho * g / ||g||
        grad_norm = torch.norm(torch.stack([p.grad.norm(p=2) for p in self.model.parameters() if p.grad is not None]), p=2)
        scale = self.sam_rho / (grad_norm + 1e-12)
        
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                param.data.add_(param.grad * scale)
                
    def _restore_state(self):
        for name, param in self.model.named_parameters():
            if name in self.state:
                param.data = self.state[name]

    def _full_sort_predict(self, interaction):
        if self.enable_swa and self.eval_type == 'valid': # Use SWA model for validation if ready
             # This is complex because RecBole swaps models. 
             # Simplification: Only create SWA model at end.
             pass
        return super()._full_sort_predict(interaction)

    def fit(self, train_data, valid_data=None, verbose=True, saved=True, show_progress=False, callback_fn=None):
        # Override to handle SWA scheduler update
        return super().fit(train_data, valid_data, verbose, saved, show_progress, callback_fn)

# Simplified SAM Trainer focusing just on the SAM step logic override
# Since fully overriding fit/train_epoch in RecBole is heavy due to many dependencies, 
# we might be better off just using a custom Optimizer class passed to RecBole configuration if possible.
# But RecBole constructs optimizer from string name.
