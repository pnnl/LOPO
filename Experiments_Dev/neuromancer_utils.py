from neuromancer.loggers import BasicLogger
from neuromancer.callbacks import Callback
import torch

class CallbackChild(Callback):
    def begin_train(self,trainer):
        trainer.train_losses_epoch = []
        trainer.dev_losses_epoch   = []
        trainer.resid_L2_dev_epoch = []
    def end_epoch(self,trainer,output):
        resid_L2_dev = torch.norm(output['dev_cnv_gap'],p=2,dim=1).mean().item()
        trainer.train_losses_epoch.append(output['train_loss'].item())
        trainer.dev_losses_epoch.append(output['dev_loss'].item())
        trainer.resid_L2_dev_epoch.append(resid_L2_dev)

class LoggerChild(BasicLogger):
    def log_artifacts(self, artifacts):
        return
