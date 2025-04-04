import logging
import os
import time

import torch

from utils.utils_logging import AverageMeter


class CallBackLogging(object):
    def __init__(self, frequent, total_step, batch_size, resume=0, rem_total_steps=None):
        self.frequent: int = frequent
        self.time_start = time.time()
        self.total_step: int = total_step
        self.batch_size: int = batch_size

        self.resume = resume
        self.rem_total_steps = rem_total_steps

        self.init = False
        self.tic = 0

    def __call__(self, global_step, epoch: int, loss_verif: AverageMeter, loss_privacy: AverageMeter ):
        if global_step > 0 and global_step % self.frequent == 0:
            if self.init:

                speed_total: float = self.frequent * self.batch_size / (time.time() - self.tic)


                time_now = (time.time() - self.time_start) / 3600
                if self.resume:
                    time_total = time_now / ((global_step + 1) / self.rem_total_steps)
                else:
                    time_total = time_now / ((global_step + 1) / self.total_step)
                time_for_end = time_total - time_now


                msg = "Speed %.2f samples/sec   Loss %.4f Loss %.4f Epoch: %d   Global Step: %d   Required: %1.f hours" % (
                    speed_total, loss_verif.avg, loss_privacy.avg, epoch, global_step, time_for_end
                )

                logging.info(msg)
                loss_verif.reset()
                self.tic = time.time()
            else:
                self.init = True
                self.tic = time.time()




class CallBackModelCheckpoint(object):
    def __init__(self, output="./"):
        self.output: str = output
    def __call__(self, global_step, epoch, ftn_layers: torch.nn.Module, header: torch.nn.Module ):

        if global_step > 100 :
            if (epoch == 0) or ((epoch+1)%5 == 0):
                save_path = str(global_step) + f"epoch_{epoch}.pth"
            else:
                save_path = str(global_step) + "filter_weights.pth"

            torch.save(ftn_layers.state_dict(), os.path.join(self.output, save_path))
            if header:
                torch.save(header.state_dict(), os.path.join(self.output, str(global_step)+ "header.pth"))


