import os
from abc import abstractmethod

import time
import torch
import pandas as pd
from numpy import inf
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import wandb
from utils.wandb import log_image_report_table


class BaseTrainer(object):
    def __init__(self, model, criterion, metric_ftns, gpu_id, optimizer, args):
        self.args = args
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
            
        if args.n_gpu > 1:
            self.DDP_flag = True
            self.model = DDP(self.model, device_ids=[gpu_id])
            if gpu_id == 0:
                print(f"Training with {args.n_gpu} GPUs in a distributed manner.")
        else:
            self.DDP_flag = False
            print("Training with single GPU.")

        self.criterion = criterion
        self.metric_ftns = metric_ftns
        self.optimizer = optimizer

        self.epochs = self.args.epochs
        self.save_period = self.args.save_period

        self.mnt_mode = args.monitor_mode
        self.mnt_metric = 'val_' + args.monitor_metric
        self.mnt_metric_test = 'test_' + args.monitor_metric
        assert self.mnt_mode in ['min', 'max']

        self.mnt_best = inf if self.mnt_mode == 'min' else -inf
        self.early_stop = getattr(self.args, 'early_stop', inf)

        self.start_epoch = 1
        self.checkpoint_dir = args.save_dir

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir, exist_ok=True)

        if args.resume is not None:
            self._resume_checkpoint(args.resume)

        self.best_recorder = {'val': {self.mnt_metric: self.mnt_best},
                              'test': {self.mnt_metric_test: self.mnt_best}}

    @abstractmethod
    def _train_epoch(self, epoch):
        raise NotImplementedError

    def train(self):
        not_improved_count = 0
        early_stop_flag = False

        start_total_time = time.time()
        for epoch in range(self.start_epoch, self.epochs + 1):
            start_epoch_time = time.time()

            result = self._train_epoch(epoch)
            
            if self.gpu_id==0: # NOTE: whatever the training mode is, only the master GPU will do the following operations
                # save logged informations into log dict
                log = {'epoch': epoch}
                log.update(result)
                self._record_best(log) # judge whether the model performance in val set improved or not, and update the best_recorder

                wandb.log(log) # log the training information to wandb

                # print logged informations to the screen
                for key, value in log.items():
                    print('\t{:15s}: {}'.format(str(key), value))

                # evaluate model performance according to configured metric, save best checkpoint as model_best
                # NOTE: the difference with the self._record_best() is that the self._record_best() will update the best_recorder 
                # according to the performance in val set, 
                # while the self._save_checkpoint() will save the best model according to the performance in test set
                best = False
                if self.mnt_mode != 'off':
                    try:
                        # check whether model performance improved or not, according to specified metric(mnt_metric)
                        improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
                                (self.mnt_mode ==
                                    'max' and log[self.mnt_metric] >= self.mnt_best)
                    except KeyError:
                        print("Warning: Metric '{}' is not found. " "Model performance monitoring is disabled.".format(
                            self.mnt_metric))
                        self.mnt_mode = 'off'
                        improved = False

                    if improved:
                        self.mnt_best = log[self.mnt_metric]
                        not_improved_count = 0
                        best = True
                    else:
                        not_improved_count += 1

                    if not_improved_count > self.early_stop:
                        print("Validation performance didn\'t improve for {} epochs. " "Training stops.".format(
                            self.early_stop))
                        early_stop_flag = True
                        early_stop_flag = torch.tensor([early_stop_flag]).to(self.gpu_id)
                        if args.n_gpu > 1:
                            dist.broadcast(early_stop_flag, src=self.gpu_id) # synchronize the early_stop_flag only when using multiple GPUs

                if epoch % self.save_period == 0:
                    self._save_checkpoint(epoch, save_best=best, DDP=self.DDP_flag)
            epoch_time = time.time() - start_epoch_time
            # if one GPU detects the early_stop_flag, then all the GPUs will stop training
            if early_stop_flag:
                break
        
        if self.gpu_id==0:
            self._print_best()
            self._print_best_to_file()
        
        total_time = time.time() - start_train_time   

    def _print_best_to_file(self):
        crt_time = time.asctime(time.localtime(time.time()))
        self.best_recorder['val']['time'] = crt_time
        self.best_recorder['test']['time'] = crt_time
        self.best_recorder['val']['seed'] = self.args.seed
        self.best_recorder['test']['seed'] = self.args.seed
        self.best_recorder['val']['best_model_from'] = 'val'
        self.best_recorder['test']['best_model_from'] = 'test'

        if not os.path.exists(self.args.record_dir):
            os.makedirs(self.args.record_dir)
        record_path = os.path.join(
            self.args.record_dir, f"{self.args.exp_name}.csv")
        if not os.path.exists(record_path):
            record_table = pd.DataFrame()
        else:
            record_table = pd.read_csv(record_path)
        val_df = pd.DataFrame([self.best_recorder['val']])
        test_df = pd.DataFrame([self.best_recorder['test']])

        record_table = pd.concat([record_table, val_df], ignore_index=True)
        record_table = pd.concat([record_table, test_df], ignore_index=True)
        record_table.to_csv(record_path, index=False)

    def _prepare_device(self, n_gpu_use):
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            print(
                "Warning: There\'s no GPU available on this machine," "training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            print(
                "Warning: The number of GPU\'s configured to use is {}, but only {} are available " "on this machine.".format(
                    n_gpu_use, n_gpu))
            n_gpu_use = n_gpu
        device = torch.device('cuda' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids

    def _save_checkpoint(self, epoch, save_best=False, DDP=True):
        
        if DDP:
            state = {
            'epoch': epoch,
            'state_dict': self.model.module.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best
        }
        else:
            state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best
        }
        filename = os.path.join(self.checkpoint_dir, 'current_checkpoint.pth')
        torch.save(state, filename)
        print("Saving checkpoint: {} ...".format(filename))
        if save_best:
            best_path = os.path.join(self.checkpoint_dir, 'model_best.pth')
            torch.save(state, best_path)
            print("Saving current best: model_best.pth ...")

    def _resume_checkpoint(self, resume_path):
        resume_path = str(resume_path)
        print("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

        print("Checkpoint loaded. Resume training from epoch {}".format(
            self.start_epoch))

    def _record_best(self, log):
        improved_val = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.best_recorder['val'][
            self.mnt_metric]) or \
            (self.mnt_mode == 'max' and log[self.mnt_metric]
             >= self.best_recorder['val'][self.mnt_metric]) # judge whether the model performance in val set improved or not
        if improved_val:
            self.best_recorder['val'].update(log)

        improved_test = (self.mnt_mode == 'min' and log[self.mnt_metric_test] <= self.best_recorder['test'][
            self.mnt_metric_test]) or \
            (self.mnt_mode == 'max' and log[self.mnt_metric_test] >= self.best_recorder['test'][
                self.mnt_metric_test]) # judge whether the model performance in test set improved or not
        if improved_test:
            self.best_recorder['test'].update(log)

    def _print_best(self):
        print('Best results (w.r.t {}) in validation set:'.format(
            self.args.monitor_metric))
        for key, value in self.best_recorder['val'].items():
            print('\t{:15s}: {}'.format(str(key), value))

        print('Best results (w.r.t {}) in test set:'.format(
            self.args.monitor_metric))
        for key, value in self.best_recorder['test'].items():
            print('\t{:15s}: {}'.format(str(key), value))


class Trainer(BaseTrainer):
    def __init__(self, model, criterion, metric_ftns, gpu_id, optimizer, args, lr_scheduler, train_dataloader, val_dataloader,
                 test_dataloader):
        super(Trainer, self).__init__(
            model, criterion, metric_ftns, gpu_id, optimizer, args)
        self.lr_scheduler = lr_scheduler
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader

    def _train_epoch(self, epoch):
        # 获取可见的GPU数量
        num_gpus = torch.cuda.device_count()

        if self.gpu_id == 0:
            # 打印每个GPU设备的名称
            for i in range(num_gpus):
                gpu_name = torch.cuda.get_device_name(i)
                print(f"GPU {i}: {gpu_name}")
        train_loss = 0
        start_train_time = time.time()
        self.model.train()
        if self.DDP_flag:
            self.train_dataloader.sampler.set_epoch(epoch)
        for batch_idx, (images_id, images, reports_ids, reports_masks) in enumerate(self.train_dataloader):
            images, reports_ids, reports_masks = images.to(self.gpu_id), reports_ids.to(self.gpu_id), reports_masks.to(
                self.gpu_id)
            output = self.model(images, reports_ids, mode='train')
            loss = self.criterion(output, reports_ids, reports_masks) # NOTE: reports_ids is the ground truth, reports_masks is the padding mask
            train_loss += loss.item()
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), 0.1)
            self.optimizer.step()
        train_time = time.time() - start_train_time
        log = {'train_loss': train_loss / len(self.train_dataloader)}

        # NOTE: all the GPUs will do the validation and test, 
        # since if not, other gpu will wait for a long time for the master gpu 
        # to finish the validation and test
        start_val_time = time.time()
        self.model.eval()
        with torch.no_grad():
            val_gts, val_res = [], []
            for batch_idx, (images_id, images, reports_ids, reports_masks) in enumerate(self.val_dataloader):
                images, reports_ids, reports_masks = images.to(self.gpu_id), reports_ids.to(
                    self.gpu_id), reports_masks.to(self.gpu_id)
                output = self.model(images, mode='sample')

                if self.args.n_gpu > 1:
                    reports = self.model.module.tokenizer.decode_batch(
                        output.cpu().numpy())
                    ground_truths = self.model.module.tokenizer.decode_batch(
                        reports_ids[:, 1:].cpu().numpy())
                else:
                    reports = self.model.tokenizer.decode_batch(
                        output.cpu().numpy())
                    ground_truths = self.model.tokenizer.decode_batch(
                        reports_ids[:, 1:].cpu().numpy())

                # wandb table log
                if batch_idx == 0 and self.gpu_id == 0:
                    log_image_report_table(images, reports, ground_truths, "val_report_table")
                
                val_res.extend(reports)
                val_gts.extend(ground_truths)
                
            val_met = self.metric_ftns({i: [gt] for i, gt in enumerate(val_gts)}, # compute metrics in val set
                                    {i: [re] for i, re in enumerate(val_res)})
            log.update(**{'val_' + k: v for k, v in val_met.items()}) # update val log
        val_time = time.time() - start_val_time

        start_test_time = time.time()
        self.model.eval()
        with torch.no_grad():
            test_gts, test_res = [], []
            for batch_idx, (images_id, images, reports_ids, reports_masks) in enumerate(self.test_dataloader):
                images, reports_ids, reports_masks = images.to(self.gpu_id), reports_ids.to(
                    self.gpu_id), reports_masks.to(self.gpu_id)
                output = self.model(images, mode='sample')
                if self.args.n_gpu > 1:
                    reports = self.model.module.tokenizer.decode_batch(
                        output.cpu().numpy())
                    ground_truths = self.model.module.tokenizer.decode_batch(
                        reports_ids[:, 1:].cpu().numpy())
                else:
                    reports = self.model.tokenizer.decode_batch(
                        output.cpu().numpy())
                    ground_truths = self.model.tokenizer.decode_batch(
                        reports_ids[:, 1:].cpu().numpy())
                # wandb table log
                if batch_idx == 0 and self.gpu_id == 0:
                    log_image_report_table(images, reports, ground_truths, "test_report_table")

                test_res.extend(reports)
                test_gts.extend(ground_truths)
            test_met = self.metric_ftns({i: [gt] for i, gt in enumerate(test_gts)}, # compute metrics in test set
                                        {i: [re] for i, re in enumerate(test_res)})
            log.update(**{'test_' + k: v for k, v in test_met.items()}) # update test log
        
        test_time = time.time() - start_test_time

        self.lr_scheduler.step()

        return log
