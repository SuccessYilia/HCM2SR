

import torch
import torch.optim as optim
from tqdm import tqdm
from time import time

from multirec.base.base_evaluator import BaseEvaluator
from multirec.utils.utils import early_stopping


class BaseTrainer:
    def __init__(self, args, model):
        self.args = args
        self.model = model

        self.epochs = args.epochs
        self.lr = args.lr
        self.optimizer_type = args.optimizer_type
        self.device = args.device
        self.log_interval = args.log_interval
        self.eval_interval = args.eval_interval
        self.save_interval = args.save_interval
        self.stopping_step = args.stopping_step
        self.valid_from_scratch = args.valid_from_scratch
        try:
            self.saved_file = args.saved_file
        except AttributeError:
            pass

        self.logger = args.logger
        self.optimizer = self._build_optimizer()
        self.evaluator = BaseEvaluator()

        self.cur_step = 0
        self.best_valid_score = -1
        self.best_valid_result = None

    def _build_optimizer(self):
        if self.optimizer_type.lower() == 'adam':
            optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        elif self.optimizer_type.lower() == 'sgd':
            optimizer = optim.SGD(self.model.parameters(), lr=self.lr)
        elif self.optimizer_type.lower() == 'adagrad':
            optimizer = optim.Adagrad(self.model.parameters(), lr=self.lr)
        else:
            self.logger('Received unrecognized optimizer, set default Adam optimizer')
            optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        return optimizer

    def _generate_log_output(self, epoch, step, batch_idx, dataloader, losses):
        output = '\rTrain Epoch: %d, Step: %d, [%d / %d (%.2f%%)]\t' % (
            epoch, step, (batch_idx + 1) * self.args.batch_size, len(dataloader.dataset),
            100. * (batch_idx + 1) / len(dataloader))

        if isinstance(losses, tuple):
            des = 'loss%d: %.6f'
            output += ', '.join(des % (idx + 1, loss / self.log_interval) for idx, loss in enumerate(losses))
        else:
            output += 'loss: %.6f' % (losses / self.log_interval)
        return output

    def _generate_eval_output(self, epoch_idx, step, results, cur_step):
        output = '\rValid Epoch: %d, Step: %d, ' % (epoch_idx, step)
        for metric_name, metric_value in results.items():
            output += 'valid ' + metric_name + ': %.4f' % metric_value + '\t'
        output += '\tCur valid step: %d' % cur_step
        return output

    def generate_final_output(self, results):
        output = 'Final results: '
        for metric_name, metric_value in results.items():
            output += metric_name + ': %.4f' % metric_value + '\t'
        return output

    def _save_checkpoint(self, file):
        state = {
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        torch.save(state, file)

    def load_checkpoint(self, checkpoint_file, strict=True, load_optimizer=False):
        self.logger('Load pretrained checkpoint from %s.' % checkpoint_file)
        checkpoint = torch.load(checkpoint_file)
        self.logger('\tLoad state_dict with strict: %s' % str(strict))
        self.model.load_state_dict(checkpoint['state_dict'], strict=strict)
        if load_optimizer:
            self.logger('\tLoad optimizer')
            self.optimizer.load_state_dict(checkpoint['optimizer'])

    def train(self, train_dataloader, eval_dataloader=None):
        step = 1
        total_loss = None
        stop_flag = False
        eval_interval = len(train_dataloader) // self.eval_interval
        valid_step_threshold = 0 if self.valid_from_scratch else len(train_dataloader) // 2
        for epoch_idx in range(self.epochs):

            if stop_flag:
                break

            for batch_idx, raw_instances in enumerate(tqdm(train_dataloader)):

                # Training
                self.model.train()
                instances = []
                for feature in raw_instances:
                    instances.append(feature.to(self.device))
                self.optimizer.zero_grad()
                losses = self.model.calculate_loss(*instances)
                if isinstance(losses, tuple):
                    loss = sum(losses)
                    loss_tuple = tuple(per_loss.item() for per_loss in losses)
                    total_loss = loss_tuple if total_loss is None else tuple(map(sum, zip(total_loss, loss_tuple)))
                else:
                    loss = losses
                    total_loss = losses.item() if total_loss is None else total_loss + losses.item()
                loss.backward()
                self.optimizer.step()

                # Logging
                if step > 0 and step % self.log_interval == 0:
                    # train_result = self.evaluate(train_dataloader, load_best_model=False)
                    log_output = self._generate_log_output(epoch_idx, step, batch_idx, train_dataloader, total_loss)
                    self.logger(log_output)
                    total_loss = None

                # Valid
                if eval_dataloader is not None and step >= valid_step_threshold and step % eval_interval == 0:
                    valid_result = self.evaluate(eval_dataloader, load_best_model=False)
                    valid_score = valid_result['auc']
                    eval_output = self._generate_eval_output(epoch_idx, step, valid_result, self.cur_step)
                    self.logger(eval_output)
                    self.best_valid_score, self.cur_step, stop_flag, update_flag = early_stopping(
                        valid_score, self.best_valid_score, self.cur_step,
                        max_step=self.stopping_step)
                    if update_flag:
                        self.logger('Save checkpoint from Step %d.' % step)
                        self._save_checkpoint(self.saved_file)
                        self.best_valid_result = valid_result
                    if stop_flag:
                        break
                if eval_dataloader is None and step % self.save_interval == 0:
                    self.logger('Save checkpoint from Step %d.' % step)
                    self._save_checkpoint(self.saved_file)

                step += 1

        if eval_dataloader is None:
            self.logger('Save checkpoint after training.')
            self._save_checkpoint(self.saved_file)
        self.logger('Training Done.\n')

        return self.best_valid_score, self.best_valid_result

    @torch.no_grad()
    def evaluate(self, dataloader, load_best_model=False, checkpoint_file=None):
        if load_best_model:
            if checkpoint_file is None:
                checkpoint_file = self.saved_file
            checkpoint = torch.load(checkpoint_file, map_location=self.device)
            self.model.load_state_dict(checkpoint['state_dict'])
            message_output = 'Loading model structure and parameters from {}'.format(checkpoint_file)
            self.logger(message_output)

        self.model.eval()
        for batch_idx, raw_instances in enumerate(tqdm(dataloader)):
            instances = []
            for feature in raw_instances:
                instances.append(feature.to(self.device))
            label = instances[-1]
            score = self.model.evaluate(*instances)
            self.evaluator.accumulate(score, label)

        results = self.evaluator.calculate_metrics()
        return results


class BaseMultiTrainer(BaseTrainer):

    def __init__(self, args, model):
        super(BaseMultiTrainer, self).__init__(args, model)

        self.saved_file_list = args.saved_file_list

        self.cur_step_list = []
        self.best_valid_score_list = []
        self.best_valid_result_list = []

    def _generate_eval_output(self, epoch_idx, step, results, cur_step, dataloader_idx=None):
        output = '\rValid Epoch: %d, Step: %d, Dataloader: %d, ' % (epoch_idx, step, dataloader_idx)
        for metric_name, metric_value in results.items():
            output += 'valid ' + metric_name + ': %.4f' % metric_value + '\t'
        output += '\tCur valid step: %d' % cur_step
        return output

    def train(self, train_dataloader, eval_dataloader_list=None):
        step = 1
        total_loss = None
        eval_interval = len(train_dataloader) // self.eval_interval
        valid_step_threshold = 0 if self.valid_from_scratch else len(train_dataloader) // 2
        n_eval_dataloader = len(eval_dataloader_list) if eval_dataloader_list is not None else 0
        self.cur_step_list = [0] * len(eval_dataloader_list) if eval_dataloader_list else []
        self.best_valid_score_list = [-1] * len(eval_dataloader_list) if eval_dataloader_list else []
        self.best_valid_result_list = [None] * len(eval_dataloader_list) if eval_dataloader_list else []
        stop_flag_list = [False] * len(eval_dataloader_list) if eval_dataloader_list else []
        update_flag_list = [False] * len(eval_dataloader_list) if eval_dataloader_list else []
        valid_result_list = [None] * len(eval_dataloader_list) if eval_dataloader_list else []
        for epoch_idx in range(self.epochs):

            stop_flag = True
            for dataloader_idx in range(n_eval_dataloader):
                stop_flag &= stop_flag_list[dataloader_idx]
            if stop_flag and n_eval_dataloader > 0:
                break

            for batch_idx, raw_instances in enumerate(tqdm(train_dataloader)):

                # Training
                self.model.train()
                instances = []
                for feature in raw_instances:
                    instances.append(feature.to(self.device))
                self.optimizer.zero_grad()
                losses = self.model.calculate_loss(*instances)
                if isinstance(losses, tuple):
                    loss = sum(losses)
                    loss_tuple = tuple(per_loss.item() for per_loss in losses)
                    total_loss = loss_tuple if total_loss is None else tuple(map(sum, zip(total_loss, loss_tuple)))
                else:
                    loss = losses
                    total_loss = losses.item() if total_loss is None else total_loss + losses.item()
                loss.backward()
                self.optimizer.step()

                # Logging
                if step > 0 and step % self.log_interval == 0:
                    log_output = self._generate_log_output(epoch_idx, step, batch_idx, train_dataloader, total_loss)
                    self.logger(log_output)
                    total_loss = None

                # Valid
                if eval_dataloader_list is not None and step > valid_step_threshold and step % eval_interval == 0:
                    for dataloader_idx, eval_dataloader in enumerate(eval_dataloader_list):
                        if stop_flag_list[dataloader_idx]:
                            continue
                        valid_result = self.evaluate(eval_dataloader, load_best_model=False)
                        valid_score = valid_result['auc']
                        valid_result_list[dataloader_idx] = valid_result
                        eval_output = self._generate_eval_output(epoch_idx, step, valid_result, self.cur_step_list[dataloader_idx], dataloader_idx)
                        self.logger(eval_output)
                        self.best_valid_score_list[dataloader_idx], self.cur_step_list[dataloader_idx],\
                            stop_flag_list[dataloader_idx], update_flag_list[dataloader_idx] = early_stopping(
                            valid_score, self.best_valid_score_list[dataloader_idx], self.cur_step_list[dataloader_idx],
                            max_step=self.stopping_step)
                    for dataloader_idx in range(len(eval_dataloader_list)):
                        if update_flag_list[dataloader_idx]:
                            self.logger('Save checkpoint from Step %d for dataloader %d.' % (step, dataloader_idx))
                            self._save_checkpoint(self.saved_file_list[dataloader_idx])
                            self.best_valid_result_list[dataloader_idx] = valid_result_list[dataloader_idx]
                    stop_flag = True
                    for dataloader_idx in range(n_eval_dataloader):
                        stop_flag &= stop_flag_list[dataloader_idx]
                    if stop_flag and n_eval_dataloader > 0:
                        break

                step += 1
        if eval_dataloader_list is None:
            self.logger('Save checkpoint after training.')
            self._save_checkpoint(self.saved_file_list[0])

        self.logger('Training Done.\n')

        return self.best_valid_score_list, self.best_valid_result_list
