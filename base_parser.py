# -*- coding: utf-8 -*-
# Time  : 2022/4/8
# Author: slmu
# Email : mushanlei.msl@alibaba-inc.com

import argparse
import collections


class BaseParser:

    def __init__(self):
        self.parser = argparse.ArgumentParser()

        # Parameters for general setting
        self.parser.add_argument('--task', type=str, default='ctr')
        self.parser.add_argument('--seed', type=int, default=2022)
        self.parser.add_argument('--gpu_id', type=int, default=0)
        self.parser.add_argument('--logger_dir', type=str, default='logger/')
        self.parser.add_argument('--result_dir', type=str, default='result/')
        self.parser.add_argument('--saved_dir', type=str, default='saved/')
        self.parser.add_argument('--ignore_check', action='store_true')

        # Parameters for data setting
        self.parser.add_argument('--data_dir', type=str, default='data_processing/AliExpress/generate')
        self.parser.add_argument('--dataset', type=str, default='ali_express_nl')

        # Parameters for training
        self.parser.add_argument('--batch_size', type=int, default=512)
        self.parser.add_argument('--epochs', type=int, default=5)
        self.parser.add_argument('--lr', type=float, default=5e-3)
        self.parser.add_argument('--optimizer_type', type=str, default='adam')
        self.parser.add_argument('--log_interval', type=int, default=500)
        self.parser.add_argument('--save_interval', type=int, default=1000)
        self.parser.add_argument('--eval_interval', type=int, default=40)
        self.parser.add_argument('--stopping_step', type=int, default=10)
        self.parser.add_argument('--num_workers', type=int, default=4)
        self.parser.add_argument('--pretrained_model', type=str, default=None)
        self.parser.add_argument('--strict', type=int, default=1)
        self.parser.add_argument('--load_optimizer', type=int, default=0)

        # Parameters for model

        # Parameters for evaluating
        self.parser.add_argument('--valid', type=int, default=1)
        self.parser.add_argument('--valid_from_scratch', type=int, default=0)
        self.parser.add_argument('--eval_batch_size', type=int, default=2048)

        self.args = None
        self.params_dict = collections.OrderedDict()

    def parse(self, params=None):
        args, _ = self.parser.parse_known_args(params)
        self.args = args
        return args

    def get_prefix_params_dict(self):
        self.params_dict['task'] = self.args.task
        self.params_dict['dataset'] = self.args.dataset
        self.params_dict['lr'] = str(self.args.lr)
        self.params_dict['batch'] = str(self.args.batch_size)
        self.params_dict['optimizer'] = self.args.optimizer_type

    def get_postfix_params_dict(self):
        self.params_dict['seed'] = str(self.args.seed)
        if self.args.pretrained_model:
            self.params_dict['pretrain'] = self.args.pretrained_model.strip().split('/')[1].split('.')[0]

    def get_model_params_dict(self):
        pass

    def get_params_info(self):
        self.get_prefix_params_dict()
        self.get_model_params_dict()
        self.get_postfix_params_dict()

        params_info = []
        for key, value in self.params_dict.items():
            params_info.append(key + '-' + value)
        return '_'.join(params_info)


class BaseMultiParser(BaseParser):

    def __init__(self):
        super(BaseMultiParser, self).__init__()
        self.parser.add_argument('--group', type=str, default=None)
        self.parser.add_argument('--n_scenarios', type=int, default=None)
        self.parser.add_argument('--eval_dataset_list', type=str, default=None)

    def get_prefix_params_dict(self):
        self.params_dict['task'] = self.args.task
        self.params_dict['group'] = self.args.group
        self.params_dict['lr'] = str(self.args.lr)
        self.params_dict['batch'] = str(self.args.batch_size)
        self.params_dict['optimizer'] = self.args.optimizer_type
