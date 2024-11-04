import torch
import argparse

from train import init_args, check_already_paras
from multirec.utils.utils import get_saved_file, get_result_file
from multirec.utils.utils import import_model, import_trainer, import_train_dataset, import_eval_dataset
from multirec.utils.dataset_info import group2eval_dataset_list


def train(model_name, params=None):
    # init parser
    args = init_args(model_name, params)
    eval_dataset_list = args.eval_dataset_list.strip().split(',') if args.eval_dataset_list is not None\
        else group2eval_dataset_list[args.group].strip().split(',')
    result_file_list, saved_file_list = [], []
    for dataset_name in eval_dataset_list:
        result_file = get_result_file(args.result_dir, args.model, dataset_name)
        saved_file = get_saved_file(args.saved_dir, args.model, dataset_name, args.params_info)
        result_file_list.append(result_file)
        saved_file_list.append(saved_file)
    args.saved_file_list = saved_file_list

    if not args.ignore_check or model_name == 'OurDev':
        for result_file in result_file_list:
            flag = check_already_paras(args.params_info, result_file)
            if flag is not None:
                return flag

    # init dataloader
    train_dataset = import_train_dataset(model_name)(args)
    args.logger(train_dataset)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                                   pin_memory=True, sampler=None, num_workers=args.num_workers)
    test_dataloader_list = []
    for idx, dataset_name in enumerate(eval_dataset_list):
        test_dataset = import_eval_dataset(model_name)(args, dataset_name, idx, 'test')
        args.logger(test_dataset)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.eval_batch_size,
                                                      shuffle=False, sampler=None, num_workers=args.num_workers)
        test_dataloader_list.append(test_dataloader)

    valid_dataloader_list = None
    if args.valid:
        valid_dataloader_list = []
        for idx, dataset_name in enumerate(eval_dataset_list):
            valid_dataset = import_eval_dataset(model_name)(args, dataset_name, idx, 'valid')
            args.logger(valid_dataset)
            valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.eval_batch_size,
                                                           shuffle=False, sampler=None, num_workers=args.num_workers)
            valid_dataloader_list.append(valid_dataloader)

    # init model
    model = import_model(model_name)(args, train_dataset).to(args.device)
    args.logger(model)

    # init trainer
    trainer = import_trainer(model_name)(args, model)
    if args.pretrained_model is not None:
        trainer.load_checkpoint(args.pretrained_model, strict=bool(args.strict), load_optimizer=bool(args.load_optimizer))

    # train
    if args.valid:
        valid_score_list, valid_result_list = trainer.train(train_dataloader, valid_dataloader_list)
    else:
        valid_score_list, valid_result_list = trainer.train(train_dataloader)

    # evaluate
    test_result_list = []
    for dataloader_idx, test_dataloader in enumerate(test_dataloader_list):
        test_result = trainer.evaluate(test_dataloader, load_best_model=True, checkpoint_file=args.saved_file_list[dataloader_idx])
        test_result_list.append(test_result)
        result_output = trainer.generate_final_output(test_result)
        args.logger(result_output)
        with open(result_file_list[dataloader_idx], 'a') as fp:
            fp.write(args.params_info + '\n')
            fp.write(result_output + '\n\n')

    return {
        'best_valid_score': valid_score_list[0],
        'best_valid_result': valid_result_list[0],
        'test_result': test_result_list[0]
    }


if __name__ == '__main__':
    out_parser = argparse.ArgumentParser()
    out_parser.add_argument('--model', type=str, default='BaseDNN')
    out_args, _ = out_parser.parse_known_args()
    train(out_args.model)
