import torch
import argparse

from multirec.utils.utils import set_device, set_rng_seed
from multirec.utils.utils import get_logger, get_already_paras, get_saved_file, get_result_file
from multirec.utils.utils import import_parser, import_dataset, import_model, import_trainer


def init_args(model_name, params=None):
    parser = import_parser(model_name)()
    args = parser.parse(params)
    args.model = model_name
    args.device = set_device(args.gpu_id)
    args.params_info = parser.get_params_info()
    args.logger = get_logger(args.logger_dir, args.model, args.params_info)
    set_rng_seed(args.seed)
    args.logger('All Parameters: \n', args, '\n')
    args.logger('Parameters info string:\n %s\n' % args.params_info)
    return args


def check_already_paras(params_info, result_file):
    already_paras = get_already_paras(result_file)
    if params_info.strip() in already_paras:
        print('Already run these parameters')
        return {
            'best_valid_score': 0,
            'best_valid_result': None,
            'test_result': None
        }
    else:
        return None


def train(model_name, params=None):
    # init parser
    args = init_args(model_name, params)
    result_file = get_result_file(args.result_dir, args.model, args.dataset)
    args.saved_file = get_saved_file(args.saved_dir, args.model, args.dataset, args.params_info)

    if not args.ignore_check:
        flag = check_already_paras(args.params_info, result_file)
        if flag is not None:
            return flag

    # init dataloader
    train_dataset = import_dataset(model_name)(args, 'train')
    args.logger(train_dataset)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                                   pin_memory=True, sampler=None, num_workers=args.num_workers)
    test_dataset = import_dataset(model_name)(args, 'test')
    args.logger(test_dataset)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.eval_batch_size,
                                                  shuffle=False, sampler=None, num_workers=args.num_workers)
    valid_dataloader = None
    if args.valid:
        valid_dataset = import_dataset(model_name)(args, 'valid')
        args.logger(valid_dataset)
        valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.eval_batch_size,
                                                       shuffle=False, sampler=None, num_workers=args.num_workers)

    # init model
    model = import_model(model_name)(args, train_dataset).to(args.device)
    args.logger(model)

    # init trainer
    trainer = import_trainer(model_name)(args, model)
    if args.pretrained_model is not None:
        trainer.load_checkpoint(args.pretrained_model, strict=bool(args.strict), load_optimizer=bool(args.load_optimizer))

    # train
    if args.valid:
        valid_score, valid_result = trainer.train(train_dataloader, valid_dataloader)
    else:
        valid_score, valid_result = trainer.train(train_dataloader)

    # evaluate
    test_result = trainer.evaluate(test_dataloader, load_best_model=True)
    result_output = trainer.generate_final_output(test_result)
    args.logger(result_output)
    with open(result_file, 'a') as fp:
        fp.write(args.params_info + '\n')
        fp.write(result_output + '\n\n')

    return {
        'best_valid_score': valid_score,
        'best_valid_result': valid_result,
        'test_result': test_result
    }


if __name__ == '__main__':
    out_parser = argparse.ArgumentParser()
    out_parser.add_argument('--model', type=str, default='BaseDNN')
    out_args, _ = out_parser.parse_known_args()
    train(out_args.model)
