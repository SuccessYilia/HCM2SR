import os
import json

from torch.utils.data import Dataset
from multirec.utils.dataset_info import feature_region


def load_feature_mapping(file):
    with open(file, 'r') as fp:
        feature_mapping = json.load(fp)
    feature_names = []
    for feature_name in feature_mapping.keys():
        if feature_mapping[feature_name]['Type'] != 'Raw':
            feature_names.append(feature_name)
    feature_names.sort()
    return feature_mapping, feature_names


class BaseDataset(Dataset):
    def __init__(self, args, state):
        self.task = args.task
        self.data_dir = args.data_dir
        self.dataset = args.dataset
        self.state = state
        self.feature_region = feature_region

        self.feature_mapping_file = os.path.join(self.data_dir, 'common', 'feature_mapping.json')
        self.feature_mapping, self.feature_names = load_feature_mapping(self.feature_mapping_file)

        self.data_size = 0

    @staticmethod
    def load_dataset_contents(file):
        with open(file, 'r') as fp:
            next(fp)
            contents = fp.readlines()
        return contents

    def read_data_from_line(self, raw_data):
        data = raw_data.strip().split(',')
        discrete_features = data[self.feature_region[0]:self.feature_region[1] + 1]
        discrete_features = list(map(int, discrete_features))
        label = float(data[-2]) if self.task.lower() == 'ctr' else float(data[-1])
        return discrete_features, label

    def __len__(self):
        return self.data_size

    def __str__(self):
        output_string = '-------- dataset info --------\n'
        output_string += 'Dataset: ' + self.dataset + '\n'
        output_string += 'State: ' + self.state + '\n'
        output_string += 'Data size: %d' % self.data_size + '\n'
        return output_string
