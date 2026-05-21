import os
from yacs.config import CfgNode
import argparse


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
IMAGE_EXTS = {'.jpg', '.jpeg', '.png'}
PATH_KEYS = [
    'raw_dataset_path',
    'train_dataset_path',
    'test_dataset_path',
    'groundtruth_path',
    'output_path',
    'figure_path',
    'log_path',
    'model_path',
    'config_path',
]


def resolve_path(path):
    if path and not os.path.isabs(path):
        return os.path.join(BASE_DIR, path)
    return path


def is_existing_path(path):
    return os.path.isfile(path) or os.path.isdir(path)


def dataset_output_name(path):
    normalized = os.path.normpath(path)
    stem, ext = os.path.splitext(os.path.basename(normalized))
    if ext.lower() in IMAGE_EXTS:
        return stem
    return os.path.basename(normalized)


def resolve_dataset_input(dataset_name, dataset_base):
    if not dataset_name:
        dataset_path = resolve_path(dataset_base)
        return dataset_path, dataset_output_name(dataset_path)

    dataset_base = resolve_path(dataset_base)
    candidate_cwd = os.path.abspath(os.path.expanduser(dataset_name))
    candidate_ulsd = resolve_path(os.path.expanduser(dataset_name))
    candidate_dataset = os.path.join(dataset_base, dataset_name)

    for candidate in (candidate_cwd, candidate_ulsd):
        if is_existing_path(candidate):
            return candidate, dataset_output_name(candidate)

    if is_existing_path(candidate_dataset):
        return candidate_dataset, dataset_name

    return candidate_dataset, dataset_name


def parse():
    parser = argparse.ArgumentParser()

    parser.add_argument('-g', '--gpu', type=int, help='gpu id')
    parser.add_argument('-o', '--order', type=int, choices=[1, 2, 3, 4, 5, 6], help='order of Bezier curve')
    parser.add_argument('-v', '--version', type=str, help='version')
    parser.add_argument('-d', '--dataset_name', type=str, help='dataset name')
    parser.add_argument('-t', '--type', type=str, help='type')
    parser.add_argument('-l', '--last_epoch', type=int, help='last epoch')
    parser.add_argument('-s', '--save_image', action='store_true', help='save image')
    parser.add_argument('-e', '--evaluate', action='store_true', help='evaluate')
    parser.add_argument('-c', '--save_checkpoint', action='store_true', help='save temporary files')
    parser.add_argument('-m', '--model_name', type=str, help='model name')
    parser.add_argument('--config_path', type=str, default='config', help='config path')
    parser.add_argument('--config_file', type=str, default='default.yaml', help='config filename')

    opts = parser.parse_args()
    opts_dict = vars(opts)
    opts_list = []
    for key, value in zip(opts_dict.keys(), opts_dict.values()):
        if value is not None:
            opts_list.append(key)
            opts_list.append(value)

    #yaml_file = os.path.join(opts.config_path, opts.config_file)

    if not os.path.isabs(opts.config_path):
        opts.config_path = os.path.join(BASE_DIR, opts.config_path)

    yaml_file = os.path.join(opts.config_path, opts.config_file)

    cfg = CfgNode.load_cfg(open(yaml_file))
    cfg.merge_from_list(opts_list)

    for key, value in zip(cfg.dataset_dict.keys(), cfg.dataset_dict.values()):
        if cfg.dataset_name in value:
            cfg.type = key
            break
    if cfg.model_name == '':
        cfg.version = f'{cfg.type}-{cfg.version}-{cfg.order}'
        cfg.model_name = f'{cfg.version}.pkl'
    else:
        cfg.version = '.'.join(cfg.model_name.split('.')[:-1])
    test_dataset_path, output_name = resolve_dataset_input(cfg.dataset_name, cfg.test_dataset_path)

    cfg.log_path = f'{cfg.log_path}/{cfg.version}'
    cfg.raw_dataset_path = os.path.join(cfg.raw_dataset_path, output_name + '_raw')
    cfg.train_dataset_path = os.path.join(cfg.train_dataset_path, output_name + f'_{cfg.order}')
    cfg.test_dataset_path = test_dataset_path
    cfg.groundtruth_path = os.path.join(cfg.groundtruth_path, output_name)
    cfg.output_path = os.path.join(cfg.output_path, output_name + f'_{cfg.version}')
    cfg.figure_path = os.path.join(cfg.figure_path, output_name)

    cfg.image_size = tuple(cfg.image_size)
    cfg.heatmap_size = tuple(cfg.heatmap_size)

    for key in PATH_KEYS:
        cfg[key] = resolve_path(cfg[key])
    cfg.freeze()

    # Print cfg
    for k, v in cfg.items():
        print(f'{k}: {v}')

    return cfg

