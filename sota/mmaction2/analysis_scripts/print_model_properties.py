"""
This script has to be run through docker in RTX3 system. It needs mmaction2
to work
"""
from millify import millify
import argparse
import os
import os.path as osp

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
from mmcv.runner.fp16_utils import wrap_fp16_model

from mmaction.apis import multi_gpu_test, single_gpu_test
from mmaction.datasets import build_dataloader, build_dataset
from mmaction.models import build_model


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMAction2 test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--out',
        default=None,
        help='output result file in pkl/yaml/json format')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g.,'
        ' "top_k_accuracy", "mean_class_accuracy" for video dataset')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
    parser.add_argument('--options', nargs='+', help='custom options')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        default={},
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. For example, '
        "'--cfg-options model.backbone.depth=18 model.backbone.with_cp=True'")
    parser.add_argument(
        '--average-clips',
        choices=['score', 'prob', None],
        default=None,
        help='average type when averaging test clips')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def merge_configs(cfg1, cfg2):
    # Merge cfg2 into cfg1
    # Overwrite cfg1 if repeated, ignore if value is None.
    cfg1 = {} if cfg1 is None else cfg1.copy()
    cfg2 = {} if cfg2 is None else cfg2
    for k, v in cfg2.items():
        if v:
            cfg1[k] = v
    return cfg1


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)

    cfg.merge_from_dict(args.cfg_options)

    # Load output_config from cfg
    output_config = cfg.get('output_config', {})
    # Overwrite output_config from args.out
    output_config = merge_configs(output_config, dict(out=args.out))

    # Load eval_config from cfg
    eval_config = cfg.get('eval_config', {})
    # Overwrite eval_config from args.eval
    eval_config = merge_configs(eval_config, dict(metrics=args.eval))
    # Add options from args.option
    eval_config = merge_configs(eval_config, args.options)

    assert output_config or eval_config, \
        ('Please specify at least one operation (save or eval the '
         'results) with the argument "--out" or "--eval"')

    # set cudnn benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.data.test.test_mode = True

    if cfg.test_cfg is None:
        cfg.test_cfg = dict(average_clips=args.average_clips)
    else:
        # You can set average_clips during testing, it will override the
        # original settting
        if args.average_clips is not None:
            cfg.test_cfg.average_clips = args.average_clips

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # build the dataloader
    dataset = build_dataset(cfg.data.test, dict(test_mode=True))
    dataloader_setting = dict(
        videos_per_gpu=cfg.data.get('videos_per_gpu', 2),
        workers_per_gpu=cfg.data.get('workers_per_gpu', 0),
        dist=distributed,
        shuffle=False)
    dataloader_setting = dict(dataloader_setting,
                              **cfg.data.get('test_dataloader', {}))
    data_loader = build_dataloader(dataset, **dataloader_setting)

    # build the model and load checkpoint
    model = build_model(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)

    # ---- Learnable parameters ----- #
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total number of learnable parameters: {millify(pytorch_total_params)}")


if __name__ == '__main__':
    main()
