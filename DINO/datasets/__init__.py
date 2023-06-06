# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch.utils.data
import torchvision

from .coco import build as build_coco
from .mc_sample import MCSampler, SeqPartialSampler


def get_coco_api_from_dataset(dataset):
    for _ in range(10):
        # if isinstance(dataset, torchvision.datasets.CocoDetection):
        #     break
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    if isinstance(dataset, torchvision.datasets.CocoDetection):
        return dataset.coco


def build_dataset(image_set, args):
    if args.dataset_file == 'coco':
        return build_coco(image_set, args)
    if args.dataset_file == 'coco_panoptic':
        # to avoid making panopticapi required for coco
        from .coco_panoptic import build as build_coco_panoptic
        return build_coco_panoptic(image_set, args)
    if args.dataset_file == 'o365':
        from .o365 import build_o365_combine
        return build_o365_combine(image_set, args)
    if args.dataset_file == 'vanke':
        from .vanke import build_vanke
        return build_vanke(image_set, args)
    raise ValueError(f'dataset {args.dataset_file} not supported')

def test_data_sample(dataset_val, args, idxs = None):
    if args.active_test_type == 'None':
        return torch.utils.data.SequentialSampler(dataset_val)
    if args.active_test_type == 'MC': # MC sample
        return MCSampler(dataset_val, args.test_sample_size)
    if args.active_test_type[:3] == 'ASE':
        return SeqPartialSampler(dataset_val, idxs)
    raise ValueError(f'active_test_type {args.active_test_type} not supported')