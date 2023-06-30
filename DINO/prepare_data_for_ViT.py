import argparse
import datetime
import json
import random
import time
from pathlib import Path
import os, sys
import numpy as np
import torch

from main import build_model_main
from util.slconfig import SLConfig
from datasets import build_dataset
from util.visualizer import COCOVisualizer
from util import box_ops
import pickle
import copy
import random
from util.utils import slprint, to_device
import util.misc as utils
from engine import evaluate
from torch.utils.data import DataLoader
from datasets import build_dataset, get_coco_api_from_dataset

def read_one_image_results(path):
    with open(path, "r") as outfile:
        data = json.load(outfile)
    return data

def write_one_results(path, json_data):
    with open(path, "w") as outfile:
        json.dump(json_data, outfile)
        
def get_numpy_data(data_path, annotation_path, img_nums):
    X = None
    Y = None
    for img_idx in range(img_nums):
        results = read_one_image_results(data_path + str(img_idx) + ".json")
        pred_logits = np.array(results['input']['pred_logits'])
        pred_boxes = np.array(results['input']['pred_boxes'])
        pred_results = np.concatenate((pred_boxes, pred_logits), axis=2)
        annotation_data = read_one_image_results(annotation_path + str(img_idx) + ".json")
        selected_index = annotation_data['selected_index']
        out_results = pred_results[:,selected_index]
        loss = annotation_data['loss']
        pred_logits_max = np.max(pred_logits, axis=2).squeeze()
        sort_indexs = np.argsort(-pred_logits_max)
        topk_indexs = sort_indexs[:196]
        pred_results = pred_results[:,topk_indexs]
        arrSortedIndex  = np.lexsort((pred_results[:,:,0], pred_results[:,:,1])).squeeze()
        pred_results = pred_results[:,arrSortedIndex]
        one_X = None
        for i in range(out_results.shape[1]):
            temp = np.append(out_results[:,i], pred_results)
            temp = temp.reshape((1,pred_results.shape[1]+1, pred_results.shape[2]))
            if one_X is None:
                one_X = temp
            else:
                one_X = np.concatenate((one_X, temp), axis=0)
        if X is None:
            X = one_X
        else:
            X = np.concatenate((X, one_X), axis=0)
        if Y is None:
            Y = loss
        else:
            Y = np.concatenate((Y, loss))
        if img_idx % 100 == 0:
            print(f"{img_idx} finished")
    return X, Y

def main():
    train_data_path = "./data/5_scale_31/train/data/"
    test_data_path = "./data/5_scale_31/val/data/"
    train_annotation_path = "./data/5_scale_31/train/box_annotation/"
    test_annotation_path = "./data/5_scale_31/val/box_annotation/"
    train_X, train_Y = get_numpy_data(train_data_path, train_annotation_path, 50000)
    base_path = "./data/5_scale_31/"
    split = "train"
    store_preprocess_inputs_path = base_path + split + f"/pre_data/{split}_box_level_ViT_inputs.npy"
    with open(store_preprocess_inputs_path, "wb") as outfile:
        np.save(outfile, train_X)
    store_preprocess_annotations_path = base_path + split + f"/pre_data/{split}_box_level_ViT_annotations.npy"
    with open(store_preprocess_annotations_path, "wb") as outfile:
        np.save(outfile, train_Y)

if __name__ == "__main__":
    main()