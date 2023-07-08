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

def np_read(file):
    with open(file, "rb") as outfile:
        data = np.load(outfile)
    return data
def np_write(data, file):
    with open(file, "wb") as outfile:
        np.save(outfile, data)
        
def get_numpy_data(data_path, annotation_path, start_nums, end_nums):
    X = None
    Y = None
    for img_idx in range(start_nums, end_nums):
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

def get_feature_data(data_path, annotation_path, feature_path, start_nums, end_nums):
    X = None
    Y = None
    for img_idx in range(start_nums, end_nums):
        feature = np_read(feature_path + str(img_idx) + ".npy")
        new_feature = np.zeros((1, feature.shape[1], feature.shape[2]*feature.shape[0]))
        steps = feature.shape[2]
        for i in range(feature.shape[0]):
            new_feature[:,:,(steps*i):(steps*i+steps)] = feature[i]
        results = read_one_image_results(data_path + str(img_idx) + ".json")
        pred_logits = np.array(results['input']['pred_logits'])
        pred_boxes = np.array(results['input']['pred_boxes'])
        pred_results = np.concatenate((pred_boxes, pred_logits), axis=2)
        annotation_data = read_one_image_results(annotation_path + str(img_idx) + ".json")
        selected_index = annotation_data['selected_index']
        loss = annotation_data['loss']
        pred_logits_max = np.max(pred_logits, axis=2).squeeze()
        sort_indexs = np.argsort(-pred_logits_max)
        topk_indexs = sort_indexs[:196]
        pred_results = pred_results[:,topk_indexs]
        query_feature = new_feature[:,topk_indexs]
        arrSortedIndex  = np.lexsort((pred_results[:,:,0], pred_results[:,:,1])).squeeze()
        query_feature = query_feature[:,arrSortedIndex]
        one_X = None
        for i in selected_index:
            temp = np.concatenate((np.expand_dims(new_feature[:,i], axis=1), query_feature), axis=1)
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

def prepare_feature_data(data_path, annotation_path, feature_path, end_nums, stored_path):
    Y = None
    count = 0
    for img_idx in range(end_nums):
        feature = np_read(feature_path + str(img_idx) + ".npy")
        new_feature = np.zeros((1, feature.shape[1], feature.shape[2]*feature.shape[0]))
        steps = feature.shape[2]
        for i in range(feature.shape[0]):
            new_feature[:,:,(steps*i):(steps*i+steps)] = feature[i]
        results = read_one_image_results(data_path + str(img_idx) + ".json")
        pred_logits = np.array(results['input']['pred_logits'])
        pred_boxes = np.array(results['input']['pred_boxes'])
        pred_results = np.concatenate((pred_boxes, pred_logits), axis=2)
        annotation_data = read_one_image_results(annotation_path + str(img_idx) + ".json")
        selected_index = annotation_data['selected_index']
        loss = annotation_data['loss']
        pred_logits_max = np.max(pred_logits, axis=2).squeeze()
        sort_indexs = np.argsort(-pred_logits_max)
        topk_indexs = sort_indexs[:196]
        pred_results = pred_results[:,topk_indexs]
        query_feature = new_feature[:,topk_indexs]
        arrSortedIndex  = np.lexsort((pred_results[:,:,0], pred_results[:,:,1])).squeeze()
        query_feature = query_feature[:,arrSortedIndex].squeeze(axis=0)
        np_write(query_feature, stored_path + "feature" +str(img_idx) + ".npy")
        for i in selected_index:
            one_json = {"self_feature": new_feature[:,i].tolist(), "feature_idx": img_idx}
            write_one_results(stored_path + str(count) + ".json", one_json)
            count += 1
        if Y is None:
            Y = loss
        else:
            Y = np.concatenate((Y, loss))
        if img_idx % 100 == 0:
            print(f"{img_idx} finished")
    np_write(Y, stored_path + "annotation.npy")
    return

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_nums", type = int, default = 0,
                        help="Start No.")
    parser.add_argument("--split", type=str, default="val",
                        help="split type: val or train.")
    parser.add_argument("--end_nums", type = int, default = 1000,
                        help="Start No.")
    args = parser.parse_args()
    train_data_path = "./data/5_scale_31/train/data/"
    test_data_path = "./data/5_scale_31/val/data/"
    train_annotation_path = "./data/5_scale_31/train/box_annotation/"
    test_annotation_path = "./data/5_scale_31/val/box_annotation/"
    base_path = "./data/5_scale_31/"
    train_feature_path = base_path + "train/feature_data/"
    test_feature_path = base_path + "val/feature_data/"
    start_nums = args.start_nums
    split = args.split
    end_nums = args.end_nums
    if split == "val":
        # train_X, train_Y = get_feature_data(test_data_path, test_annotation_path, test_feature_path, start_nums, start_nums + 1000)
        stored_file = base_path + split + "/feature_pre_data/"
        prepare_feature_data(test_data_path, test_annotation_path, test_feature_path, end_nums, stored_file)
    elif split == "train":
        # train_X, train_Y = get_feature_data(train_data_path, train_annotation_path, train_feature_path, start_nums, start_nums + 1000)
        stored_file = base_path + split + "/feature_pre_data/"
        prepare_feature_data(train_data_path, train_annotation_path, train_feature_path, end_nums, stored_file)
    else:
        assert False
    # train_X, train_Y = get_feature_data(train_data_path, train_annotation_path, train_feature_path, start_nums, start_nums + 10000)
    # train_X, train_Y = get_feature_data(test_data_path, test_annotation_path, test_feature_path, 0, start_nums + 1000)
    # store_preprocess_inputs_path = base_path + split + f"/pre_data/{split}_box_level_ViT_inputs_{start_nums}.npy"
    # store_preprocess_inputs_path = base_path + split + f"/pre_data/{split}_feature_box_level_ViT_inputs_{start_nums}.npy"
    # with open(store_preprocess_inputs_path, "wb") as outfile:
    #     np.save(outfile, train_X)
    # store_preprocess_annotations_path = base_path + split + f"/pre_data/{split}_box_level_ViT_annotations_{start_nums}.npy"
    # store_preprocess_annotations_path = base_path + split + f"/pre_data/{split}_feature_box_level_ViT_annotations_{start_nums}.npy"
    # with open(store_preprocess_annotations_path, "wb") as outfile:
    #     np.save(outfile, train_Y)

if __name__ == "__main__":
    main()