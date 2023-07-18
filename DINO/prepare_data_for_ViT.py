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
        
from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
from scipy.optimize import linear_sum_assignment
def hungarian_matching(out_logits, out_bbox, tgt_logits, tgt_bbox, cost_class = 2.0, cost_bbox = 5.0, cost_giou = 2.0, focal_alpha = 0.25, cost_threshold = 2):
    """ Performs the matching
    Params:
        outputs/targets: This is a dict that contains at least these entries:
             "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits, batch_size = 1
             "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates
             "lables": Tensor of dim [num_queries] with the label of each predicted box
        cost_threshold: threshold for distance between outputs and targets
    Returns:
        cost_matrix
    """
    num_queries = out_logits.shape[0]
 
    # We flatten to compute the cost matrices in a batch
    out_prob = torch.from_numpy(out_logits).sigmoid()  # [batch_size * num_queries, num_classes]
    out_bbox = torch.from_numpy(out_bbox)  # [batch_size * num_queries, 4]
    
    tgt_ids = np.argmax(tgt_logits, axis=1)
    tgt_bbox = torch.from_numpy(tgt_bbox)
    
    # Compute the classification cost.
    alpha = focal_alpha
    gamma = 2.0
    neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
    pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
    cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]
    
    # Compute the L1 cost between boxes
    cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
    
    # Compute the giou cost betwen boxes            
    cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))
    
    # Final cost matrix
    C = cost_bbox * cost_bbox + cost_class * cost_class + cost_giou * cost_giou
    # C = C.view(num_queries, -1)
    C = C.numpy()
    C = C.T
    return C
        
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

# def prepare_feature_data(data_path, annotation_path, feature_path, start_nums, end_nums, stored_path):
#     Y = None
#     count = 271162
#     for img_idx in range(start_nums, end_nums):
#         feature = np_read(feature_path + str(img_idx) + ".npy")
#         new_feature = np.zeros((1, feature.shape[1], feature.shape[2]*feature.shape[0]))
#         steps = feature.shape[2]
#         for i in range(feature.shape[0]):
#             new_feature[:,:,(steps*i):(steps*i+steps)] = feature[i]
#         results = read_one_image_results(data_path + str(img_idx) + ".json")
#         pred_logits = np.array(results['input']['pred_logits'])
#         pred_boxes = np.array(results['input']['pred_boxes'])
#         pred_results = np.concatenate((pred_boxes, pred_logits), axis=2)
#         annotation_data = read_one_image_results(annotation_path + str(img_idx) + ".json")
#         selected_index = annotation_data['selected_index']
#         loss = annotation_data['loss']
#         pred_logits_max = np.max(pred_logits, axis=2).squeeze()
#         sort_indexs = np.argsort(-pred_logits_max)
#         topk_indexs = sort_indexs[:196]
#         pred_results = pred_results[:,topk_indexs]
#         query_feature = new_feature[:,topk_indexs]
#         arrSortedIndex  = np.lexsort((pred_results[:,:,0], pred_results[:,:,1])).squeeze()
#         query_feature = query_feature[:,arrSortedIndex].squeeze(axis=0)
#         np_write(query_feature, stored_path + "feature" +str(img_idx) + ".npy")
#         for i in selected_index:
#             one_json = {"self_feature": new_feature[:,i].tolist(), "feature_idx": img_idx}
#             write_one_results(stored_path + str(count) + ".json", one_json)
#             count += 1
#         if Y is None:
#             Y = loss
#         else:
#             Y = np.concatenate((Y, loss))
#         if img_idx % 100 == 0:
#             print(f"{img_idx} finished")
#     np_write(Y, stored_path + "annotation_1.npy")
#     return

def prepare_feature_data(data_path, annotation_path, feature_path, start_nums, end_nums, stored_path):
    Y = None
    count = 0
    for img_idx in range(start_nums, end_nums):
        results = read_one_image_results(data_path + str(img_idx) + ".json")
        pred_logits = np.array(results['input']['pred_logits']).squeeze()
        pred_boxes = np.array(results['input']['pred_boxes']).squeeze()
        annotation_data = read_one_image_results(annotation_path + str(img_idx) + ".json")
        selected_index = annotation_data['selected_index']
        loss = annotation_data['loss']
        tgt_logits = pred_logits[selected_index]
        tgt_bbox = pred_boxes[selected_index]
        cost_matrix = hungarian_matching(pred_logits, pred_boxes, tgt_logits, tgt_bbox)
        topk_idx = np.argsort(cost_matrix, axis=1)[:, :196]
        self_index = np.expand_dims(np.array(selected_index), axis=1)
        final_index = np.concatenate((self_index, topk_idx), axis=1)
        for i in range(len(selected_index)):
            one_json = {"selected_idxs": final_index[i].tolist(), "feature_idx": img_idx}
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

def prepare_feature_data_annotation(data_path, annotation_path, feature_path, start_nums, end_nums, stored_path, loss_type):
    Y = None
    for img_idx in range(start_nums, end_nums):
        annotation_data = read_one_image_results(annotation_path + str(img_idx) + ".json")
        loss = annotation_data[loss_type]
        if Y is None:
            Y = loss
        else:
            Y = np.concatenate((Y, loss))
        if img_idx % 100 == 0:
            print(f"{img_idx} finished")
    np_write(Y, stored_path + f"annotation_{loss_type}.npy")
    return

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_nums", type = int, default = 0,
                        help="Start No.")
    parser.add_argument("--split", type=str, default="val",
                        help="split type: val or train.")
    parser.add_argument("--end_nums", type = int, default = 1000,
                        help="Start No.")
    parser.add_argument("--loss_type", type=str, default="loss",
                        help="loss type: loss, loss_ce, loss_bbox, loss_giou")
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
    loss_type = args.loss_type
    if split == "val":
        # train_X, train_Y = get_feature_data(test_data_path, test_annotation_path, test_feature_path, start_nums, start_nums + 1000)
        stored_file = base_path + split + "/feature_pre_data/"
        prepare_feature_data_annotation(test_data_path, test_annotation_path, test_feature_path, start_nums, end_nums, stored_file, loss_type)
    elif split == "train":
        # train_X, train_Y = get_feature_data(train_data_path, train_annotation_path, train_feature_path, start_nums, start_nums + 1000)
        stored_file = base_path + split + "/feature_pre_data/"
        prepare_feature_data_annotation(train_data_path, train_annotation_path, train_feature_path, start_nums, end_nums, stored_file, loss_type)
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