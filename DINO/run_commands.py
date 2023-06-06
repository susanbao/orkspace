import sys
import os
import copy
import json
import time

# MC ResNet 5 scale
# python main_mc.py --output_dir logs/DINO/R50_COCO_temp -c config/DINO/DINO_5scale.py --coco_path ../coco/ --eval --resume ./ckpts/checkpoint0031_5scale.pth --active_test_type $test_type --test_sample_size $size --result_json_path $json_path --seed $seed --options dn_scalar=100 embed_init_tgt=TRUE dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False dn_box_noise_scale=1.0 >results/MC/R50_COCO_31_$size_$seed.logs 2>results/MC/R50_COCO_31_$size_$seed.err&

# test_type = "MC"
# json_path = "./results/MC_R50_90_runs.json"
# basic_command = "python main_mc.py --output_dir logs/DINO/R50_COCO_temp -c config/DINO/DINO_5scale.py --coco_path ../coco/ --eval --resume ./ckpts/checkpoint0031_5scale.pth"
# basic_command += " --active_test_type " + test_type
# basic_command += " --result_json_path " + json_path
# later_command = " --options dn_scalar=100 embed_init_tgt=TRUE dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False dn_box_noise_scale=1.0 "

# size_set = [50, 100, 150, 200, 250, 500, 750, 1000, 1500, 2000, 3000]
# # random_seed_set = [4519, 9524, 5901, 1028, 6382, 5383, 5095, 7635,  890,  608]
# random_seed_set = [787, 7139, 3082, 5583, 5404, 5950, 8868, 6872, 6672, 3136, 5427,
#        2131, 9432, 6930, 9491, 4187, 8110,  481, 3141, 7746, 3549, 4096,
#        1767, 2126, 4834, 2409, 9729, 4254, 9513, 7544, 4352, 1513, 8175,
#        2811, 2804, 4470, 2389, 4837, 5658,  907, 2484, 2403, 5460, 9269,
#        4132, 4111, 9473, 5592, 7820,  803, 4586, 6444, 6787, 1748,  173,
#        1124, 8080, 5474, 2869, 5744, 5410, 5907, 7729, 6235, 6274, 1314,
#        9561, 1343, 3234,  374, 9414, 6446, 2103, 3120,    0,  733, 3938,
#        4215, 2467, 6064, 5355, 4593, 6251, 5541, 5512, 4428,  990, 1723,
#        8246, 2992]
# # size_set = [50,100]
# # random_seed_set = [4519, 9524]
# for size in size_set:
#     for seed in random_seed_set:
#         file_path = "results/MC/R50_COCO_31_" + str(size) + "_"+str(seed)
#         command = basic_command + " --test_sample_size " + str(size) + " --seed " + str(seed) + later_command + ">" + file_path + ".logs" + " 2>" + file_path + ".err&"
#         print(command)
#         os.system(command)
#         time.sleep(size//4+30)

# MC SWIN 4 scale
# python main_mc.py --output_dir logs/DINO/SWIN_COCO_temp -c config/DINO/DINO_4scale_swin.py --coco_path ../coco/ --eval --resume ./ckpts/checkpoint0029_4scale_swin.pth --active_test_type MC --result_json_path ./results/MC_swin.json --test_sample_size 50 --seed 5 --options dn_scalar=100 embed_init_tgt=TRUE dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False dn_box_noise_scale=1.0 backbone_dir=./

# # MC SWIN 4 scale
# test_type = "MC"
# json_path = "./results/MC_swin_90_runs.json"
# basic_command = "python main_mc.py --output_dir logs/DINO/SWIN_COCO_temp -c config/DINO/DINO_4scale_swin.py --coco_path ../coco/ --eval --resume ./ckpts/checkpoint0029_4scale_swin.pth"
# basic_command += " --active_test_type " + test_type
# basic_command += " --result_json_path " + json_path
# later_command = " --options dn_scalar=100 embed_init_tgt=TRUE dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False dn_box_noise_scale=1.0 backbone_dir=./ "

# size_set = [50, 100, 150, 200, 250, 500, 750, 1000, 1500, 2000, 3000]
# # random_seed_set = [4519, 9524, 5901, 1028, 6382, 5383, 5095, 7635,  890,  608]
# random_seed_set = [787, 7139, 3082, 5583, 5404, 5950, 8868, 6872, 6672, 3136, 5427,
#        2131, 9432, 6930, 9491, 4187, 8110,  481, 3141, 7746, 3549, 4096,
#        1767, 2126, 4834, 2409, 9729, 4254, 9513, 7544, 4352, 1513, 8175,
#        2811, 2804, 4470, 2389, 4837, 5658,  907, 2484, 2403, 5460, 9269,
#        4132, 4111, 9473, 5592, 7820,  803, 4586, 6444, 6787, 1748,  173,
#        1124, 8080, 5474, 2869, 5744, 5410, 5907, 7729, 6235, 6274, 1314,
#        9561, 1343, 3234,  374, 9414, 6446, 2103, 3120,    0,  733, 3938,
#        4215, 2467, 6064, 5355, 4593, 6251, 5541, 5512, 4428,  990, 1723,
#        8246, 2992]
# # size_set = [50,100]
# # random_seed_set = [4519, 9524]
# for size in size_set:
#     for seed in random_seed_set:
#         file_path = "results/MC/SWIN_COCO_29_" + str(size) + "_"+str(seed)
#         command = basic_command + " --test_sample_size " + str(size) + " --seed " + str(seed) + later_command + ">" + file_path + ".logs" + " 2>" + file_path + ".err&"
#         print(command)
#         os.system(command)
#         time.sleep(size//4+30)
        

# ASE ResNet 5 scale, after active_test_demo.ipynb
# python main_mc.py --output_dir logs/DINO/R50_COCO_temp -c config/DINO/DINO_5scale.py --coco_path ../coco/ --eval --resume ./ckpts/checkpoint0031_5scale.pth --active_test_type ASE --test_sample_size 50 --result_json_path ./results/ASE_R50_10_runs.json --seed 1028 --idx_json_path ./results/ASE_samples/samples_50_1028.json --options dn_scalar=100 embed_init_tgt=TRUE dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False dn_box_noise_scale=1.0 

# ASE regression ResNet 5 scale
# python main_mc.py --output_dir logs/DINO/R50_COCO_temp -c config/DINO/DINO_5scale.py --coco_path ../coco/ --eval --resume ./ckpts/checkpoint0031_5scale.pth --active_test_type ASE_regression --test_sample_size 50 --result_json_path ./results/ASE_reg_R50_10_runs.json --seed 1028 --idx_json_path ./results/ASE_regression_samples/samples_50_1028.json --options dn_scalar=100 embed_init_tgt=TRUE dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False dn_box_noise_scale=1.0 

# ASE regression + classification ResNet 5 scale
# python main_mc.py --output_dir logs/DINO/R50_COCO_temp -c config/DINO/DINO_5scale.py --coco_path ../coco/ --eval --resume ./ckpts/checkpoint0031_5scale.pth --active_test_type ASE_all --test_sample_size 50 --result_json_path ./results/ASE_all_R50_10_runs.json --seed 1028 --idx_json_path ./results/ASE_all_samples/samples_50_1028.json --options dn_scalar=100 embed_init_tgt=TRUE dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False dn_box_noise_scale=1.0 

test_type = "ASE_true_loss_rs"
json_path = "./results/ASE_true_loss_R50_10_runs.json"
samples_path = "./results/ASE_true_loss_samples/samples_"
basic_command = "python main_mc.py --output_dir logs/DINO/R50_COCO_temp -c config/DINO/DINO_5scale.py --coco_path ../coco/ --eval --resume ./ckpts/checkpoint0031_5scale.pth"
basic_command += " --active_test_type " + test_type
basic_command += " --result_json_path " + json_path
later_command = " --options dn_scalar=100 embed_init_tgt=TRUE dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False dn_box_noise_scale=1.0 "

# size_set = [50, 100, 150, 200, 250, 500, 750, 1000, 1500, 2000, 3000]
size_set = [50, 100, 150, 200, 250, 500, 750, 1000]
random_seed_set = [4519, 9524, 5901, 1028, 6382, 5383, 5095, 7635,  890,  608]
# random_seed_set = [787, 7139, 3082, 5583, 5404, 5950, 8868, 6872, 6672, 3136, 5427,
#        2131, 9432, 6930, 9491, 4187, 8110,  481, 3141, 7746, 3549, 4096,
#        1767, 2126, 4834, 2409, 9729, 4254, 9513, 7544, 4352, 1513, 8175,
#        2811, 2804, 4470, 2389, 4837, 5658,  907, 2484, 2403, 5460, 9269,
#        4132, 4111, 9473, 5592, 7820,  803, 4586, 6444, 6787, 1748,  173,
#        1124, 8080, 5474, 2869, 5744, 5410, 5907, 7729, 6235, 6274, 1314,
#        9561, 1343, 3234,  374, 9414, 6446, 2103, 3120,    0,  733, 3938,
#        4215, 2467, 6064, 5355, 4593, 6251, 5541, 5512, 4428,  990, 1723,
#        8246, 2992]
# size_set = [50,100]
# random_seed_set = [4519, 9524]
for size in size_set:
    for seed in random_seed_set:
        file_path = "results/ASE/R50_COCO_31_" + str(size) + "_"+str(seed)
        command = basic_command + " --test_sample_size " + str(size) + " --seed " + str(seed) + " --idx_json_path " + samples_path + str(size) + "_" + str(seed) + ".json" + later_command + ">" + file_path + ".logs" + " 2>" + file_path + ".err&"
        print(command)
        os.system(command)
        time.sleep(size//5+30)