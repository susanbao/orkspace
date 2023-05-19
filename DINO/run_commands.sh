#!/bin/bash

test_type=MC
json_path=./results/MC.json

for size in 50 100 150 200 250 500 750 1000 1500 2000 3000
do
    for seed in 4856 9140 4151 1294 781 8573 1543 8163 6068 9034
    do
        python main_mc.py --output_dir logs/DINO/R50_COCO_temp -c config/DINO/DINO_5scale.py --coco_path ../coco/ --eval --resume ./ckpts/checkpoint0031_5scale.pth --active_test_type $test_type --test_sample_size $size --result_json_path $json_path --seed $seed --options dn_scalar=100 embed_init_tgt=TRUE dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False dn_box_noise_scale=1.0 >results/MC/R50_COCO_31_$size_$seed.logs 2>results/MC/R50_COCO_31_$size_$seed.err&
    done
done