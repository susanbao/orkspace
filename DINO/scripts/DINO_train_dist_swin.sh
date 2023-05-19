coco_path=$1
python -m torch.distributed.launch --nproc_per_node=4 main.py \
	--output_dir logs/DINO/Swin -c config/DINO/DINO_4scale_swin.py --coco_path $coco_path \
	--options dn_scalar=100 embed_init_tgt=TRUE \
	dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False \
	dn_box_noise_scale=1.0 backbone_dir=$backbone_dir
