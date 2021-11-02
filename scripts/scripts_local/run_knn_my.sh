
PROJ_PATH=.
DATA_PATH=$PROJ_PATH/data/cifar10

# -------------------------------------------------------

OUT_PATH=$PROJ_PATH/exp_output/esvit_exp/swin/swin_tiny/bl_lr0.0005_gpu16_bs32_dense_multicrop_epoch300
CKPT_PATH=$PROJ_PATH/exp_output/esvit_exp/swin/swin_tiny/bl_lr0.0005_gpu16_bs32_dense_multicrop_epoch300/checkpoint_best.pth

# python -m torch.distributed.launch --nproc_per_node=4 eval_linear.py --data_path $DATA_PATH --output_dir $OUT_PATH/lincls/epoch0300 --pretrained_weights $CKPT_PATH --checkpoint_key teacher --batch_size_per_gpu 256 --arch swin_tiny --cfg experiments/imagenet/swin/swin_tiny_patch4_window7_224.yaml --n_last_blocks 4 --num_labels 1000 MODEL.NUM_CLASSES 0

python3 -m torch.distributed.launch --nproc_per_node=1 eval_knn.py --data_path $DATA_PATH --dump_features $OUT_PATH/features/epoch0300 --pretrained_weights $CKPT_PATH --checkpoint_key teacher --batch_size_per_gpu 16 --arch swin_tiny --cfg experiments/imagenet/swin/swin_tiny_patch4_window7_224.yaml MODEL.NUM_CLASSES 0
