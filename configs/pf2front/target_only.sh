N_GPUS=2
BATCH_SIZE=64
# DATA_ROOT=/home/zhaozj/Datasets
DATA_ROOT=/network_space/server127/shared/zhaozijing/datasets
OUTPUT_DIR=./outputs/pf2front/target_only

CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=8 torchrun \
--rdzv_endpoint localhost:28501 \
--nproc_per_node=${N_GPUS} \
main.py \
--mode source_only \
--data_root ${DATA_ROOT} \
--src_dataset 3dfront \
--tgt_dataset 3dfront \
--num_points_preload 100000 \
--num_points 40000 \
--categories bed cabinet chair desk lamp shelf sofa table tv_stand others \
--axis_aligned 1 \
--epoch_num 50 \
--epoch_eval 1 \
--batch_size ${BATCH_SIZE} \
--eval_batch_size $((BATCH_SIZE * 2)) \
--lr 0.008 \
--lr_decay_steps 35 45 \
--lr_decay_rates 0.1 0.1 \
--weight_decay 0.01 \
--output_dir ${OUTPUT_DIR} \
--seed 1618
