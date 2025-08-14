N_GPUS=1
BATCH_SIZE=32
DATA_ROOT=<your_data_root>
OUTPUT_DIR=<your_output_dir>/proc2scan/size_prior

CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 torchrun \
--rdzv_endpoint localhost:28506 \
--nproc_per_node=${N_GPUS} \
main.py \
--mode source_only \
--data_root ${DATA_ROOT} \
--src_dataset procthor \
--tgt_dataset scannet \
--num_points_preload 100000 \
--num_points 40000 \
--categories bed cabinet chair desk lamp shelf sofa table others \
--axis_aligned 1 \
--mean_size_prior 1 \
--epoch_num 90 \
--epoch_eval 1 \
--batch_size ${BATCH_SIZE} \
--eval_batch_size $((BATCH_SIZE * 2)) \
--lr 0.008 \
--lr_decay_steps 65 80 \
--lr_decay_rates 0.1 0.1 \
--weight_decay 0.01 \
--output_dir ${OUTPUT_DIR} \
--seed 1618
