N_GPUS=1
BATCH_SIZE=32
DATA_ROOT=<your_data_root>
OUTPUT_DIR=<your_output_dir>/front2sun/vss

python vss_process.py \
--data_root ${DATA_ROOT} \
--src_dataset 3dfront \
--split_set train \
--axis_aligned 0

python vss_process.py \
--data_root ${DATA_ROOT} \
--src_dataset 3dfront \
--split_set val \
--axis_aligned 0

CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 torchrun \
--rdzv_endpoint localhost:27502 \
--nproc_per_node=${N_GPUS} \
main.py \
--mode source_only \
--data_root ${DATA_ROOT} \
--src_dataset 3dfront_vss \
--tgt_dataset sunrgbd \
--num_points_preload 100000 \
--num_points 40000 \
--categories bed bookshelf cabinet chair desk lamp night_stand shelf sofa table others \
--axis_aligned 0 \
--epoch_num 20 \
--epoch_eval 1 \
--batch_size ${BATCH_SIZE} \
--eval_batch_size $((BATCH_SIZE * 2)) \
--lr 0.0001 \
--lr_decay_steps 5 15 \
--lr_decay_rates 0.1 0.1 \
--weight_decay 0.01 \
--output_dir ${OUTPUT_DIR} \
--ckpt_detector ${OUTPUT_DIR}/../source_only/model_best.pth
--seed 1618
