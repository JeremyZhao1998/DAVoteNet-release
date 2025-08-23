N_GPUS=1
BATCH_SIZE=32
DATA_ROOT=<your_data_root>
OUTPUT_DIR=<your_output_dir>/pf2front/vss

python vss_process.py \
--data_root ${DATA_ROOT} \
--src_dataset procfront \
--split_set train \
--axis_aligned 1

python vss_process.py \
--data_root ${DATA_ROOT} \
--src_dataset procfront \
--split_set val \
--axis_aligned 1

CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 torchrun \
--rdzv_endpoint localhost:28502 \
--nproc_per_node=${N_GPUS} \
main.py \
--mode source_only \
--data_root ${DATA_ROOT} \
--src_dataset procfront_vss \
--tgt_dataset 3dfront \
--num_points_preload 100000 \
--num_points 40000 \
--categories bed cabinet chair desk lamp shelf sofa table tv_stand others \
--axis_aligned 1 \
--epoch_num 10 \
--epoch_eval 1 \
--batch_size ${BATCH_SIZE} \
--eval_batch_size $((BATCH_SIZE * 2)) \
--lr 1e-8 \
--lr_decay_steps 8 \
--lr_decay_rates 0.1 \
--weight_decay 0.01 \
--ckpt_detector ${OUTPUT_DIR}/../source_only/model_best.pth \
--output_dir ${OUTPUT_DIR} \
--seed 1618
