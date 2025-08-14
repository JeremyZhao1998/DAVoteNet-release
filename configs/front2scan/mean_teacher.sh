N_GPUS=1
BATCH_SIZE=16
DATA_ROOT=<your_data_root>
OUTPUT_DIR=<your_output_dir>/front2scan/mean_teacher

CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 torchrun \
--rdzv_endpoint localhost:26503 \
--nproc_per_node=${N_GPUS} \
main.py \
--mode teaching \
--data_root ${DATA_ROOT} \
--src_dataset 3dfront \
--tgt_dataset scannet \
--num_points_preload 100000 \
--num_points 40000 \
--categories bed bookshelf cabinet chair desk lamp night_stand shelf sofa table others \
--axis_aligned 1 \
--epoch_num 20 \
--epoch_eval 1 \
--batch_size ${BATCH_SIZE} \
--eval_batch_size $((BATCH_SIZE * 2)) \
--lr 1e-9 \
--lr_decay_steps 10 \
--lr_decay_rates 0.1 \
--weight_decay 0.01 \
--coef_tgt 0.5 \
--obj_threshold 0.9 \
--sem_threshold 0.9 \
--output_dir ${OUTPUT_DIR} \
--ckpt_detector ${OUTPUT_DIR}/../source_only/model_best.pth \
--seed 1618
