N_GPUS=1
BATCH_SIZE=32
DATA_ROOT=<your_data_root>

OUTPUT_DIR=<your_output_dir>/scan2sun/10_shots
echo "Processing 10-shots annotated target samples fine-tuning"

CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 torchrun \
--rdzv_endpoint localhost:29505 \
--nproc_per_node=${N_GPUS} \
main.py \
--mode source_only \
--data_root ${DATA_ROOT} \
--src_dataset sunrgbd \
--tgt_dataset sunrgbd \
--num_points_preload 100000 \
--num_points 40000 \
--categories bed bookshelf cabinet chair desk garbage_can lamp night_stand shelf sink sofa table toilet tv others \
--axis_aligned 1 \
--few_shot 10 \
--epoch_num 20 \
--epoch_eval 1 \
--batch_size 10 \
--eval_batch_size $((BATCH_SIZE * 2)) \
--lr 1e-7 \
--lr_decay_steps 10 15 \
--lr_decay_rates 0.1 0.1 \
--weight_decay 0.01 \
--ckpt_detector ${OUTPUT_DIR}/../source_only/model_best.pth \
--output_dir ${OUTPUT_DIR} \
--seed 1618

OUTPUT_DIR=<your_output_dir>/scan2sun/100_shots
echo "Processing 100-shots annotated target samples fine-tuning"

CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 torchrun \
--rdzv_endpoint localhost:29505 \
--nproc_per_node=${N_GPUS} \
main.py \
--mode source_only \
--data_root ${DATA_ROOT} \
--src_dataset sunrgbd \
--tgt_dataset sunrgbd \
--num_points_preload 100000 \
--num_points 40000 \
--categories bed bookshelf cabinet chair desk garbage_can lamp night_stand shelf sink sofa table toilet tv others \
--axis_aligned 1 \
--few_shot 100 \
--epoch_num 20 \
--epoch_eval 1 \
--batch_size ${BATCH_SIZE} \
--eval_batch_size $((BATCH_SIZE * 2)) \
--lr 1e-3 \
--lr_decay_steps 15 \
--lr_decay_rates 0.1 \
--weight_decay 0.01 \
--ckpt_detector ${OUTPUT_DIR}/../source_only/model_best.pth \
--output_dir ${OUTPUT_DIR} \
--seed 1618
