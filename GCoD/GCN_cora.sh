python train.py \
    --model GCN \
    --dataset Cora \
    --partition \
    --device cuda:0 \
    --save_prefix pretrain_partition \
    --quant \
    --enable_chunk_q \
    --num_act_bits 6 \
    --num_wei_bits 6 \
    --num_agg_bits 6