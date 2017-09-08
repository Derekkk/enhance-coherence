export CUDA_VISIBLE_DEVICES=0
python run.py \
  --model=seqmatch \
  --data_path=data/cnndaily/test.pkl \
  --input_vocab=data/cnndaily/training.vocab \
  --input_vsize 150000 \
  --ckpt_root=checkpoints/archive/cnndaily/seqmatch/ex13 \
  --mode=decode \
  --batch_size 1 \
  --use_bucketing False \
  --truncate_input True \
  --seqmatch_type "conv_match" \
  --max_sent_len 50 \
  --sm_emb_dim 64 \
  --sm_margin 1.0 \
  --sm_conv1d_filter 128 \
  --sm_conv1d_width 3 \
  --sm_conv_filters "256,512" \
  --sm_conv_heights "3,3" \
  --sm_conv_widths "3,3" \
  --sm_maxpool_widths "2,2" \
  --sm_fc_num_units "512,256" \
  --sm_eval_1_in_k 4 \
  --decode_dir=out/cnndaily/seqmatch/ex13 \

