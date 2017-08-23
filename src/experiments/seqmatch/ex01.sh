export CUDA_VISIBLE_DEVICES=2
python run.py \
  --model=seqmatch \
  --data_path=data/cnndaily/training_shuf.pkl \
  --input_vocab=data/cnndaily/training.vocab \
  --input_vsize 150000 \
  --ckpt_root=checkpoints/cnndaily/ex01 \
  --summary_dir=log/cnndaily/ex01 \
  --mode=train \
  --lr 0.1 \
  --min_lr 0.0001 \
  --decay_step 30000 \
  --decay_rate 0.5 \
  --dropout 0.0 \
  --max_run_steps 1000000 \
  --batch_size 64 \
  --valid_path=data/cnndaily/validation_shuf.pkl \
  --valid_freq 1000 \
  --display_freq 100 \
  --checkpoint_secs 1200 \
  --use_bucketing False \
  --truncate_input True \
  --emb_dim 128 \
  --num_hidden 256 \
  --max_sent_len 50 \
  --conv_filters 256 \
  --conv_width 3 \
  --maxpool_width 2 \

