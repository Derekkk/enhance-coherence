export CUDA_VISIBLE_DEVICES=2
python run.py \
  --model=seqmatch \
  --data_path=data/cnndaily/training_shuf.pkl \
  --input_vocab=data/cnndaily/training.vocab \
  --input_vsize 200000 \
  --ckpt_root=checkpoints/cnndaily/ex04 \
  --summary_dir=log/cnndaily/ex04 \
  --mode=train \
  --lr 0.2 \
  --min_lr 0.1 \
  --decay_step 30000 \
  --decay_rate 0.98 \
  --dropout 0.0 \
  --max_run_steps 1500000 \
  --batch_size 64 \
  --valid_path=data/cnndaily/validation_shuf.pkl \
  --valid_freq 1000 \
  --display_freq 100 \
  --checkpoint_secs 1200 \
  --use_bucketing False \
  --truncate_input True \
  --emb_dim 128 \
  --num_hidden 512 \
  --max_sent_len 50 \
  --conv_filters 512 \
  --conv_width 5 \
  --maxpool_width 2 \

