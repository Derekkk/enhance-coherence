export CUDA_VISIBLE_DEVICES=1
python run.py \
  --model=summarunner \
  --data_path=data/cnn/training_shuf.pkl \
  --input_vocab=data/cnn/training.vocab \
  --input_vsize 150000 \
  --ckpt_root=checkpoints/ex08 \
  --summary_dir=log/ex08 \
  --mode=train \
  --lr 0.5 \
  --min_lr 0.00001 \
  --decay_step 30000 \
  --decay_rate 0.5 \
  --batch_size 64 \
  --max_run_steps 1000000 \
  --num_gpus 1 \
  --valid_path=data/cnn/validation_shuf.pkl \
  --valid_freq 1000 \
  --checkpoint_secs 1800 \
  --display_freq 100 \
  --use_bucketing False \
  --truncate_input True \
  --emb_dim 150 \
  --num_sentences 80 \
  --num_words_sent 50 \
  --rel_pos_max_idx 11 \
  --enc_num_hidden 300 \
  --pos_emb_dim 50 \
  --doc_repr_dim 750 \
  --word_conv_k_sizes '3,5,7' \
  --word_conv_filter 100 \
  --min_num_input_sents 3 \
