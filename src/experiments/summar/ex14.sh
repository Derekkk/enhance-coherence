export CUDA_VISIBLE_DEVICES=1
python run.py \
  --model=summarunner_rf \
  --data_path=data/cnndaily/training_shuf.pkl \
  --input_vocab=data/cnndaily/training.vocab \
  --input_vsize 150000 \
  --ckpt_root=checkpoints/cnndaily/summar/ex14 \
  --summary_dir=log/cnndaily/summar/ex14 \
  --mode=train \
  --train_mode='sl+rl' \
  --rl_coef 0.3 \
  --lr 0.5 \
  --min_lr 0.01 \
  --decay_step 30000 \
  --decay_rate 0.98 \
  --dropout 0.0 \
  --batch_size 32 \
  --max_run_steps 200000 \
  --display_freq 100 \
  --valid_path=data/cnndaily/validation_shuf.pkl \
  --valid_freq 1000 \
  --num_valid_batch 60 \
  --checkpoint_secs 1200 \
  --use_bucketing False \
  --truncate_input True \
  --min_num_input_sents 3 \
  --emb_dim 128 \
  --num_sentences 80 \
  --num_words_sent 50 \
  --rel_pos_max_idx 11 \
  --enc_num_hidden 256 \
  --enc_layers 1 \
  --pos_emb_dim 128 \
  --doc_repr_dim 512 \
  --word_conv_k_sizes '3,5,7' \
  --word_conv_filter 128 \
  --mlp_num_hidden '512,256' \
  
