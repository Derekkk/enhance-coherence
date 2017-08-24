export CUDA_VISIBLE_DEVICES=0
python run.py \
  --model=summarunner \
  --data_path=data/cnndaily/test.pkl \
  --input_vocab=data/cnndaily/training.vocab \
  --input_vsize 150000 \
  --ckpt_root=checkpoints/archive/cnndaily/summar/ex02 \
  --mode=decode \
  --dropout 0.0 \
  --batch_size 1 \
  --use_bucketing False \
  --truncate_input True \
  --decode_dir out/cnndaily/summar/ex02 \
  --extract_topk 3 \
  --emb_dim 128 \
  --num_sentences 80 \
  --num_words_sent 50 \
  --rel_pos_max_idx 11 \
  --enc_num_hidden 512 \
  --enc_layers 1 \
  --pos_emb_dim 64 \
  --doc_repr_dim 512 \
  --word_conv_k_sizes '1,3,5,7' \
  --word_conv_filter 128 \

