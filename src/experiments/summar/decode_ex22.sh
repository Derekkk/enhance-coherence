export CUDA_VISIBLE_DEVICES=1
python run.py \
  --model=cohere_extract \
  --data_path=data/cnndaily/test.pkl \
  --input_vocab=data/cnndaily/training.vocab \
  --input_vsize 150000 \
  --ckpt_root=checkpoints/archive/cnndaily/summar/ex22 \
  --mode=decode \
  --dropout 0.0 \
  --batch_size 1 \
  --use_bucketing False \
  --truncate_input True \
  --decode_dir out/cnndaily/summar/ex22 \
  --extract_topk 3 \
  --emb_dim 128 \
  --num_sents_doc 80 \
  --num_words_sent 50 \
  --rel_pos_max_idx 11 \
  --enc_num_hidden 256 \
  --enc_layers 1 \
  --pos_emb_dim 64 \
  --doc_repr_dim 512 \
  --word_conv_widths '3,5,7' \
  --word_conv_filters '128,256,256' \
  --mlp_num_hiddens '512,256' \

