python qlora_train.py \
    --model_id "meta-llama/Meta-Llama-3-8B-Instruct" \
    --batch_size 2 \
    --epochs 5 \
    --save_dir /data/jykim/plav/llama_3/exp1 \
    --exp_name llama_3_exp_1 \
    --token <hf-token>