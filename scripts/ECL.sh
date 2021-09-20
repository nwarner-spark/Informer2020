set -e

GPU=$1
ENC_TYPE=$2

python3 -u main_informer.py --model informer --data ECL --features M --attn prob --d_layers 2 --e_layers 3 --itr 3 --label_len 168 --pred_len 24 --seq_len 168 --des 'Exp' --gpu $GPU --encoder_type $ENC_TYPE --itr 1

## Multivariate

python3 -u main_informer.py --model informer --data ECL --features M --attn prob --d_layers 2 --e_layers 3 --itr 3 --label_len 168 --pred_len 24 --seq_len 168 --des 'Exp' --gpu $GPU --encoder_type $ENC_TYPE --itr 1

python -u main_informer.py --model informer --data ECL --features M --attn prob --d_layers 2 --e_layers 3 --itr 3 --label_len 96 --pred_len 48 --seq_len 96 --des 'Exp' --gpu $GPU --encoder_type $ENC_TYPE --itr 1

python3 -u main_informer.py --model informer --data ECL --features M --attn prob --d_layers 2 --e_layers 3 --itr 3 --label_len 168 --pred_len 168 --seq_len 336 --des 'Exp' --gpu $GPU --encoder_type $ENC_TYPE --itr 1

python -u main_informer.py --model informer --data ECL --features M --attn prob --d_layers 2 --e_layers 3 --itr 3 --label_len 168 --pred_len 336 --seq_len 720 --des 'Exp' --gpu $GPU --encoder_type $ENC_TYPE --itr 1

# [run on V100:]
python -u main_informer.py --model informer --data ECL —features M --attn prob --d_layers 2 --e_layers 3 --itr 3 --label_len 336 --pred_len 720 --seq_len 720 --des 'Exp' --gpu $GPU --encoder_type $ENC_TYPE --itr 1
