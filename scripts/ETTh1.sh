set -e

GPU=$1
ENC_TYPE=$2
DEC_TYPE=$3
DES=$4

ITERS=1

### M

python -u main_informer.py --model informer --data ETTh1 --features M --seq_len 48 --label_len 48 --pred_len 24 --e_layers 2 --d_layers 1 --attn full --des $DES --itr $ITERS --factor 3 --gpu $GPU --encoder_type $ENC_TYPE --decoder_type $DEC_TYPE

python -u main_informer.py --model informer --data ETTh1 --features M --seq_len 96 --label_len 48 --pred_len 48 --e_layers 2 --d_layers 1 --attn full --des $DES --itr $ITERS --gpu $GPU --encoder_type $ENC_TYPE --decoder_type $DEC_TYPE

python -u main_informer.py --model informer --data ETTh1 --features M --seq_len 168 --label_len 168 --pred_len 168 --e_layers 2 --d_layers 1 --attn full --des $DES --itr $ITERS --gpu $GPU --encoder_type $ENC_TYPE --decoder_type $DEC_TYPE

python -u main_informer.py --model informer --data ETTh1 --features M --seq_len 168 --label_len 168 --pred_len 336 --e_layers 2 --d_layers 1 --attn full --des $DES --itr $ITERS --gpu $GPU --encoder_type $ENC_TYPE --decoder_type $DEC_TYPE

python -u main_informer.py --model informer --data ETTh1 --features M --seq_len 336 --label_len 336 --pred_len 720 --e_layers 2 --d_layers 1 --attn full --des $DES --itr $ITERS --gpu $GPU --encoder_type $ENC_TYPE --decoder_type $DEC_TYPE

### S

python -u main_informer.py --model informer --data ETTh1 --features S --seq_len 720 --label_len 168 --pred_len 24 --e_layers 2 --d_layers 1 --attn full --des $DES --itr $ITERS --gpu $GPU --encoder_type $ENC_TYPE --decoder_type $DEC_TYPE

python -u main_informer.py --model informer --data ETTh1 --features S --seq_len 720 --label_len 168 --pred_len 48 --e_layers 2 --d_layers 1 --attn full --des $DES --itr $ITERS --gpu $GPU --encoder_type $ENC_TYPE --decoder_type $DEC_TYPE

python -u main_informer.py --model informer --data ETTh1 --features S --seq_len 720 --label_len 336 --pred_len 168 --e_layers 2 --d_layers 1 --attn full --des $DES --itr $ITERS --gpu $GPU --encoder_type $ENC_TYPE --decoder_type $DEC_TYPE

python -u main_informer.py --model informer --data ETTh1 --features S --seq_len 720 --label_len 336 --pred_len 336 --e_layers 2 --d_layers 1 --attn full --des $DES --itr $ITERS --gpu $GPU --encoder_type $ENC_TYPE --decoder_type $DEC_TYPE

python -u main_informer.py --model informer --data ETTh1 --features S --seq_len 720 --label_len 336 --pred_len 720 --e_layers 2 --d_layers 1 --attn full --des $DES --itr $ITERS --gpu $GPU --encoder_type $ENC_TYPE --decoder_type $DEC_TYPE
