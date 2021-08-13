set -e

N_VARS=8

if [ ! -d "data/VAR_n_vars=$N_VARS" ] 
then
    # copy data from GCP
    gsutil cp -R gs://amb-dev/nbm-data/benchmarks/VAR/VAR_n_vars=$N_VARS data/
fi

for n_delays in 8 16 32 64 128 256 512 1024
do
    echo "Running n_delays=$n_delays"
    python -u main_informer.py --model informer --data custom --root_path data/VAR_n_vars=$N_VARS/ --data_path n_delays=$n_delays.csv --target 7 --enc_in $N_VARS --dec_in $N_VARS --c_out $N_VARS --features M --seq_len 96 --label_len 48 --pred_len 48 --e_layers 2 --d_layers 1 --attn full --des 'Exp' --itr 5
done
