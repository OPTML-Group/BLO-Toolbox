#!/usr/bin/bash
#echo "*********************** generate data ***********************"
python3 generate_data.py --distribution Rayleigh-Rice-Geometry10-Geometry50 --num_train 2000-2000-2000-2000 --noise 1 --o data/dataset_balance.pt
TrainScript="--hidden_layers 200-80-80 --batch_size 500 --noise 1 --mini_batch_size 10 --lr 0.001 --n_memories 200 --data_file data/dataset_balance.pt --file_ext _balance"

# python3 generate_data.py --distribution Geometry10-Geometry20-Geometry50-Rayleigh --num_train 2000-2000-2000-2000 --noise 1 --o data/dataset_balance.pt
# TrainScript="--hidden_layers 200-80-80 --batch_size 500 --noise 1 --mini_batch_size 10 --lr 0.001 --n_memories 200 --data_file data/dataset_balance.pt --file_ext _balance"

# python3 generate_data.py --distribution Rice-Geometry10-Rayleigh-Geometry50 --num_train 2000-2000-2000-2000 --noise 1 --o data/dataset_balance.pt
# TrainScript="--hidden_layers 200-80-80 --batch_size 500 --noise 1 --mini_batch_size 50 --lr 0.001 --n_memories 200 --data_file data/dataset_balance.pt --file_ext _balance"


echo "*********************** TL ***********************"
python3 main.py $TrainScript --model single

echo "*********************** SIGD ***********************"
python3 main.py $TrainScript --model bilevel --lr 0.001 

echo "*********************** Generate Figure ***********************"
python3 generate_figure.py --ext _balance
