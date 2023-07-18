for i in 20 50 100
do
    for seed in 2020 2021 2022
    do
        python train.py --alg MAML --num_shot $i --num_epoch 50 --result_path result$seed --seed $seed
        python train.py --alg iMAML --num_shot $i --num_epoch 50 --result_path result$seed --seed $seed
        python train.py --alg FOMAML --num_shot $i --num_epoch 50 --result_path result$seed --seed $seed
        python train.py --alg SignMAML --num_shot $i --num_epoch 50 --result_path result$seed --seed $seed
    done
done