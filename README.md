<h1>Introduction</h1>

This is our Pytorch implementation for the paper: "Multimodal Contrastive Learning for Sequential Recommendation".

<h1>Usage</h1>

In the training phase, please run:


>python3 src/main.py --emb_size 64 --model_name MMCrec --fusion_method attention --random_seed 0 --lmd 0.01 --gpu 1 --lr 1e-4 --l2 0 --history_max 50 --batch_size 256 --dataset 'beauty' --test_all 1


for the testing phase, please run:

>python3 src/main.py --emb_size 64 --model_name MMCrec --fusion_method attention --load 1 --random_seed 0 --lmd 0.01 --gpu 1 --lr 1e-4 --l2 0 --history_max 50 --batch_size 256 --dataset 'beauty' --test_all 1

