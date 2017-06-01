python AGE_64.py --dataset imagenet --batch_size 64 --nepoch 12 --drop_lr 3 --c_dim 3 --z_dim 128 --miu 10 --lamb 2000 --g_step 2

python AGE_64.py --dataset imagenet --batch_size 256 --nepoch 12 --drop_lr 3 --c_dim 3 --z_dim 128 --miu 10 --lamb 2000 --g_step 2 --restore True