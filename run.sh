#python AGE_32.py --dataset imagenet --batch_size 64 --nepoch 6 --drop_lr 3 --c_dim 3 --z_dim 128 --miu 10 --lamb 2000 --g_step 2

#python AGE_32.py --dataset imagenet --batch_size 256 --nepoch 6 --drop_lr 3 --c_dim 3 --z_dim 128 --miu 10 --lamb 2000 --g_step 2 --restore True

python AGE_64.py --dataset celeba --batch_size 64 --nepoch 20 --drop_lr 10 --c_dim 3 --z_dim 64 --miu 10 --lamb 1000 --g_step 3

python AGE_64.py --dataset celeba --batch_size 256 --nepoch 20 --drop_lr 10 --c_dim 3 --z_dim 64 --miu 15 --lamb 1000 --g_step 3 --restore True