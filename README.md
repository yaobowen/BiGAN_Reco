# BiGAN_Reco
To Load the data:
```python
dict = load("path_path_to_tinyimagenet")
class_names, X_train, y_train, X_val, y_val, X_test, y_test = dict["whole"]
class_names, X_train_toy, y_train_toy, X_val_toy, y_val_toy, X_test, y_test = dict["toy"]

```
To get the minibatch

```python
for n in xrange(n_epochs):
    for batch in getMiniBatch(X_train, y_train, batch_size):
        x_batch, y_batch = batch
        ...
```

To train
```bash
python AGE_32.py --dataset imagenet --c_dim 3 --z_dim 128 --nepoch 6 --drop_lr 3 --miu 10 --lamb 2000 --g_step 2
python AGE_64.py --dataset celeba --c_dim 3 --z_dim 64 --nepoch 5 --drop_lr 5 --miu 10 --lamb 500 --g_step 2
python AGE_32.py --dataset mnist --c_dim 1 --z_dim 10 --nepoch 10 --drop_lr 5 --miu 10 --lamb 500 --g_step 2
```

To sample
```bash
python AGE_32.py --mode sample --dataset imagenet --c_dim 3 --z_dim 128 --sample_size 8 --sample_seed 123
```