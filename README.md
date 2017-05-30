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

To run
```bash
python AGE_64.py --dataset imagenet --nepoch 6 --drop_lr 3 --c_dim 3 --z_dim 128 --miu 10 --lamb 1000 --g_step 3
python AGE_64.py --dataset celeba --nepoch 5 --drop_lr 5 --c_dim 3 --z_dim 64 --miu 10 --lamb 500 --g_step 2
python AGE_32.py --dataset mnist --nepoch 10 --drop_lr 5 --c_dim 1 --z_dim 10 --miu 10 --lamb 500 --g_step 2
```