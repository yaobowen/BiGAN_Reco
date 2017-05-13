# BiGAN_Reco
To Load the data:
```python
dict = load("path_path_to_tinyimagenet")
class_names, X_train, y_train, X_val, y_val, X_test, y_test = dict["whole"]
class_names, X_train_toy, y_train_toy, X_val_toy, y_val_toy, X_test, y_test = dict["toy"]

```
