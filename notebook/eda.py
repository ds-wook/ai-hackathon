# %%
import numpy as np
import pandas as pd
# %%
mem1 = pd.read_csv("../input/ai-hackaton/mem-128-2-model1.csv")
mem1.head()
# %%
mem1.sort_values(by="id")
# %%
cnn_oof = np.load("../submit/total_cat_oof.npy")
cnn_preds = np.load("../submit/total_cat_pred.npy")

train_labels = pd.read_csv("../input/ai-hackaton/train_labels.csv")
cnn_oof = pd.DataFrame(cnn_oof, columns=[i for i in range(61)])

cnn_oof = pd.concat([train_labels.id, cnn_oof], axis=1)
cnn_oof.to_csv("total_cat_oof.csv", index=False)

submission = pd.read_csv("../input/ai-hackaton/sample_submission.csv")
submission.iloc[:, 1:] = np.load("../submit/total_cat_pred.npy")
submission.to_csv("total_cat_preds.csv", index=False)
# %%
train_y = pd.read_csv("../input/ai-hackaton/train_labels.csv")
train_y.label
# %%
from sklearn.metrics import log_loss

log_loss(train_y.label, cnn_oof)
# %%
