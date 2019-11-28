import pandas as pd
import numpy as np
import csv
from sklearn.metrics import classification_report
from utils import read_json_data, write_json_data


# samples = read_json_data("qna_data/test.json")
# label_samples = []

# for s in samples:
#     for p in s["paragraphs"]:
#         p["label"] = ""
#     label_samples.append(s)

# write_json_data("qna_data/label_test.json", label_samples)


# check labeled test
old_details = pd.read_csv("models/bert-base-multilingual-cased-domain/test_bert-base-multilingual-cased-domain_512_8_3.0_qnli_details.csv")
labeled_test_df = pd.read_csv("qna_data/glue_data/vi/ltest.tsv", sep="\t")

preds = old_details['pred'].values
y = np.array([0 if l == 'entailment' else 1 for l in labeled_test_df['label']])


df_confusion = pd.crosstab(y, preds, rownames=['Actual'], colnames=['Predicted'], margins=True)
print (classification_report(y, preds, target_names=['has_answer', 'no_answer']))
print (df_confusion)

diff = y != preds

y1_pred0 = diff & (y == 1)
y0_pred1 = diff & (y == 0)
print (y1_pred0.sum())
print (y0_pred1.sum())
y1_pred0_ids = old_details.loc[y1_pred0, 'id'].values
y0_pred1_ids = old_details.loc[y0_pred1, 'id'].values

with open('y1_pred0_ids.txt', 'w') as f:
    for _id in y1_pred0_ids:
        f.write('{}\n'.format(_id))

with open('y0_pred1_ids.txt', 'w') as f:
    for _id in y0_pred1_ids:
        f.write('{}\n'.format(_id))
