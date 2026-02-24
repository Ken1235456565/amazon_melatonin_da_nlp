# ============================================================
# 2_train_model.py
# 读取 labeled.csv → embedding → 训练多标签分类器 → 保存模型
# ============================================================
import pandas as pd
import numpy as np
import ast
import joblib

from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ----------------------------------------------------------
# 1. 读取打标签数据
# ----------------------------------------------------------
df = pd.read_csv("labeled.csv")
df['labels'] = df['labels'].apply(ast.literal_eval)   # 字符串→list
print(f"加载 {len(df)} 条有标签样本")

# ----------------------------------------------------------
# 2. Sentence-transformers embedding
# ----------------------------------------------------------
encoder = SentenceTransformer('all-MiniLM-L6-v2')
print("Encoding...")
X = encoder.encode(
    df['clean_text'].tolist(),
    batch_size=64,
    show_progress_bar=True
)

# ----------------------------------------------------------
# 3. 多标签二值化
# ----------------------------------------------------------
mlb = MultiLabelBinarizer()
y   = mlb.fit_transform(df['labels'])
print("标签类别:", mlb.classes_)

# ----------------------------------------------------------
# 4. 训练
# ----------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
clf = OneVsRestClassifier(LogisticRegression(max_iter=1000, C=1.0))
clf.fit(X_train, y_train)

# ----------------------------------------------------------
# 5. 评估
# ----------------------------------------------------------
y_pred = clf.predict(X_test)
print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred, target_names=mlb.classes_))

# ----------------------------------------------------------
# 6. 保存
# ----------------------------------------------------------
joblib.dump(clf, 'clf_model.pkl')
joblib.dump(mlb, 'mlb.pkl')
encoder.save('encoder_model')
print("模型已保存: clf_model.pkl / mlb.pkl / encoder_model/")
