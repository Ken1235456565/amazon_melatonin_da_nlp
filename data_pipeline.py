# ============================================================
# 1_data_pipeline.py
# 加载原始数据 → 清洗 → 弱监督打标签 → 保存 labeled.csv
# ============================================================
import pandas as pd
import re
import warnings
warnings.filterwarnings('ignore')

# ----------------------------------------------------------
# 1. 加载 & 合并
# ----------------------------------------------------------
files = [
    "melatonin-amazon-1mg.csv",
    "melatonin-amazon-3mg.csv",
    "melatonin-amazon-5mg.csv",
    "melatonin-amazon-10mg.csv",
    "melatonin-amazon-12mg.csv",
    "melatonin-amazon-20mg.csv",
]
raw_df = pd.concat(
    [pd.read_csv(f, encoding="ISO-8859-1") for f in files],
    ignore_index=True
)

# ----------------------------------------------------------
# 2. 清洗
# ----------------------------------------------------------
raw_df['HelpfulCounts'] = pd.to_numeric(raw_df['HelpfulCounts'], errors='coerce').fillna(0)
raw_df['ReviewDate']    = pd.to_datetime(raw_df['ReviewDate'], errors='coerce')
raw_df = raw_df.drop(columns=['Images'], errors='ignore')
raw_df = raw_df[raw_df['ReviewContent'].notna()]
raw_df['ReviewTitle'] = raw_df['ReviewTitle'].fillna('')

df = raw_df[raw_df['Verified'] == True].copy()
df.drop(["PageUrl","ParentId","ProductLink","ProductTitle","Reviewer","Verified"],
        axis=1, inplace=True, errors='ignore')
df = df.drop_duplicates(subset='ReviewContent')

def clean(text):
    text = re.sub(r'https?://\S+', '', str(text))
    text = re.sub(r'[^A-Za-z0-9\s]', ' ', text)
    return text.lower().strip()

df['clean_text'] = df['ReviewContent'].apply(clean)

# ----------------------------------------------------------
# 3. 弱监督打标签（基于真实评论文本，无合成数据）
# ----------------------------------------------------------
label_rules = {
    'efficacy': [
        r'works?\s*(great|well|perfectly|amazing|wonders|quick|fast)',
        r'(helped?|helps?)\s+me\s+sleep',
        r'\beffective\b', r'put me to sleep',
        r"didn'?t?\s+work", r'no effect', r'not effective',
        r'stopped working', r"doesn'?t?\s+work",
    ],
    'taste': [
        r'taste[sd]?\s+(good|great|bad|awful|gross|horrible)',
        r'(good|great|bad|awful)\s+flavor',
        r'\bstrawberry\b', r'\bchalky\b', r'\bbitter\b',
        r'\baftertaste\b', r'\byummy\b', r'\bdelicious\b',
    ],
    'side_effect': [
        r'\bgroggy\b', r'\bgrogginess\b', r'\bhangover\b',
        r'\bnightmare', r'vivid\s+dream', r'weird\s+dream',
        r'\bheadache\b', r'\bdizziness?\b', r'\bnausea\b',
        r'no\s+side\s+effects?', r'no\s+groggy',
        r"doesn'?t?\s+make\s+me\s+groggy", r'no\s+hangover',
    ],
    'sleep_quality': [
        r'fall\s+asleep', r'fell\s+asleep',
        r'stay\s+asleep', r'slept?\s+through\s+the\s+night',
        r'all\s+night', r'wake\s+up\s+(in\s+the\s+)?middle',
        r'waking\s+up', r'kept\s+waking',
        r'trouble\s+falling\s+asleep', r'\binsomnia\b',
    ],
}

def assign_labels(text):
    labels = []
    for label, patterns in label_rules.items():
        for p in patterns:
            if re.search(p, text):
                labels.append(label)
                break
    return labels

df['labels'] = df['clean_text'].apply(assign_labels)
labeled = df[df['labels'].map(len) > 0].copy()

# ----------------------------------------------------------
# 4. 统计 & 保存
# ----------------------------------------------------------
from collections import Counter
print(f"总样本: {len(df)}  有标签样本: {len(labeled)}")
print("各标签数量:", Counter(l for ls in labeled['labels'] for l in ls))

labeled[['clean_text', 'labels', 'ReviewScore']].to_csv("labeled.csv", index=False)
print("已保存 labeled.csv")
