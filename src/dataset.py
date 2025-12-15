import pandas as pd
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from functools import partial

class BookDataset(Dataset):
	def __init__(self, data_path):
		super().__init__()
		self.df = pd.read_parquet(data_path)

		# 100개 미만인 카테고리는 노이즈로 간주하고 삭제
		counts = self.df['category'].value_counts()
		valid_categories = counts[counts > 100].index
		self.df = self.df[self.df['category'].isin(valid_categories)]
		self.df = self.df.reset_index(drop = True) # 데이터셋 인덱스 문제로 인한 성능저하 수정

		# 라벨 인코딩
		le = LabelEncoder()
		le.fit(self.df['category'])   # 전체 데이터로 학습
		self.df['label'] = le.transform(self.df['category'])

		# 입력 텍스트 생성!! 새 컬럼 text에 대해서.... 문장 만듦
		self.df["text"] = self.df.apply(self.build_text, axis=1)
	
	def __len__(self): 
		return len(self.df)

	def __getitem__(self, idx):
		row = self.df.iloc[idx]
		return {
			"text": row["text"],
			"label": row["label"]
		}

	def build_text(self, row):
		parts = [
			f"Title: {row['title']} |",
			f"Description: {row['description']}"
		]
		return " ".join( # 리스트의 문자열들을 공백으로 연결할건데.....
			[p for p in parts if isinstance(p, str)] # NaN이나 None이 있으면 제외함
		) # 최종적으로 하나의 문장 형태로 반환한다고 함!! "Title: ... Category: ... Description: ..."

def collate_fn(batch, tokenizer): # DataLoader가 batch마다 호출
    # texts = [f"passage: {x['text']}" for x in batch]
    texts = [f"query: {x['text']}" for x in batch]
    labels = torch.tensor([x['label'] for x in batch])  # 라벨을 int 리스트 → torch.tensor 로 변환

    inputs = tokenizer(
      texts, padding=True, truncation=True, max_length=256, return_tensors="pt")

    return inputs, labels

def get_loader(data_path, batch_size, tokenizer):
	dataset = BookDataset(data_path)

	train_dataset, valid_dataset = random_split(
		dataset, [int(len(dataset)*0.8), len(dataset) - int(len(dataset)*0.8)]
	)

	collate_fn_token = partial(collate_fn, tokenizer=tokenizer)
	train_loader = DataLoader(
		train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_token
	)
	valid_loader = DataLoader(
		valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_token
	)
	
	return train_loader, valid_loader