import torch
from torch.utils.data import Dataset

class RuSentimentDataset(Dataset):
	def __init__(self, texts, labels, tokenizer):
		self.texts = texts
		self.labels = labels
		self.tokenizer = tokenizer

	def __len__(self):
		return len(self.texts)

	def __getitem__(self, idx):
		encoding = self.tokenizer(
			self.texts[idx],
			padding="max_length",
			truncation=True,
			max_length=512,
			return_tensors="pt"
		)
		return {
			"input_ids": encoding["input_ids"].squeeze(0),
			"attention_mask": encoding["attention_mask"].squeeze(0),
			"labels": torch.tensor(self.labels[idx], dtype=torch.long),
		}