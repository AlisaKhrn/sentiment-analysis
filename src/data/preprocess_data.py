import re
import emoji
import pandas as pd
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from config import WB_RAW_PATH, WB_CLEANED_PATH, RAW_RUREVIEWS_PATH, RUREVIEWS_PATH

def clean_text(text: str) -> str:
	text = str(text).lower()
	text = emoji.replace_emoji(text, replace=' ')
	text = re.sub(r'[!"№;%:?*()\-=\/\|@#$^&{}\'.,~`\n\r\t]+', ' ', text)
	text = re.sub(r'\s+', ' ', text).strip()
	return text

def is_english(text: str) -> bool:
	return bool(re.fullmatch(r"[A-Za-z0-9\s]+", text))

def preprocess_wb_data() -> pd.DataFrame:
	wb = pd.read_csv(WB_RAW_PATH)
	wb['text'] = wb['text'].astype(str).apply(clean_text)
	wb = wb[~wb["text"].apply(is_english)]
	wb = wb[wb["text"].str.split().str.len() > 3]
	wb = wb.dropna(subset=['rating', 'text'])
	wb = wb.drop_duplicates(subset='text', keep='first')
	return wb

def preprocess_rureviews() -> pd.DataFrame:
	labels = {"positive": 5, "negative": 1, "neautral": 3}
	data_ru = []
	with open(RAW_RUREVIEWS_PATH, "r", encoding="utf-8") as f:
		for line in f:
			line = line.strip()
			words = line.split()
			if words and words[-1] in labels:
				rating = labels[words[-1]]
				text = " ".join(words[:-1])
				data_ru.append([rating, text])
	ru = pd.DataFrame(data_ru, columns=["rating", "text"])
	ru["text"] = ru["text"].astype(str).apply(clean_text)
	ru = ru[ru["text"].str.strip() != ""]
	ru = ru[~ru["text"].apply(is_english)]
	ru = ru[ru["text"].str.split().str.len() > 1]
	return ru

def remove_cross_duplicates(df1: pd.DataFrame, df2: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
	common_texts = set(df1['text']).intersection(set(df2['text']))
	df2_clean = df2[~df2['text'].isin(common_texts)]
	return df1, df2_clean

def save_processed_data(ru: pd.DataFrame, wb: pd.DataFrame):
	ru.to_csv(RUREVIEWS_PATH, index=False)
	wb.to_csv(WB_CLEANED_PATH, index=False)

def preprocess_all_data():
	wb = preprocess_wb_data()
	ru = preprocess_rureviews()
	ru, wb = remove_cross_duplicates(ru, wb)
	save_processed_data(ru, wb)
	return ru, wb

if __name__ == "__main__":
	preprocess_all_data()