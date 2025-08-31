import re
import emoji
import pandas as pd
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from paths import WB_RAW_PATH, WB_CLEANED_PATH, RAW_RUREVIEWS_PATH, RUREVIEWS_PATH

def clean_text(text):
	text = str(text).lower()
	text = emoji.replace_emoji(text, replace=' ')
	text = re.sub(r'[!"№;%:?*()=\/\|@#$^&{}\,.~`\n\r\t]+', ' ', text)
	text = re.sub(r'\d+', ' ', text)
	text = re.sub(r'\s+', ' ', text).strip()
	return text

def contains_english(text, threshold=3):
	english_pattern = r'[a-zA-Z]{' + str(threshold) + ',}'
	return bool(re.search(english_pattern, text))

def contains_russian(text, min_chars=5):
	russian_chars = re.findall(r'[а-яё]', text, re.IGNORECASE)
	return len(russian_chars) >= min_chars

def preprocess_wb():
	wb = pd.read_csv(WB_RAW_PATH)
	wb['text'] = wb['text'].astype(str).apply(clean_text)
	wb = wb[~wb["text"].apply(contains_english)]
	wb = wb[wb["text"].apply(contains_russian)]
	wb = wb[wb["text"].str.split().str.len() > 3]
	wb = wb.dropna(subset=['rating', 'text'])
	wb = wb.drop_duplicates(subset='text', keep='first')
	return wb

def preprocess_rureviews():
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
	ru = ru[~ru["text"].apply(contains_english)]
	ru = ru[ru["text"].apply(contains_russian)]
	ru = ru[ru["text"].str.split().str.len() > 1]
	return ru

def remove_cross_duplicates(df1, df2):
	df1_clean = df1.copy()
	df2_clean = df2.copy()
	common_texts = set(df1_clean['text']).intersection(set(df2_clean['text']))
	mask_df1 = ~df1_clean['text'].isin(common_texts)
	mask_df2 = ~df2_clean['text'].isin(common_texts)
	df1_clean = df1_clean[mask_df1]
	df2_clean = df2_clean[mask_df2]
	return df1_clean, df2_clean

def save_processed_data(ru, wb):
	ru.to_csv(RUREVIEWS_PATH, index=False)
	wb.to_csv(WB_CLEANED_PATH, index=False)

def preprocess_all_data():
	wb = preprocess_wb()
	ru = preprocess_rureviews()
	ru, wb = remove_cross_duplicates(ru, wb)
	save_processed_data(ru, wb)
	return ru, wb