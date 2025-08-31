import torch
import pandas as pd
from transformers import (AutoTokenizer, AutoModelForSequenceClassification, Trainer, DataCollatorWithPadding)
from sklearn.metrics import classification_report, f1_score
import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from paths import TEST_PATH, MODELS_DIR
from src.dataset import RuSentimentDataset

def load_test_data():
	test_df = pd.read_csv(TEST_PATH)
	label_mapping = {1: 0, 3: 1, 5: 2}
	test_df['label'] = test_df['rating'].map(label_mapping)
	return test_df['text'].tolist(), test_df['label'].values

def main():
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	
	test_texts, test_labels = load_test_data()
	
	tokenizer = AutoTokenizer.from_pretrained(MODELS_DIR)
	model = AutoModelForSequenceClassification.from_pretrained(MODELS_DIR)
	model.to(device)
	
	test_dataset = RuSentimentDataset(test_texts, test_labels, tokenizer)
	data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
	
	trainer = Trainer(
		model=model,
		data_collator=data_collator,
	)
	
	predictions = trainer.predict(test_dataset)
	y_pred = np.argmax(predictions.predictions, axis=-1)
	
	print(classification_report(test_labels, y_pred, target_names=['Negative', 'Neutral', 'Positive']))
	print(f"Macro F1: {round(f1_score(test_labels, y_pred, average='macro'), 4)*100}")
	

if __name__ == "__main__":
	main()