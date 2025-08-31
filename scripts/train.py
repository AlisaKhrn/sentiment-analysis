import torch
import pandas as pd
from transformers import (AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding)
from sklearn.metrics import f1_score, accuracy_score
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from paths import TRAIN_PATH, VALIDATION_PATH, MODELS_DIR
from src.dataset import RuSentimentDataset

MODEL_NAME = "ai-forever/ruBert-base"
DROPOUT_RATE = 0.2

def compute_metrics(eval_pred):
	logits, labels = eval_pred
	predictions = logits.argmax(axis=-1)
	return {
		"macro_f1": f1_score(labels, predictions, average='macro'),
		"accuracy": accuracy_score(labels, predictions)
	}

def load_train_val_data():
	train_df = pd.read_csv(TRAIN_PATH)
	val_df = pd.read_csv(VALIDATION_PATH)
	
	label_mapping = {1: 0, 3: 1, 5: 2}
	train_df['label'] = train_df['rating'].map(label_mapping)
	val_df['label'] = val_df['rating'].map(label_mapping)
	
	return (
		train_df['text'].tolist(), train_df['label'].values,
		val_df['text'].tolist(), val_df['label'].values
	)

def main():
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	
	train_texts, train_labels, val_texts, val_labels = load_train_val_data()
	
	tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
	
	train_dataset = RuSentimentDataset(train_texts, train_labels, tokenizer)
	val_dataset = RuSentimentDataset(val_texts, val_labels, tokenizer)
	
	model = AutoModelForSequenceClassification.from_pretrained(
		MODEL_NAME,
		num_labels=3,
		hidden_dropout_prob=DROPOUT_RATE,
		attention_probs_dropout_prob=DROPOUT_RATE,
		classifier_dropout=DROPOUT_RATE,
		use_safetensors=True
	)
	model.to(device)
	
	training_args = TrainingArguments(
		output_dir=Path(MODELS_DIR)/"training_output",
		eval_strategy="epoch",
		save_strategy="epoch",
		learning_rate=2e-5,
		per_device_train_batch_size=32,
		per_device_eval_batch_size=32,
		num_train_epochs=3,
		weight_decay=0.1,
		load_best_model_at_end=True,
		metric_for_best_model="macro_f1",
		greater_is_better=True,
		save_total_limit=1,
		report_to=None,
		fp16=torch.cuda.is_available(),
		logging_strategy="steps",
		logging_steps=10,
		dataloader_pin_memory=True if torch.cuda.is_available() else False
	)
	
	data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
	
	trainer = Trainer(
		model=model,
		args=training_args,
		train_dataset=train_dataset,
		eval_dataset=val_dataset,
		compute_metrics=compute_metrics,
		data_collator=data_collator,
	)
	
	trainer.train()
	trainer.save_model(Path(MODELS_DIR)/"final_model")
	tokenizer.save_pretrained(Path(MODELS_DIR)/"final_model")
	
	history = trainer.state.log_history

	train_losses = []
	eval_f1_scores = []
	
	for log in history:
		if 'loss' in log:
			train_losses.append(log['loss'])
		if 'eval_macro_f1' in log:
			eval_f1_scores.append(log['eval_macro_f1'])
	
	eval_results = trainer.evaluate()
	print(f"F1: {round(eval_results['eval_macro_f1'], 4)*100}")
	print(f"Accuracy: {round(eval_results['eval_accuracy'], 4)*100}")
	
	for epoch, (loss, f1) in enumerate(zip(train_losses, eval_f1_scores), 1):
		print(f"Эпоха {epoch}: Loss = {loss:.4f}, F1 = {f1:.4f}")

if __name__ == "__main__":
	main()