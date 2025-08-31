import pandas as pd
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from paths import RUREVIEWS_PATH, WB_CLEANED_PATH, TEST_PATH, VALIDATION_PATH, TRAIN_PATH

def load_data():
	ru = pd.read_csv(RUREVIEWS_PATH)
	wb = pd.read_csv(WB_CLEANED_PATH)
	return ru, wb

def extract_samples(df, samples_per_rating):
	selected_samples = []
	remaining_df = df.copy()
	for rating, n_samples in samples_per_rating.items():
		rating_df = remaining_df[remaining_df["rating"] == rating]
		if len(rating_df) > 0:
			samples = rating_df.sample(n=min(n_samples, len(rating_df)), random_state=42)
			selected_samples.append(samples)
			remaining_df = remaining_df.drop(samples.index)

	result_df = pd.concat(selected_samples, ignore_index=True) if selected_samples else pd.DataFrame()
	return result_df, remaining_df

def add_missing_samples(target_df, source_df, target_counts):
	result_df = target_df.copy()
	for rating, target_count in target_counts.items():
		current_count = len(result_df[result_df['rating'] == rating])
		deficit = target_count - current_count
		if deficit > 0:
			available_samples = source_df[source_df['rating'] == rating]
			used_texts = set(result_df['text'])
			available_samples = available_samples[~available_samples['text'].isin(used_texts)]
			if len(available_samples) > 0:
				samples_to_add = available_samples.sample(n=min(deficit, len(available_samples)), random_state=42)
				result_df = pd.concat([result_df, samples_to_add], ignore_index=True)

	return result_df

def prepare_test_val_datasets(ru, wb):
	test_samples = {5: 7000, 3: 7000, 1: 7000}
	test_df, ru_remaining = extract_samples(ru, test_samples)
	
	val_samples_ru = {5: 3500, 3: 3500, 1: 3500}
	val_ru, ru_remaining = extract_samples(ru_remaining, val_samples_ru)
	val_samples_wb = {5: 3500, 3: 3500, 1: 3500}
	val_wb, wb_remaining = extract_samples(wb, val_samples_wb)
	val_df = pd.concat([val_ru, val_wb], ignore_index=True)
	
	test_df = test_df.sample(frac=1, random_state=42).reset_index(drop=True)
	val_df = val_df.sample(frac=1, random_state=42).reset_index(drop=True)
	
	test_df = test_df.drop_duplicates(subset='text', keep='first')
	val_df = val_df.drop_duplicates(subset='text', keep='first')
	
	return test_df, val_df, ru_remaining, wb_remaining

def prepare_train_dataset(ru_remaining, wb_remaining):
	train_samples = []
	
	for rating in [1, 3, 5]:
		ru_rating = ru_remaining[ru_remaining["rating"] == rating]
		train_samples.append(ru_rating)
	
	train_df = pd.concat(train_samples, ignore_index=True)
	train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
	train_df = train_df.drop_duplicates(subset='text', keep='first')
	
	return train_df

def remove_cross_duplicates_final(test_df, val_df, train_df):
	test_df['source'] = 'test'
	val_df['source'] = 'validation'
	train_df['source'] = 'train'
	all_data = pd.concat([test_df, val_df, train_df], ignore_index=True)
	duplicate_mask = all_data.duplicated(subset='text', keep=False)
	duplicates = all_data[duplicate_mask]
	
	if not duplicates.empty:
		unique_data = all_data.drop_duplicates(subset='text', keep='first')
		test_clean = unique_data[unique_data['source'] == 'test'].drop('source', axis=1)
		val_clean = unique_data[unique_data['source'] == 'validation'].drop('source', axis=1)
		train_clean = unique_data[unique_data['source'] == 'train'].drop('source', axis=1)
		return test_clean, val_clean, train_clean
	else:
		return test_df.drop('source', axis=1), val_df.drop('source', axis=1), train_df.drop('source', axis=1)

def balance_datasets_to_target(test_df, val_df, train_df, wb_remaining):
	target_per_rating = {1: 7000, 3: 7000, 5: 7000}
	test_balanced = add_missing_samples(test_df, wb_remaining, target_per_rating)
	val_balanced = add_missing_samples(val_df, wb_remaining, target_per_rating)
	train_target_counts = {1: 56000, 3: 56000, 5: 56000}
	train_balanced = add_missing_samples(train_df, wb_remaining, train_target_counts)
	return test_balanced, val_balanced, train_balanced

def process_and_save_datasets():
	ru, wb = load_data()
	test_df, val_df, ru_remaining, wb_remaining = prepare_test_val_datasets(ru, wb)
	train_df = prepare_train_dataset(ru_remaining, wb_remaining)
	test_clean, val_clean, train_clean = remove_cross_duplicates_final(test_df, val_df, train_df)
	test_final, val_final, train_final = balance_datasets_to_target(
		test_clean, val_clean, train_clean, wb_remaining
	)
	test_final = test_final.sample(frac=1, random_state=42).reset_index(drop=True)
	val_final = val_final.sample(frac=1, random_state=42).reset_index(drop=True)
	train_final = train_final.sample(frac=1, random_state=42).reset_index(drop=True)
	
	test_final.to_csv(TEST_PATH, index=False)
	val_final.to_csv(VALIDATION_PATH, index=False)
	train_final.to_csv(TRAIN_PATH, index=False)

	return test_final, val_final, train_final
