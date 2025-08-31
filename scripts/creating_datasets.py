import sys
from pathlib import Path
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.append(str(project_root))
from src.extract_raw_data import extract_data_from_duckdb
from src.preprocess_data import preprocess_all_data
from src.spliting_data import process_and_save_datasets 

def create_datasets_pipeline():
	print("Извлечение данных из wb-feedbacks")
	try:
		extract_data_from_duckdb()
		print("Данные извлечены")
	except Exception as e:
		print(f"Ошибка при извлечении данных: {e}")
		return
	
	print("Предобработка данных")
	try:
		ru_processed, wb_processed = preprocess_all_data()
		print("Данные предобработаны")
		print(f"RuReviews: {len(ru_processed)} отзывов")
		print(f"wb-feedbacks: {len(wb_processed)} отзывов")
	except Exception as e:
		print(f"Ошибка при предобработке данных: {e}")
		return
	
	print("Разделение и сохранение данных")
	try:
		test_df, val_df, train_df = process_and_save_datasets()
		print("Данные разделены и сохранены")
		print(f"Train: {len(train_df)} отзывов")
		print(f"Validation: {len(val_df)} отзывов")
		print(f"Test: {len(test_df)} отзывов")
	except Exception as e:
		print(f"Ошибка при разделении данных: {e}")
		return

if __name__ == "__main__":
	create_datasets_pipeline()