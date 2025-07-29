import duckdb
import pandas as pd
import sys

from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from config import DUCKDB_PATH, WB_RAW_PATH

def extract_data_from_duckdb():
	con = duckdb.connect(str(DUCKDB_PATH))
	try:
		ratings = [1, 3, 5]
		dfs = []
		for rating in ratings:
			query = f"""
			SELECT productValuation AS rating, text
			FROM data
			WHERE productValuation = {rating}
			LIMIT 120000;
			"""
			df = con.execute(query).fetchdf()
			dfs.append(df)
		combined_df = pd.concat(dfs, ignore_index=True)
		output_path = WB_RAW_PATH
		combined_df.to_csv(output_path, index=False)
	finally:
		con.close()

if __name__ == "__main__":
	extract_data_from_duckdb()