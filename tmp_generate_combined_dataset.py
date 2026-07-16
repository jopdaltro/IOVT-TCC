from pathlib import Path
import sys
import pandas as pd

root = Path(__file__).resolve().parent
sys.path.insert(0, str(root))
from models.scripts.preprocess_datasets import find_existing_path, parse_car_hacking_dataset, parse_ciciov_dataset

car_dir = find_existing_path(str(root), [
    '9) Car-Hacking Dataset',
    'Car-Hacking Dataset',
    str(root / 'data' / 'raw' / 'CARDt'),
])
ciciov_dir = find_existing_path(str(root), [
    str(root / 'CICIoV2024' / 'decimal'),
    str(root / 'data' / 'raw' / 'CICIoV2024' / 'decimal'),
])
print('car_dir', car_dir)
print('ciciov_dir', ciciov_dir)

car_df = parse_car_hacking_dataset(car_dir)
print('car_df counts', car_df['flag'].value_counts().to_dict())

ciciov_df = parse_ciciov_dataset(str(ciciov_dir), include_types=['BENIGN', 'DoS'])
print('ciciov_df counts', ciciov_df['flag'].value_counts().to_dict())

combined = pd.concat([car_df, ciciov_df], ignore_index=True)
print('combined shape', combined.shape)
print('combined counts', combined['flag'].value_counts().to_dict())
out_path = root / 'data' / 'processed' / 'carhacking_ciciov_benign_dos_aligned.csv'
combined.to_csv(out_path, index=False)
print('saved', out_path)
