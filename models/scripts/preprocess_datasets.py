import os
from typing import Dict, List

import numpy as np
import pandas as pd


def normalize_label(raw_label: str) -> str:
    text = str(raw_label).strip().lower()

    if 'benign' in text or text in {'r', 'normal'}:
        return 'BENIGN'
    if 'dos' in text:
        return 'DoS'
    if 'fuzzy' in text:
        return 'Fuzzy'
    if 'rpm' in text:
        return 'RPM'
    if 'gear' in text or 'steering' in text:
        return 'GEAR'
    if 'speed' in text or 'gas' in text:
        return 'SPEED'

    return str(raw_label).strip()


def _parse_hex_int(value: str) -> int:
    text = str(value).strip().lower().replace('0x', '')
    if text == '':
        return 0
    try:
        return int(text, 16)
    except ValueError:
        try:
            return int(float(text))
        except ValueError:
            return 0


def parse_raw_can_file_fast(path: str, attack_label: str, source_name: str) -> pd.DataFrame:
    raw_cols = [
        'timestamp', 'id_hex', 'dlc', 'd0', 'd1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7', 'packet_flag'
    ]

    df = pd.read_csv(
        path,
        header=None,
        names=raw_cols,
        dtype=str,
        engine='c',
        on_bad_lines='skip',
    )

    for col in raw_cols:
        if col not in df.columns:
            df[col] = '0'

    df = df.fillna('0')

    out = pd.DataFrame({
        'timestamp': pd.to_numeric(df['timestamp'], errors='coerce').fillna(0.0),
        'id': df['id_hex'].map(_parse_hex_int).astype(np.int64),
        'dlc': pd.to_numeric(df['dlc'], errors='coerce').fillna(8).astype(np.int64),
        'val0': df['d0'].map(_parse_hex_int).astype(np.int64),
        'val1': df['d1'].map(_parse_hex_int).astype(np.int64),
        'val2': df['d2'].map(_parse_hex_int).astype(np.int64),
        'val3': df['d3'].map(_parse_hex_int).astype(np.int64),
        'val4': df['d4'].map(_parse_hex_int).astype(np.int64),
        'val5': df['d5'].map(_parse_hex_int).astype(np.int64),
        'val6': df['d6'].map(_parse_hex_int).astype(np.int64),
        'val7': df['d7'].map(_parse_hex_int).astype(np.int64),
        'flag': normalize_label(attack_label),
        'source': source_name,
    })

    return out


def parse_car_hacking_dataset(car_dir: str) -> pd.DataFrame:
    mapping = {
        'DoS_dataset.csv': 'DoS',
        'Fuzzy_dataset.csv': 'Fuzzy',
        'gear_dataset.csv': 'GEAR',
        'RPM_dataset.csv': 'RPM',
    }

    frames = []
    for filename, label in mapping.items():
        path = os.path.join(car_dir, filename)
        if os.path.exists(path):
            print(f'Lendo {path}...')
            parsed = parse_raw_can_file_fast(path, label, 'CARDT')
            print(f'Linhas lidas em {filename}: {len(parsed)}')
            frames.append(parsed)

    normal_path = os.path.join(car_dir, 'normal_run_data', 'normal_run_data.txt')
    if os.path.exists(normal_path):
        print(f'Lendo {normal_path}...')
        parsed = parse_raw_can_file_fast(normal_path, 'BENIGN', 'CARDT')
        print(f'Linhas lidas em normal_run_data.txt: {len(parsed)}')
        frames.append(parsed)

    if not frames:
        raise FileNotFoundError('Nenhum arquivo do Car-Hacking foi encontrado.')

    return pd.concat(frames, ignore_index=True)


def parse_ciciov_dataset(ciciov_decimal_dir: str) -> pd.DataFrame:
    csv_files = [
        x for x in os.listdir(ciciov_decimal_dir)
        if x.lower().endswith('.csv') and x.lower().startswith('decimal_')
    ]

    frames = []
    for filename in sorted(csv_files):
        path = os.path.join(ciciov_decimal_dir, filename)
        print(f'Lendo {path}...')

        df = pd.read_csv(path, low_memory=False)
        rename_map = {col: col.strip().upper() for col in df.columns}
        df = df.rename(columns=rename_map)

        if 'ID' not in df.columns:
            continue

        specific = 'SPECIFIC_CLASS' if 'SPECIFIC_CLASS' in df.columns else None
        category = 'CATEGORY' if 'CATEGORY' in df.columns else None
        label_col = specific or category
        if label_col is None:
            continue

        out = pd.DataFrame({
            'timestamp': 0.0,
            'id': pd.to_numeric(df['ID'], errors='coerce').fillna(0).astype(np.int64),
            'dlc': 8,
            'val0': pd.to_numeric(df.get('DATA_0', 0), errors='coerce').fillna(0).astype(np.int64),
            'val1': pd.to_numeric(df.get('DATA_1', 0), errors='coerce').fillna(0).astype(np.int64),
            'val2': pd.to_numeric(df.get('DATA_2', 0), errors='coerce').fillna(0).astype(np.int64),
            'val3': pd.to_numeric(df.get('DATA_3', 0), errors='coerce').fillna(0).astype(np.int64),
            'val4': pd.to_numeric(df.get('DATA_4', 0), errors='coerce').fillna(0).astype(np.int64),
            'val5': pd.to_numeric(df.get('DATA_5', 0), errors='coerce').fillna(0).astype(np.int64),
            'val6': pd.to_numeric(df.get('DATA_6', 0), errors='coerce').fillna(0).astype(np.int64),
            'val7': pd.to_numeric(df.get('DATA_7', 0), errors='coerce').fillna(0).astype(np.int64),
            'flag': df[label_col].astype(str).map(normalize_label),
            'source': 'CICIoV2024',
        })
        frames.append(out)

    if not frames:
        raise FileNotFoundError('Nenhum CSV decimal do CICIoV2024 foi encontrado.')

    return pd.concat(frames, ignore_index=True)


def find_existing_path(root: str, candidates: List[str]) -> str:
    for candidate in candidates:
        path = os.path.join(root, candidate)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f'Nenhum caminho encontrado entre: {candidates}')


def main():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

    car_dir = find_existing_path(root, [
        '9) Car-Hacking Dataset',
        'Car-Hacking Dataset',
        os.path.join('data', 'raw', 'CARDt'),
    ])

    ciciov_dir = find_existing_path(root, [
        os.path.join('CICIoV2024', 'decimal'),
        os.path.join('data', 'raw', 'CICIoV2024', 'decimal'),
    ])

    processed_dir = os.path.join(root, 'data', 'processed')
    os.makedirs(processed_dir, exist_ok=True)

    cardt_df = parse_car_hacking_dataset(car_dir)
    ciciov_df = parse_ciciov_dataset(ciciov_dir)

    all_df = pd.concat([cardt_df, ciciov_df], ignore_index=True)
    all_df = all_df.dropna(subset=['flag'])
    all_df['flag'] = all_df['flag'].map(normalize_label)

    aligned_path = os.path.join(processed_dir, 'all_datasets_aligned.csv')
    all_df.to_csv(aligned_path, index=False)

    benign_df = all_df[all_df['flag'] == 'BENIGN']
    attack_df = all_df[all_df['flag'] != 'BENIGN']

    target_benign = min(1_000_000, len(benign_df))
    if len(benign_df) > target_benign:
        benign_sampled = benign_df.sample(n=target_benign, random_state=42)
    else:
        benign_sampled = benign_df

    balanced_df = pd.concat([benign_sampled, attack_df], ignore_index=True)
    balanced_df = balanced_df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    balanced_path = os.path.join(processed_dir, 'all_datasets_aligned_balanced.csv')
    balanced_df.to_csv(balanced_path, index=False)

    metadata_path = os.path.join(processed_dir, 'metadata.md')
    class_counts = balanced_df['flag'].value_counts().sort_index()

    with open(metadata_path, 'w', encoding='utf-8') as f:
        f.write('# Metadata dos dados processados\n\n')
        f.write(f'- Total alinhado: {len(all_df)}\n')
        f.write(f'- Total balanceado: {len(balanced_df)}\n')
        f.write('- Colunas: timestamp, id, dlc, val0..val7, flag, source\n\n')
        f.write('## Distribuicao por classe (balanceado)\n\n')
        for cls, count in class_counts.items():
            f.write(f'- {cls}: {int(count)}\n')

    print(f'Arquivo alinhado salvo em: {aligned_path}')
    print(f'Arquivo balanceado salvo em: {balanced_path}')
    print(f'Metadata salva em: {metadata_path}')


if __name__ == '__main__':
    main()
