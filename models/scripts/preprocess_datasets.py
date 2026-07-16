import argparse
import os
import re
from typing import Dict, List, Optional

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


def parse_normal_run_data_file(path: str, source_name: str) -> pd.DataFrame:
    rows = []

    timestamp_re = re.compile(r'Timestamp:\s*([0-9]+\.[0-9]+)')
    id_re = re.compile(r'ID:\s*([0-9A-Fa-f]+)')
    dlc_re = re.compile(r'DLC:\s*(\d+)')
    byte_re = re.compile(r'\b[0-9A-Fa-f]{2}\b')

    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            timestamp_m = timestamp_re.search(line)
            id_m = id_re.search(line)
            dlc_m = dlc_re.search(line)
            byte_values = byte_re.findall(line)
            if not timestamp_m or not id_m or not dlc_m or len(byte_values) < 8:
                continue

            rows.append({
                'timestamp': float(timestamp_m.group(1)),
                'id_hex': id_m.group(1),
                'dlc': int(dlc_m.group(1)),
                'd0': byte_values[0],
                'd1': byte_values[1],
                'd2': byte_values[2],
                'd3': byte_values[3],
                'd4': byte_values[4],
                'd5': byte_values[5],
                'd6': byte_values[6],
                'd7': byte_values[7],
                'packet_flag': 'R',
            })

    if not rows:
        raise FileNotFoundError(f'Nenhum linha válida encontrada em {path}')

    df = pd.DataFrame(rows)
    return pd.DataFrame({
        'timestamp': df['timestamp'],
        'id': df['id_hex'].map(_parse_hex_int).astype(np.int64),
        'dlc': df['dlc'].astype(np.int64),
        'val0': df['d0'].map(_parse_hex_int).astype(np.int64),
        'val1': df['d1'].map(_parse_hex_int).astype(np.int64),
        'val2': df['d2'].map(_parse_hex_int).astype(np.int64),
        'val3': df['d3'].map(_parse_hex_int).astype(np.int64),
        'val4': df['d4'].map(_parse_hex_int).astype(np.int64),
        'val5': df['d5'].map(_parse_hex_int).astype(np.int64),
        'val6': df['d6'].map(_parse_hex_int).astype(np.int64),
        'val7': df['d7'].map(_parse_hex_int).astype(np.int64),
        'flag': 'BENIGN',
        'source': source_name,
    })


def parse_raw_can_file_fast(path: str, attack_label: str, source_name: str, normal_text: bool = False) -> pd.DataFrame:
    if normal_text:
        return parse_normal_run_data_file(path, source_name)

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
    df['packet_flag'] = df['packet_flag'].astype(str).str.strip().str.upper()

    attack_label_norm = normalize_label(attack_label)
    if attack_label_norm == 'BENIGN':
        flags = ['BENIGN'] * len(df)
    else:
        flags = [
            'BENIGN' if flag == 'R' else attack_label_norm
            for flag in df['packet_flag']
        ]

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
        'flag': flags,
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


def _should_skip_ciciov_file(filename: str, exclude_types: Optional[List[str]]) -> bool:
    if not exclude_types:
        return False
    filename_lower = filename.lower()
    for exclude in exclude_types:
        if exclude.strip().lower() in filename_lower:
            return True
    return False


def parse_ciciov_dataset(ciciov_decimal_dir: str, exclude_types: Optional[List[str]] = None, include_types: Optional[List[str]] = None) -> pd.DataFrame:
    csv_files = [
        x for x in os.listdir(ciciov_decimal_dir)
        if x.lower().endswith('.csv') and x.lower().startswith('decimal_')
    ]

    frames = []
    for filename in sorted(csv_files):
        if include_types is not None:
            basename = os.path.splitext(filename)[0].lower()
            if not any(t.lower() in basename for t in include_types):
                print(f'Pulando {filename} (classe CICIoV não incluída)')
                continue
        elif _should_skip_ciciov_file(filename, exclude_types):
            print(f'Pulando {filename} (classe CICIoV excluída)')
            continue

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


def balance_to_targets(df: pd.DataFrame, targets: Dict[str, int], random_state: int = 42) -> pd.DataFrame:
    df = df[df['flag'].isin(targets.keys())].copy()
    frames = []
    for label, target in targets.items():
        subset = df[df['flag'] == label]
        if len(subset) == 0:
            raise ValueError(f'Nenhum exemplo encontrado para classe {label}')
        if len(subset) > target:
            subset = subset.sample(n=target, random_state=random_state)
        elif len(subset) < target:
            subset = subset.sample(n=target, replace=True, random_state=random_state)
        frames.append(subset)
    balanced = pd.concat(frames, ignore_index=True)
    return balanced.sample(frac=1.0, random_state=random_state).reset_index(drop=True)


def find_existing_path(root: str, candidates: List[str]) -> str:
    for candidate in candidates:
        path = os.path.join(root, candidate)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f'Nenhum caminho encontrado entre: {candidates}')


def main():
    parser = argparse.ArgumentParser(
        description='Preprocessa CARDT e CICIoV2024 e grava datasets alinhados. '
                    'Use --bytes-only para salvar apenas os bytes e a flag.'
    )
    parser.add_argument(
        '--bytes-only',
        action='store_true',
        help='Salvar apenas as colunas val0..val7 e flag no dataset final.',
    )
    parser.add_argument(
        '--exclude-ciciov-types',
        nargs='*',
        default=['GAS', 'SPEED', 'STEERING_WHEEL'],
        help='Tipos de ataque decimal CICIoV a excluir do dataset.',
    )
    parser.add_argument(
        '--include-ciciov-types',
        nargs='*',
        default=None,
        help='Tipos de CICIoV a incluir explicitamente no dataset (override de exclusão).',
    )
    parser.add_argument(
        '--use-article-balance',
        action='store_true',
        help='Balancear o dataset com as quantidades exatas do artigo.',
    )
    parser.add_argument(
        '--balanced-benign-target',
        type=int,
        default=1_000_000,
        help='Número máximo de amostras BENIGN após undersampling.',
    )
    parser.add_argument(
        '--rename-val1',
        action='store_true',
        help='Renomear colunas val0..val7 para val1..val8 no arquivo final.',
    )
    parser.add_argument(
        '--drop-id',
        action='store_true',
        help='Remover a coluna `id` do arquivo final (como no artigo).',
    )
    parser.add_argument(
        '--min-class-count',
        type=int,
        default=0,
        help='Remover classes com menos de N amostras antes de salvar (0 = nenhuma).',
    )
    args = parser.parse_args()

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
    if args.use_article_balance and args.include_ciciov_types is None:
        args.include_ciciov_types = ['BENIGN', 'DoS']

    ciciov_df = parse_ciciov_dataset(
        ciciov_dir,
        exclude_types=args.exclude_ciciov_types,
        include_types=args.include_ciciov_types,
    )

    all_df = pd.concat([cardt_df, ciciov_df], ignore_index=True)
    all_df = all_df.dropna(subset=['flag'])
    all_df['flag'] = all_df['flag'].map(normalize_label)

    # Remover classes com poucas amostras, se solicitado
    if args.min_class_count and args.min_class_count > 0:
        counts = all_df['flag'].value_counts()
        to_drop = counts[counts < args.min_class_count].index.tolist()
        if to_drop:
            print(f"Removendo classes com menos de {args.min_class_count} amostras: {to_drop}")
            all_df = all_df[~all_df['flag'].isin(to_drop)]

    if args.include_ciciov_types is not None:
        print(f'Incluindo apenas tipos CICIoV: {args.include_ciciov_types}')

    if args.use_article_balance:
        targets = {
            'BENIGN': 1_000_000,
            'DoS': 662_184,
            'RPM': 654_897,
            'GEAR': 597_252,
            'Fuzzy': 491_846,
        }
        print('Balanceando dataset para os valores exatos do artigo.')
        final_df = balance_to_targets(all_df, targets)
    else:
        benign_df = all_df[all_df['flag'] == 'BENIGN']
        attack_df = all_df[all_df['flag'] != 'BENIGN']

        target_benign = min(args.balanced_benign_target, len(benign_df))
        if len(benign_df) > target_benign:
            benign_sampled = benign_df.sample(n=target_benign, random_state=42)
        else:
            benign_sampled = benign_df

        final_df = pd.concat([benign_sampled, attack_df], ignore_index=True)
        final_df = final_df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    if args.bytes_only:
        final_df = final_df[['val0', 'val1', 'val2', 'val3', 'val4', 'val5', 'val6', 'val7', 'flag']]
        prefix = 'bytes_only_'
    else:
        prefix = ''

    # Opcao para renomear val0..val7 -> val1..val8
    if args.rename_val1:
        rename_map = {f'val{i}': f'val{i+1}' for i in range(0, 8)}
        final_df = final_df.rename(columns=rename_map)

    # Opcao para dropar a coluna id (se quiser reproduzir processamento que removeu id)
    if args.drop_id and 'id' in final_df.columns:
        final_df = final_df.drop(columns=['id'])

    # Remover duplicatas no dataset final
    before_dedup = len(final_df)
    final_df = final_df.drop_duplicates().reset_index(drop=True)
    removed_duplicates = before_dedup - len(final_df)
    if removed_duplicates > 0:
        print(f'Removendo {removed_duplicates} registros duplicados do dataset final.')

    aligned_path = os.path.join(processed_dir, f'{prefix}aligned.csv')
    final_df.to_csv(aligned_path, index=False)

    balanced_df = final_df
    balanced_path = os.path.join(processed_dir, f'{prefix}aligned_balanced.csv')
    balanced_df.to_csv(balanced_path, index=False)

    metadata_path = os.path.join(processed_dir, 'metadata.md')
    class_counts = balanced_df['flag'].value_counts().sort_index()

    with open(metadata_path, 'w', encoding='utf-8') as f:
        f.write('# Metadata dos dados processados\n\n')
        f.write(f'- Total alinhado: {len(all_df)}\n')
        f.write(f'- Total balanceado: {len(balanced_df)}\n')
        cols = list(all_df.columns)
        f.write(f"- Colunas: {', '.join(cols)}\n\n")
        f.write('## Distribuicao por classe (balanceado)\n\n')
        for cls, count in class_counts.items():
            f.write(f'- {cls}: {int(count)}\n')

    print(f'Arquivo alinhado salvo em: {aligned_path}')
    print(f'Arquivo balanceado salvo em: {balanced_path}')
    print(f'Metadata salva em: {metadata_path}')


if __name__ == '__main__':
    main()
