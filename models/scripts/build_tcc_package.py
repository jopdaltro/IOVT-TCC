import datetime as dt
import glob
import json
import os
import shutil
import subprocess
import sys
from typing import List


def safe_copy(files: List[str], dest_dir: str) -> int:
    os.makedirs(dest_dir, exist_ok=True)
    copied = 0
    for path in files:
        if os.path.isfile(path):
            shutil.copy2(path, os.path.join(dest_dir, os.path.basename(path)))
            copied += 1
    return copied


def run_script(root: str, script_name: str):
    script_path = os.path.join(root, 'models', 'scripts', script_name)
    if not os.path.exists(script_path):
        print(f'Script nao encontrado: {script_path}')
        return

    print(f'Executando {script_name}...')
    completed = subprocess.run([sys.executable, script_path], cwd=root, check=False)
    if completed.returncode != 0:
        print(f'Aviso: {script_name} retornou codigo {completed.returncode}.')


def main():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    package_root = os.path.join(root, 'results', 'tcc_package')

    figures_dir = os.path.join(package_root, 'figures')
    tables_dir = os.path.join(package_root, 'tables')
    reports_dir = os.path.join(package_root, 'reports')
    artifacts_dir = os.path.join(package_root, 'artifacts')

    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(artifacts_dir, exist_ok=True)

    run_script(root, 'build_model_comparison.py')
    run_script(root, 'build_learning_curves.py')

    metrics_files = glob.glob(os.path.join(root, 'results', 'metrics', '**', '*.json'), recursive=True)
    metrics_files += glob.glob(os.path.join(root, 'results', 'metrics', '**', '*.csv'), recursive=True)
    metrics_files += glob.glob(os.path.join(root, 'results', 'metrics', '**', '*.txt'), recursive=True)

    copied_metrics = safe_copy(metrics_files, os.path.join(artifacts_dir, 'metrics_raw'))

    legacy_figures = glob.glob(os.path.join(root, 'results', 'visualizations', '*'))
    copied_legacy = safe_copy(legacy_figures, os.path.join(figures_dir, 'legacy'))

    manifest = {
        'generated_at': dt.datetime.now().isoformat(),
        'package_root': package_root,
        'copied_metrics_files': copied_metrics,
        'copied_legacy_figures': copied_legacy,
        'notes': [
            'Comparacao de modelos sem score composto.',
            'Analise de overfitting por learning curves com F1 macro.',
            'AUC/ROC nao usados nos novos relatorios do pacote TCC.',
        ],
    }

    with open(os.path.join(package_root, 'manifest.json'), 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)

    print('Pacote TCC montado em: results/tcc_package')


if __name__ == '__main__':
    main()
