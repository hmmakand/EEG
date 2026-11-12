"""Final-test ROC, subject scores, validation curves and selected epochs for version 29."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', type=Path, nargs='+')
    args = parser.parse_args(argv)
    for path in args.results:
        data = json.loads(path.read_text())
        folder = path.with_name(path.stem + '_figures')
        folder.mkdir(parents=True, exist_ok=True)
        folds = [(subject, fold) for subject, result in data['subjects'].items()
                 for fold in result.get('folds', [result])]
        title = f"{data['dataset']} · {data['protocol']} · validation-selected refit"
        fig, ax = plt.subplots()
        for subject, fold in folds:
            history = fold['validation']['history']
            ax.plot([r['epoch'] for r in history], [r['balanced_accuracy'] for r in history], alpha=.35)
        ax.set(xlabel='Selection epoch', ylabel='Validation balanced accuracy', title=title, ylim=(0,1))
        fig.tight_layout(); fig.savefig(folder/'validation_curves.png', dpi=200); plt.close(fig)
        fig, ax = plt.subplots()
        ax.hist([fold['selected_epoch'] for _, fold in folds], bins='auto')
        ax.set(xlabel='Selected refit epochs', ylabel='Outer folds', title=title)
        fig.tight_layout(); fig.savefig(folder/'selected_epochs.png', dpi=200); plt.close(fig)
        fig, ax = plt.subplots()
        subjects = sorted(data['subjects'], key=int)
        ax.bar(subjects, [data['subjects'][s]['accuracy'] for s in subjects])
        ax.set(xlabel='Subject', ylabel='Final test accuracy', ylim=(0,1), title=title)
        ax.tick_params(axis='x', labelrotation=90)
        fig.tight_layout(); fig.savefig(folder/'subjects.png', dpi=200); plt.close(fig)
        labels = [v for _, fold in folds for v in fold['labels']]
        scores = [v for _, fold in folds for v in fold['scores']]
        fig, ax = plt.subplots()
        if len(set(labels)) == 2:
            fpr,tpr,_ = roc_curve(labels,scores)
            ax.plot(fpr,tpr,label=f'Pooled test AUC = {roc_auc_score(labels,scores):.3f}')
        ax.plot([0,1],[0,1],'--',color='gray')
        ax.set(xlabel='False positive rate',ylabel='True positive rate',title=title)
        ax.legend(); fig.tight_layout(); fig.savefig(folder/'roc.png',dpi=200); plt.close(fig)
        (folder/'measurement_notes.txt').write_text(
            'ROC pools outer-test predictions; it is not mean subject AUC.\n'
            + json.dumps(data.get('validity', {}),indent=2) + '\n')
        print(f'Figures saved to {folder}')


if __name__ == '__main__':
    main()
