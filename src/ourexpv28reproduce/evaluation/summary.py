"""The readable summary table of a baseline run with several scores (EEGNet, M-ShallowConvNet)."""
from .metrics import METRICS, format_metrics


def epoch_note(scores: dict) -> str:
    if "epoch" in scores:
        return f"   (epoch {scores['epoch']})"
    if "epochs" in scores:
        return f"   (epochs {'/'.join(str(e) for e in scores['epochs'])})"
    return ""


def scores_summary(result: dict) -> list[str]:
    protocol = result["protocol"].replace("_", " ")
    extra = "".join(f", {name.replace('_', ' ')} {result[name]}" for name in ("dropout", "cutcat_ratio", "transformer_depth", "selected_channels_count", "input_features")
                    if name in result)
    lines = [f"{result['paper']}, {result['model']}, {result['dataset']} ({result['class_names'][1]} against "
             f"{result['class_names'][0]}), {protocol}, seed {result['training']['seed']}{extra}",
             "Samples, subjects and folds of version 27.", ""]
    for subject, scores in result["results"].items():
        for i, name in enumerate(result["scores"]):
            prefix = f"S{subject:<3}" if i == 0 else "    "
            lines.append(f"{prefix} {name.replace('_', ' '):<20} {format_metrics(scores[name])}{epoch_note(scores[name])}")
    lines += ["", f"Mean over {len(result['results'])} subjects"]
    for name in result["scores"]:
        lines.append(f"     {name.replace('_', ' '):<20} "
                     f"{format_metrics({k: result['summary'][name][k] for k in METRICS})}")
    return lines + [""] + [f"{name}: {text}" for name, text in result["evaluation"].items()]
