"""Original trial extraction only; no classifier or generator partitions."""
from pathlib import Path
import numpy as np
if __package__ == 'datasetsynthetic':
    from dataset.data_io import load_mat_file, read_sessions
    from dataset.trials import extract_trials
else:
    from ..dataset.data_io import load_mat_file, read_sessions
    from ..dataset.trials import extract_trials

def extract_original(path, subject, config):
    sessions = list(read_sessions(load_mat_file(path)))
    trials = extract_trials(sessions, subject, config.sample_rate, config.trials)
    # Canonical original order, independent of class grouping and legacy interleaving.
    trials.sort(key=lambda trial: (trial.session_id, trial.trial_id))
    by_session = {session.session_id: session for session in sessions}
    identities = {(trial.session_id, trial.trial_id) for trial in trials}
    exclusions = []
    for session in sessions:
        if session.session_id not in config.trials.session_indices:
            continue
        for index, label in enumerate(session.labels):
            if (session.session_id, index) not in identities:
                exclusions.append(dict(subject_id=subject, session_id=session.session_id,
                                       original_trial_index=index, original_class_label=int(label),
                                       reason='unselected_class' if int(label) not in dict(config.trials.label_map)
                                       else 'incomplete_trial'))
    rows = []
    for i, trial in enumerate(trials):
        session = by_session[trial.session_id]
        position = session.positions[trial.trial_id].item()
        rows.append(dict(source_file=Path(path).name, subject_id=subject,
                         session_id=trial.session_id, original_trial_index=trial.trial_id,
                         original_event_position=position,
                         start_sample=int(position) + config.trials.event_index_offset,
                         original_class_label=int(session.labels[trial.trial_id]), class_label=trial.label,
                         array_index=i, split_seed=config.seed))
    if not trials:
        raise ValueError(f'Subject {subject} has no eligible trials')
    return np.stack([trial.signal for trial in trials]), rows, exclusions

