"""Versioned, audited cohorts for the empirical sweep."""
from pathlib import Path
import argparse
import hashlib
import io
import json
import sys
import urllib.request
import zipfile
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.rq5_guarantees.prepare import assign_splits, adult, synthetic

BASE = Path('/path/to/results/rq5/sweep_v2')
DATASETS = ['synthetic', 'adult', 'credit_default', 'bank_marketing', 'compas_race',
            'acs_income', 'acs_employment', 'acs_coverage',
            'ptb_mi', 'ptb_sttc', 'ptb_cd', 'ptb_hyp']


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def download(url, name):
    dest = BASE / 'raw' / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        tmp = dest.with_suffix(dest.suffix + '.partial')
        with urllib.request.urlopen(url, timeout=180) as src, open(tmp, 'wb') as out:
            while chunk := src.read(1024 * 1024):
                out.write(chunk)
        tmp.replace(dest)
    return dest


def frame(features, a, y, unit=None, age=None):
    features = features.copy()
    # Hash task inputs + sensitive attribute: duplicate feature rows stay together.
    if unit is None:
        unit = pd.util.hash_pandas_object(features.assign(sensitive=a), index=False).astype(str)
    out = features.add_prefix('x_')
    out['a'], out['y'], out['unit'] = np.asarray(a), np.asarray(y), np.asarray(unit).astype(str)
    if age is not None:
        out['age'] = np.asarray(age)
    return assign_splits(out.reset_index(drop=True))


def credit():
    p = download('https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip', 'credit.zip')
    with zipfile.ZipFile(p) as z:
        name = next(n for n in z.namelist() if n.endswith('.xls'))
        d = pd.read_excel(io.BytesIO(z.read(name)), header=1)
    a = d.SEX.eq(1).astype(int)
    y = d.iloc[:, -1].astype(int)
    f = d.iloc[:, :-1].drop(columns=['ID', 'SEX'])
    for c in ['EDUCATION', 'MARRIAGE'] + [c for c in f if c.startswith('PAY_') and not c.startswith('PAY_AMT')]:
        f[c] = f[c].astype(str)
    return frame(f, a, y, age=d.AGE)


def bank():
    p = download('https://archive.ics.uci.edu/static/public/222/bank+marketing.zip', 'bank.zip')
    with zipfile.ZipFile(p) as outer:
        with zipfile.ZipFile(io.BytesIO(outer.read('bank.zip'))) as z:
            d = pd.read_csv(io.BytesIO(z.read('bank-full.csv')), sep=';')
    # Age group is the sensitive attribute. Duration is post-contact leakage.
    return frame(d.drop(columns=['age', 'duration', 'y']), d.age.ge(40).astype(int),
                 d.y.eq('yes').astype(int), age=d.age)


def compas():
    p = download('https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv', 'compas.csv')
    d = pd.read_csv(p)
    d = d[d.days_b_screening_arrest.between(-30, 30) & d.is_recid.ne(-1)
          & d.c_charge_degree.ne('O') & d.score_text.ne('N/A')
          & d.race.isin(['African-American', 'Caucasian'])].copy()
    f = d[['age', 'sex', 'juv_fel_count', 'juv_misd_count', 'juv_other_count',
           'priors_count', 'c_charge_degree']].copy()
    # Identity is used only to keep repeated people together; never a predictor/output.
    unit = pd.util.hash_pandas_object(d[['name', 'dob']], index=False).astype(str)
    return frame(f, d.race.eq('African-American').astype(int), d.two_year_recid.astype(int), unit, d.age)


def target_split(d, prefix):
    d = assign_splits(d)
    d['split'] = d.split.map({'train': prefix + '_probe_train', 'validation': prefix + '_probe_validation',
                            'probe_validation': prefix + '_probe_validation', 'test': prefix + '_test'})
    return d


def acs(task_name):
    import folktables as ft
    task = getattr(ft, {'acs_income': 'ACSIncome', 'acs_employment': 'ACSEmployment',
                        'acs_coverage': 'ACSPublicCoverage'}[task_name])
    frames = []
    for state, year, domain in [('CA', 2018, 'source'), ('MI', 2018, 'geographic'), ('CA', 2019, 'temporal')]:
        raw = ft.ACSDataSource(survey_year=str(year), horizon='1-Year', survey='person',
                              root_dir=str(BASE / 'raw/acs')).get_data(states=[state], download=True)
        d = task._preprocess(raw).copy()
        # Employment restricted to adult working-age cohort; installed Folktables
        # version has no employment age filter. Record this intentional variant.
        if task_name == 'acs_employment':
            d = d[d.AGEP.between(16, 90)]
        d = d[d.SEX.isin([1, 2])].copy()
        # RELP was replaced by RELSHIPP in 2019. Omit it from ALL years to avoid
        # pretending incompatible categories are the same variable.
        f = d[[c for c in task.features if c not in ['SEX', 'RELP', 'ST']]].copy()
        for c in f:
            if c not in ['AGEP', 'PINCP', 'WKHP']:
                f[c] = f[c].astype('string').fillna('Missing').astype(str)
        f = f.add_prefix('x_')
        f['a'], f['y'] = d.SEX.eq(1).astype(int), task._target_transform(d[task.target]).astype(int)
        f['unit'] = f'{year}_{state}_' + d.SERIALNO.astype(str)
        f['age'] = d.AGEP
        f = f.reset_index(drop=True)
        frames.append(assign_splits(f) if domain == 'source' else target_split(f, domain))
    return pd.concat(frames, ignore_index=True)


def ptb(name):
    from experiments.rq4_ptbxl.prepare import load, CACHE, META
    d = load()
    signals = CACHE / 'signals100.npy'
    audit = json.loads((CACHE / 'waveform_audit.json').read_text())
    assert audit['metadata_sha256'] == sha(META / 'ptbxl_database.csv')
    x = np.load(signals, mmap_mode='r')
    assert x.shape == (len(d), 12, 1000)
    f = pd.DataFrame({'a': d.recorded_sex, 'y': d[name.removeprefix('ptb_').upper()],
                      'unit': d.patient_id.astype(str), 'row_index': d.row_index,
                      'age': d.age, 'split': d.split, 'ecg_id': d.ecg_id})
    # Fold 7/8 provides independent probe validation; fold 9 remains task val;
    # fold 10 remains official test. Patient-disjoint split guaranteed by folds.
    f.loc[d.strat_fold.isin([7, 8]), 'split'] = 'probe_validation'
    return f, str(signals)


def validate(d):
    assert d.row_id.is_unique and d.unit.notna().all()
    assert d.groupby('unit').split.nunique().max() == 1
    assert d.a.isin([0, 1]).all() and d.y.isin([0, 1]).all()
    support = d.groupby(['split', 'a', 'y']).size().unstack(['a', 'y'], fill_value=0)
    assert len(support.columns) == 4 and support.min().min() >= 8, support
    assert {'train', 'validation', 'probe_validation', 'test'} <= set(d.split)
    return support


def prepare(name):
    folder = BASE / 'prepared' / name
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'manifest.json').exists():
        manifest = json.loads((folder / 'manifest.json').read_text())
        assert manifest['cohort_sha256'] == sha(folder / 'cohort.csv')
        assert manifest['preparation_code_sha256'] == sha(__file__), 'Preparation changed; use a fresh version.'
        print('Prepared cache verified', name, flush=True)
        return
    signal_path = None
    if name.startswith('ptb_'):
        d, signal_path = ptb(name)
    elif name.startswith('acs_'):
        d = acs(name)
    else:
        d = {'synthetic': synthetic, 'adult': adult, 'credit_default': credit,
             'bank_marketing': bank, 'compas_race': compas}[name]()
    if name == 'adult':
        d['age'] = d.x_age
    d = d.reset_index(drop=True)
    d['row_id'] = np.arange(len(d))
    validate(d)
    d.to_csv(folder / 'cohort.csv', index=False)
    d.groupby(['split', 'a', 'y']).size().rename('n').reset_index().to_csv(folder / 'support.csv', index=False)
    mapping = {'0': 'Female', '1': 'Male'}
    if name == 'bank_marketing': mapping = {'0': 'Age <40', '1': 'Age >=40'}
    if name == 'compas_race': mapping = {'0': 'Caucasian', '1': 'African-American'}
    if name == 'synthetic': mapping = {'0': 'Abstract group 0', '1': 'Abstract group 1'}
    manifest = dict(dataset=name, rows=len(d), cohort_sha256=sha(folder / 'cohort.csv'),
                    preparation_code_sha256=sha(__file__), mapping=mapping, signal_path=signal_path,
                    features=[c for c in d if c.startswith('x_')],
                    categorical_features=[c for c in d if c.startswith('x_') and not pd.api.types.is_numeric_dtype(d[c])],
                    split_seed=42, survey_weights_used=False,
                    raw_hashes={str(p.relative_to(BASE)): sha(p) for p in (BASE / 'raw').rglob('*') if p.is_file()},
                    split_support=d.groupby('split').size().to_dict())
    (folder / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print('PREPARED', name, manifest['split_support'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', choices=DATASETS, default=DATASETS)
    args = parser.parse_args()
    for name in args.datasets:
        prepare(name)
