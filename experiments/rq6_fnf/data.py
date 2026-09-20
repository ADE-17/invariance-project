"""RQ6 cohorts: UCI Adult (RQ5 cohort reused verbatim), Heritage Health member-years, and five further
tabular fairness benchmarks for the categorical FNF variant: COMPAS, Law School (LSAC), Communities & Crime
(the remaining FNF-paper datasets), Dutch Census and Taiwan credit-card default.

Convention for every dataset: a=1 is the group with the higher natural positive rate (the categorical FNF reference
group, kept unchanged), a=0 the lower-rate group whose positives the shift conditions remove.

Prevalence conditions are explicit, seeded row subsamples applied to the source
splits (train / validation / probe_validation) and, separately, to the natural
test split. Nothing here estimates population quantities; cohorts are record
samples with fixed, unit-disjoint splits.
"""
from pathlib import Path
import argparse
import json
import sys
import zipfile
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.rq5_guarantees.prepare import assign_splits
from experiments.rq5_sweep.data import sha

BASE = Path('/path/to/results/rq6/fnf_v1')
RQ5_ADULT = Path('/path/to/results/rq5/sweep_v2/prepared/adult')
DATASETS = ['adult', 'health', 'compas', 'lawschool', 'crime', 'dutch', 'credit']  # order fixed: shard indices depend on it
CONDITIONS = ['natural', 'equalized', 'shift50', 'shift75']
SOURCE_SPLITS = ['train', 'validation', 'probe_validation']
# Distinct fixed seeds per split so the same condition never reuses one draw.
CONDITION_SEEDS = {'train': 4201, 'validation': 4202, 'probe_validation': 4203, 'test': 4204}

ADULT_CATEGORICAL = ['x_workclass', 'x_education', 'x_marital', 'x_occupation', 'x_relationship', 'x_race']
ADULT_FEATURE_SETS = {'paper6': ADULT_CATEGORICAL, 'paper6_age': ADULT_CATEGORICAL + ['x_age']}
HEALTH_CORE = ['x_no_claims', 'x_no_providers', 'x_no_vendors', 'x_no_pcps', 'x_paydelay_total',
               'x_paydelay_max', 'x_paydelay_min', 'x_los_total', 'x_dsfs_max', 'x_suplos',
               'x_drug_total', 'x_drug_months', 'x_lab_total', 'x_lab_months']
HEALTH_MAPPING = {'0': 'Age at first claim <60', '1': 'Age at first claim >=60'}
ADULT_MAPPING = {'0': 'Female', '1': 'Male'}
RAW = BASE / 'raw'
# Per-dataset quantile bin counts for numeric features used by the categorical variant (edges fitted on train).
COMPAS_FEATURES = ['x_sex', 'x_age_cat', 'x_charge_degree', 'x_priors', 'x_juv_fel', 'x_juv_misd', 'x_juv_other']
LAW_PAPER = ['x_lsat', 'x_ugpa']
LAW_EXTENDED = LAW_PAPER + ['x_fam_inc', 'x_fulltime', 'x_male', 'x_tier']
CRIME_FEATURES = ['x_med_income', 'x_pct_pop_under_pov', 'x_pct_unemployed', 'x_pct_fam2par', 'x_pct_not_hs_grad', 'x_pct_urban']
DUTCH_CORE = ['x_age', 'x_edu_level', 'x_economic_status', 'x_cur_eco_activity', 'x_marital_status', 'x_household_position']
CREDIT_CORE = ['x_education', 'x_marriage', 'x_age', 'x_pay_0', 'x_limit_bal']
CREDIT_EXTENDED = CREDIT_CORE + ['x_pay_2', 'x_bill_amt1', 'x_pay_amt1']
BINNED = {'adult': {'x_age': 4}, 'health': {},
          'compas': {},
          'lawschool': {'x_lsat': 8, 'x_ugpa': 6},
          'crime': {c: 5 for c in CRIME_FEATURES},
          'dutch': {},
          'credit': {'x_age': 5, 'x_limit_bal': 5, 'x_bill_amt1': 5, 'x_pay_amt1': 5}}

LOS_DAYS = {'1 day': 1, '2 days': 2, '3 days': 3, '4 days': 4, '5 days': 5, '6 days': 6,
            '1- 2 weeks': 11, '2- 4 weeks': 21, '4- 8 weeks': 42, '26+ weeks': 180}
DSFS_MONTHS = {'0- 1 month': 1, '1- 2 months': 2, '2- 3 months': 3, '3- 4 months': 4, '4- 5 months': 5,
               '5- 6 months': 6, '6- 7 months': 7, '7- 8 months': 8, '8- 9 months': 9, '9-10 months': 10,
               '10-11 months': 11, '11-12 months': 12}
CHARLSON = {'0': 0, '1-2': 1, '3-4': 2, '5+': 3}
AGE_MIDPOINT = {'0-9': 5, '10-19': 15, '20-29': 25, '30-39': 35, '40-49': 45, '50-59': 55,
                '60-69': 65, '70-79': 75, '80+': 85}


def slug(value):
    return ''.join(ch if ch.isalnum() else '_' for ch in str(value)).strip('_').lower()


def adult():
    manifest = json.loads((RQ5_ADULT / 'manifest.json').read_text())
    assert sha(RQ5_ADULT / 'cohort.csv') == manifest['cohort_sha256'], 'RQ5 Adult cohort changed'
    d = pd.read_csv(RQ5_ADULT / 'cohort.csv', dtype={'unit': str, **{c: str for c in manifest['categorical_features']}})
    for c in manifest['categorical_features']:
        d[c] = d[c].fillna('Missing')
    d.attrs['source'] = dict(rq5_cohort_sha256=manifest['cohort_sha256'], path=str(RQ5_ADULT))
    return d


def health():
    archive = BASE / 'raw/health/data.zip'
    z = zipfile.ZipFile(archive)
    claims = pd.read_csv(z.open('data/Claims.csv'))
    claims['PayDelay'] = claims.PayDelay.replace('162+', '162').astype(int)
    claims['los_days'] = claims.LengthOfStay.map(LOS_DAYS).fillna(0).astype(int)
    claims['dsfs'] = claims.DSFS.map(DSFS_MONTHS)
    claims['charlson'] = claims.CharlsonIndex.astype(str).map(CHARLSON).astype(int)
    for c in ['Specialty', 'PlaceSvc', 'ProcedureGroup']:
        claims[c] = claims[c].fillna('Unknown')
    g = claims.groupby(['MemberID', 'Year'])
    agg = pd.DataFrame({'x_no_claims': g.size(), 'x_no_providers': g.ProviderID.nunique(),
                        'x_no_vendors': g.Vendor.nunique(), 'x_no_pcps': g.PCP.nunique(),
                        'x_paydelay_total': g.PayDelay.sum(), 'x_paydelay_max': g.PayDelay.max(),
                        'x_paydelay_min': g.PayDelay.min(), 'x_los_total': g.los_days.sum(),
                        'x_dsfs_max': g.dsfs.max().fillna(0), 'x_suplos': g.SupLOS.sum(),
                        'max_charlson': g.charlson.max()})
    for col, prefix in [('Specialty', 'spec'), ('PlaceSvc', 'place'), ('ProcedureGroup', 'proc')]:
        counts = pd.crosstab([claims.MemberID, claims.Year], claims[col])
        counts.columns = [f'x_{prefix}_{slug(c)}' for c in counts.columns]
        agg = agg.join(counts, how='left')
    drugs = pd.read_csv(z.open('data/DrugCount.csv'))
    drugs['DrugCount'] = drugs.DrugCount.astype(str).str.replace('+', '', regex=False).astype(int)
    drugs = drugs.groupby(['MemberID', 'Year']).DrugCount.agg(x_drug_total='sum', x_drug_months='count')
    labs = pd.read_csv(z.open('data/LabCount.csv'))
    labs['LabCount'] = labs.LabCount.astype(str).str.replace('+', '', regex=False).astype(int)
    labs = labs.groupby(['MemberID', 'Year']).LabCount.agg(x_lab_total='sum', x_lab_months='count')
    agg = agg.join(drugs, how='left').join(labs, how='left').fillna(0).reset_index()
    members = pd.read_csv(z.open('data/Members.csv'))
    members = members[members.AgeAtFirstClaim.notna()].copy()
    d = agg.merge(members, on='MemberID', how='inner')
    d['a'] = d.AgeAtFirstClaim.isin(['60-69', '70-79', '80+']).astype(int)
    d['y'] = (d.max_charlson > 0).astype(int)
    d['unit'] = d.MemberID.astype(str)
    d['age'] = d.AgeAtFirstClaim.map(AGE_MIDPOINT).astype(int)
    d['audit_sex'] = d.Sex.fillna('Unknown')
    d['audit_year'] = d.Year
    features = [c for c in d if c.startswith('x_')]
    out = d[features + ['a', 'y', 'unit', 'age', 'audit_sex', 'audit_year']].copy()
    for c in features:
        out[c] = out[c].astype(float)
    # The label is derived from claim-level Charlson codes; Charlson itself is never a feature.
    assert 'max_charlson' not in out and not any('charlson' in c for c in features)
    out.attrs['source'] = dict(archive_sha256=sha(archive), member_years=len(agg), members_with_age=len(members))
    return assign_splits(out.reset_index(drop=True))

def _rows_as_units(d):
    d = d.reset_index(drop=True)
    d['unit'] = np.arange(len(d)).astype(str)
    return assign_splits(d)


def compas():
    """ProPublica two-year recidivism, standard filter, African-American vs Caucasian (FNF paper dataset)."""
    path = RAW / 'compas' / 'compas-scores-two-years.csv'
    c = pd.read_csv(path)
    c = c[(c.days_b_screening_arrest <= 30) & (c.days_b_screening_arrest >= -30) & (c.is_recid != -1)
          & (c.c_charge_degree != 'O') & (c.score_text != 'N/A') & c.race.isin(['African-American', 'Caucasian'])]
    priors = pd.cut(c.priors_count, [-1, 0, 1, 3, 6, np.inf], labels=['0', '1', '2-3', '4-6', '7+']).astype(str)
    out = pd.DataFrame(dict(x_sex=c.sex.astype(str), x_age_cat=c.age_cat.astype(str), x_charge_degree=c.c_charge_degree.astype(str),
                            x_priors=priors, x_juv_fel=(c.juv_fel_count > 0).map({True: '1+', False: '0'}),
                            x_juv_misd=(c.juv_misd_count > 0).map({True: '1+', False: '0'}),
                            x_juv_other=(c.juv_other_count > 0).map({True: '1+', False: '0'}),
                            a=(c.race == 'African-American').astype(int), y=c.two_year_recid.astype(int)))
    out.attrs['source'] = dict(file_sha256=sha(path), url='https://github.com/propublica/compas-analysis', rows_after_filter=len(out))
    return _rows_as_units(out)


def lawschool():
    """LSAC bar passage (Wightman), White vs Non-White (FNF paper dataset)."""
    path = RAW / 'lawschool' / 'law_school_clean.csv'
    l = pd.read_csv(path).dropna(subset=['lsat', 'ugpa', 'race', 'pass_bar'])
    out = pd.DataFrame(dict(x_lsat=l.lsat.astype(float), x_ugpa=l.ugpa.astype(float), x_fam_inc=l.fam_inc.astype(int).astype(str),
                            x_fulltime=l.fulltime.astype(int).astype(str), x_male=l.male.astype(int).astype(str), x_tier=l.tier.astype(int).astype(str),
                            a=(l.race == 'White').astype(int), y=l.pass_bar.astype(int)))
    out.attrs['source'] = dict(file_sha256=sha(path), url='https://github.com/tailequy/fairness_dataset (law_school_clean.csv)')
    return _rows_as_units(out)


def crime():
    """UCI Communities & Crime: violent crime rate above the median; a=1 communities with racepctblack >= median (FNF paper dataset)."""
    import re
    folder = RAW / 'crime'
    names = [m.group(1) for m in re.finditer(r'@attribute (\S+)', (folder / 'communities.names').read_text())]
    cr = pd.read_csv(folder / 'communities.data', header=None, na_values='?', names=names)
    cols = dict(x_med_income='medIncome', x_pct_pop_under_pov='PctPopUnderPov', x_pct_unemployed='PctUnemployed',
                x_pct_fam2par='PctFam2Par', x_pct_not_hs_grad='PctNotHSGrad', x_pct_urban='pctUrban')
    out = pd.DataFrame({k: cr[v].astype(float) for k, v in cols.items()})
    out['a'] = (cr.racepctblack >= cr.racepctblack.median()).astype(int)
    out['y'] = (cr.ViolentCrimesPerPop > cr.ViolentCrimesPerPop.median()).astype(int)
    out['audit_state'] = cr.state.astype(str)
    out.attrs['source'] = dict(file_sha256=sha(folder / 'communities.data'), url='https://archive.ics.uci.edu/dataset/183/communities+and+crime',
                               label_threshold=float(cr.ViolentCrimesPerPop.median()), attribute_threshold=float(cr.racepctblack.median()))
    return _rows_as_units(out)


def dutch():
    """Dutch Census 2001 (Le Quy et al. mirror): high-level occupation, female (a=0) vs male (a=1)."""
    path = RAW / 'dutch' / 'dutch.csv'
    d = pd.read_csv(path)
    cols = ['age', 'edu_level', 'economic_status', 'cur_eco_activity', 'marital_status', 'household_position', 'household_size',
            'citizenship', 'country_birth', 'prev_residence_place']
    out = pd.DataFrame({f'x_{c}': d[c].astype(int).astype(str) for c in cols})
    out['a'] = (d.sex == 'male').astype(int)
    out['y'] = d.occupation.astype(int)
    out.attrs['source'] = dict(file_sha256=sha(path), url='https://github.com/tailequy/fairness_dataset (dutch.csv)')
    return _rows_as_units(out)


def credit():
    """UCI default of credit card clients (Taiwan): default next month, female (a=0) vs male (a=1)."""
    path = RAW / 'credit' / 'credit-card-clients.csv'
    r = pd.read_csv(path)
    edu = r.EDUCATION.where(r.EDUCATION.isin([1, 2, 3]), 4).astype(int).astype(str)
    mar = r.MARRIAGE.where(r.MARRIAGE.isin([1, 2]), 3).astype(int).astype(str)
    pay = lambda s: s.clip(-1, 2).astype(int).astype(str)  # -1 or less: paid duly; 0 revolving; 1; 2+ months delay
    out = pd.DataFrame(dict(x_education=edu, x_marriage=mar, x_age=r.AGE.astype(float), x_pay_0=pay(r.PAY_0), x_pay_2=pay(r.PAY_2),
                            x_limit_bal=r.LIMIT_BAL.astype(float), x_bill_amt1=r.BILL_AMT1.astype(float), x_pay_amt1=r.PAY_AMT1.astype(float),
                            a=(r.SEX == 1).astype(int), y=r['default payment'].astype(int)))
    out.attrs['source'] = dict(file_sha256=sha(path), url='https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients')
    return _rows_as_units(out)


META = {
    'adult': dict(builder=adult, mapping=ADULT_MAPPING, label='income >50K', unit='hashed feature row',
                  feature_sets=lambda d: ADULT_FEATURE_SETS),
    'health': dict(builder=health, mapping=HEALTH_MAPPING, label='max claim-level Charlson index >0 within member-year',
                   unit='MemberID (member-years share a unit)',
                   feature_sets=lambda d: {'core': HEALTH_CORE, 'extended': HEALTH_CORE + [c for c in d if c.startswith(('x_spec_', 'x_place_', 'x_proc_'))]}),
    'compas': dict(builder=compas, mapping={'0': 'Caucasian', '1': 'African-American'}, label='two-year recidivism', unit='defendant row',
                   feature_sets=lambda d: {'paper7': COMPAS_FEATURES, 'paper7_nosex': [c for c in COMPAS_FEATURES if c != 'x_sex']}),
    'lawschool': dict(builder=lawschool, mapping={'0': 'Non-White', '1': 'White'}, label='passed the bar', unit='student row',
                      feature_sets=lambda d: {'paper2': LAW_PAPER, 'extended6': LAW_EXTENDED}),
    'crime': dict(builder=crime, mapping={'0': 'racepctblack below median', '1': 'racepctblack at or above median'},
                  label='violent crimes per population above median', unit='community row',
                  feature_sets=lambda d: {'socio6': CRIME_FEATURES, 'socio4': CRIME_FEATURES[:4]}),
    'dutch': dict(builder=dutch, mapping={'0': 'Female', '1': 'Male'}, label='high-level occupation', unit='census record row',
                  feature_sets=lambda d: {'core6': DUTCH_CORE, 'core6_hh': DUTCH_CORE + ['x_household_size']}),
    'credit': dict(builder=credit, mapping={'0': 'Female', '1': 'Male'}, label='default payment next month', unit='client row',
                   feature_sets=lambda d: {'core5': CREDIT_CORE, 'extended8': CREDIT_EXTENDED}),
}


def apply_condition(d, condition, split):
    """Seeded subsample implementing one prevalence condition on one split."""
    seed = CONDITION_SEEDS[split]
    if condition == 'natural':
        return d
    if condition.startswith('shift'):
        frac = int(condition.removeprefix('shift')) / 100
        drop = d[(d.a == 0) & (d.y == 1)].sample(frac=frac, random_state=seed).index
        return d.drop(drop)
    if condition == 'equalized':
        rates = d.groupby('a').y.mean()
        high, low = int(rates.idxmax()), float(rates.min())
        pos = d[(d.a == high) & (d.y == 1)]
        neg = int(((d.a == high) & (d.y == 0)).sum())
        keep = int(round(low * neg / (1 - low)))
        drop = pos.sample(n=len(pos) - keep, random_state=seed).index
        return d.drop(drop)
    raise ValueError(condition)


def condition_manifest(d):
    rows = []
    for condition in CONDITIONS:
        for split in SOURCE_SPLITS + ['test']:
            sub = apply_condition(d[d.split == split], condition, split)
            rows.append(dict(condition=condition, split=split, n=len(sub),
                             group0_prevalence=float(sub[sub.a == 0].y.mean()),
                             group1_prevalence=float(sub[sub.a == 1].y.mean()),
                             group0_n=int((sub.a == 0).sum()), group1_n=int((sub.a == 1).sum())))
    return rows


def validate(d):
    assert d.row_id.is_unique and d.unit.notna().all()
    assert d.groupby('unit').split.nunique().max() == 1
    assert d.a.isin([0, 1]).all() and d.y.isin([0, 1]).all()
    support = d.groupby(['split', 'a', 'y']).size().unstack(['a', 'y'], fill_value=0)
    assert len(support.columns) == 4 and support.min().min() >= 8, support
    assert {'train', 'validation', 'probe_validation', 'test'} <= set(d.split)


def feature_sets(name, d):
    return META[name]['feature_sets'](d)


def load(name):
    folder = BASE / 'prepared' / name
    manifest = json.loads((folder / 'manifest.json').read_text())
    assert manifest['cohort_sha256'] == sha(folder / 'cohort.csv')
    d = pd.read_csv(folder / 'cohort.csv', dtype={'unit': str, **{c: str for c in manifest['categorical_features']}})
    validate(d)
    return d, manifest


def prepare(name):
    folder = BASE / 'prepared' / name
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'manifest.json').exists():
        manifest = json.loads((folder / 'manifest.json').read_text())
        assert manifest['cohort_sha256'] == sha(folder / 'cohort.csv')
        if manifest['preparation_code_sha256'] != sha(__file__):
            print('Prepared cache verified by cohort hash; preparation code changed since (new datasets added)', name, flush=True)
        else:
            print('Prepared cache verified', name, flush=True)
        return
    d = META[name]['builder']()
    source = d.attrs.get('source', {})
    d = d.reset_index(drop=True)
    d['row_id'] = np.arange(len(d))
    validate(d)
    d.to_csv(folder / 'cohort.csv', index=False)
    d.groupby(['split', 'a', 'y']).size().rename('n').reset_index().to_csv(folder / 'support.csv', index=False)
    conditions = condition_manifest(d)
    pd.DataFrame(conditions).to_csv(folder / 'conditions.csv', index=False)
    categorical = [c for c in d if c.startswith('x_') and not pd.api.types.is_numeric_dtype(d[c])]
    manifest = dict(dataset=name, rows=len(d), cohort_sha256=sha(folder / 'cohort.csv'),
                    preparation_code_sha256=sha(__file__), mapping=META[name]['mapping'],
                    features=[c for c in d if c.startswith('x_')], categorical_features=categorical,
                    binned_features=BINNED[name],
                    feature_sets=feature_sets(name, d), conditions=CONDITIONS, condition_seeds=CONDITION_SEEDS,
                    label=META[name]['label'], unit=META[name]['unit'],
                    split_seed=42, source=source, split_support=d.groupby('split').size().to_dict(),
                    natural_group_prevalence=d.groupby('a').y.mean().to_dict())
    (folder / 'manifest.json').write_text(json.dumps(manifest, indent=2, default=str))
    print('PREPARED', name, manifest['split_support'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', choices=DATASETS, default=DATASETS)
    args = parser.parse_args()
    for dataset in args.datasets:
        prepare(dataset)
