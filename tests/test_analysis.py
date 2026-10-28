from pathlib import Path
import hashlib
import numpy as np
from ouro_boundary_pilot.artifacts import read_raw, SHA256, project_root
from ouro_boundary_pilot.statistics import paired_question_effects, calculate, bootstrap_indices


def test_inputs_and_provenance():
    old, new = read_raw()
    assert len(old['row_ids']) == 30
    assert len(set(old['row_ids'])) == 30
    assert len(SHA256) == 2
    assert project_root().joinpath('results', 'raw').exists()
    assert (new['margins'].shape) == (30,2,2,17,3,2)


def test_exact_effect_reconstruction():
    old, new = read_raw()
    assert np.array_equal(paired_question_effects(old['margins'], old['alphas']), old['paired_effects'])
    assert np.array_equal(paired_question_effects(new['margins'], float(new['alpha'])), new['paired_effects'])


def test_statistics_reproducible_and_not_equivalent():
    old, new = read_raw()
    stats = calculate(old, new)
    assert len(stats['rows']) == 8
    assert len(stats['precision_discrepancies']) == 4
    assert all(not row['equivalence_within_plus_minus_0p005_established'] for row in stats['precision_discrepancies'])
    assert np.array_equal(bootstrap_indices(), bootstrap_indices())
    lookup = {(x['direction'],x['mode']):x for x in stats['rows']}
    assert np.isclose(lookup[('seed123_feature2437','Windows FP16')]['mean_T'], -0.039126078287760414, atol=1e-12)
    assert np.isclose(lookup[('seed42_feature2913','DGX FP32 realized')]['mean_T'], -0.03951944410800934, atol=1e-12)


def test_invalid_margins_rejected():
    old, new = read_raw()
    m = new['margins'].copy(); m[0,0,0,0,0,0] = np.nan
    try: paired_question_effects(m, 0.08)
    except ValueError: pass
    else: raise AssertionError('invalid margin accepted')
