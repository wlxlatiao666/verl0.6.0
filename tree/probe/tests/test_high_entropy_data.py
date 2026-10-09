import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import build_high_entropy_data as build
from common import atomic_json, read_json


def test_positions_high_entropy_spread_and_reproducible():
    values=[0.]*10+[1.]*190
    positions=build.select_positions(values,10,.8,8,16,42)
    assert len(positions)==8
    assert positions==build.select_positions(values,10,.8,8,16,42)
    assert all(t>=10 and values[t]>=.8 for t in positions)
    assert all(b-a>=16 for a,b in zip(positions,positions[1:]))
    assert positions[0]<80 and positions[-1]>130


def test_sparse_and_empty_positions_never_pad():
    assert build.select_positions([0.]*100,10,.8,8,16,42)==[]
    values=[0.]*100
    values[20]=values[21]=values[90]=1.
    selected=build.select_positions(values,10,.8,8,16,42)
    assert len(selected)==2 and selected[-1]==90
    assert selected[0] in (20,21)
    assert build.select_positions([],10,.8,8,16,42)==[]


def test_calibration_never_uses_heldout_entropy(tmp_path,monkeypatch):
    questions=[dict(id='a',split='train'),dict(id='b',split='val'),dict(id='c',split='test')]
    cfg=dict(min_prefix=1,entropy_quantile=.8)
    for q in questions:
        atomic_json(tmp_path/'entropy'/f"{q['id']}.json",dict(records=[dict(entropy=[100.,0.,1.,2.,3.,4.])]))
    monkeypatch.setattr(build,'load_run',lambda _: (tmp_path,cfg,questions))
    args=SimpleNamespace(work_dir=str(tmp_path))
    build.calibrate(args)
    original=read_json(tmp_path/'entropy_calibration.json')
    assert abs(original['threshold']-3.2)<1e-8
    assert original['eligible_training_tokens']==5
    for q in questions[1:]:
        atomic_json(tmp_path/'entropy'/f"{q['id']}.json",dict(records=[dict(entropy=[999.]*100)]))
    build.calibrate(args)
    assert read_json(tmp_path/'entropy_calibration.json')==original
