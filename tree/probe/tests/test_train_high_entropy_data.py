import sys
from pathlib import Path

import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import train_high_entropy_data as train
from common import atomic_json,file_digest,read_json


def test_fit_uses_frozen_entropy_threshold_and_all_eligible_rows(tmp_path,monkeypatch):
    root=tmp_path/'data';out=tmp_path/'fit'
    questions=[]
    torch.manual_seed(42)
    for split,n in [('train',6),('val',4),('test',4)]:
        q=dict(id=split,split=split);questions.append(q)
        f=root/'features'/f'{split}.json';f.parent.mkdir(parents=True,exist_ok=True)
        torch.save(torch.randn(n,8),f.with_suffix('.pt'))
        records=[];labels=[]
        for i in range(n):
            records.append(dict(id=f'{split}:{i}',trajectory=0,entropy=3.01+i*.1))
            labels.append(dict(id=f'{split}:{i}',q=[1.,0.,0.,0.] if i%2 else [0.]*4,
                utility_raw=.05 if i%2 else 0.,outcomes=[[1]*4,[0]*4,[0]*4,[0]*4] if i%2 else [[0]*4]*4))
        atomic_json(f,dict(records=records,tensor_sha256=file_digest(f.with_suffix('.pt'))))
        atomic_json(root/'labels/main'/f'{split}.json',dict(records=labels,feature_sha256=file_digest(f)))
    atomic_json(root/'questions.json',questions)
    atomic_json(root/'entropy_calibration.json',dict(threshold=3.,quantile=.8))
    monkeypatch.setattr(sys,'argv',['train','--work-dir',str(root),'--output-dir',str(out),'--pca-dim','2'])
    train.main()
    assert read_json(out/'screening.json')['splits']['train']['eligible']==6
    assert read_json(out/'validation.json')['fit_counts']['training_positions']==6
    assert read_json(out/'test.json')['eligible_positions']==4
    ckpt=torch.load(out/'A.pt',weights_only=True)
    assert ckpt['entropy_threshold']==3.
    assert ckpt['conditional_high_entropy']
