from pathlib import Path
import json
import pandas as pd
from PIL import Image
import pytest

from hospital_review import save_annotation_review


def make_review(root):
    (root/'review'/'images').mkdir(parents=True)
    Image.new('RGB',(100,100),'white').save(root/'review/images/abc.png')
    pd.DataFrame([{'tile_id':'abc','name':'test','candidate_class':'site',
    'image':'review/images/abc.png','annotation':'review/annotations/abc.json',
    'review_status':'UNREVIEWED','reviewed_label':'UNKNOWN','reviewer':'','source_checked':'NO'}]).to_csv(root/'review_manifest.csv',index=False)


def test_approved_labels_only_after_human_confirmation(tmp_path):
    make_review(tmp_path)
    with pytest.raises(ValueError,match='confirm'):
        save_annotation_review(tmp_path,{'tile_id':'abc','reviewer':'Reviewer',
            'category':'HOSPITAL_PRESENT','boxes':[[1,2,50,60]],'confirmed':False})
    assert not (tmp_path/'review'/'annotations'/'abc.json').exists()
    result=save_annotation_review(tmp_path,{'tile_id':'abc','reviewer':'Reviewer',
        'category':'HOSPITAL_PRESENT','boxes':[[1,2,50,60]],'confirmed':True})
    assert result['ok'] and result['remaining_unreviewed']==0
    doc=json.loads((tmp_path/'review'/'annotations'/'abc.json').read_text())
    assert doc['flags']['verified'] is True and len(doc['shapes'])==1
    df=pd.read_csv(tmp_path/'review_manifest.csv',keep_default_na=False)
    assert df.iloc[0].review_status=='APPROVED'
    assert df.iloc[0].source_checked=='YES'


def test_negative_rejects_drawn_boxes(tmp_path):
    make_review(tmp_path)
    with pytest.raises(ValueError,match='cannot contain'):
        save_annotation_review(tmp_path,{'tile_id':'abc','reviewer':'Reviewer',
            'category':'NO_HOSPITAL','boxes':[[1,2,50,60]],'confirmed':True})
