from pathlib import Path
import json
import pandas as pd
from PIL import Image
from hospital_review import (colab_save_review_result, colab_review_status_result,
                             review_saved_status, build_labeler_ui)


def test_json_callback_and_saved_manifest(tmp_path):
    images=tmp_path/'review'/'images';images.mkdir(parents=True)
    Image.new('RGB',(100,100),'white').save(images/'abc.png')
    pd.DataFrame([{'tile_id':'abc','name':'Test','candidate_class':'hospital_site_UNVERIFIED',
      'image':'review/images/abc.png','annotation':'review/annotations/abc.json',
      'review_status':'UNREVIEWED','reviewed_label':'UNKNOWN','reviewer':'','source_checked':'NO'}]).to_csv(tmp_path/'review_manifest.csv',index=False)
    payload={'tile_id':'abc','reviewer':'Reviewer','category':'HOSPITAL_PRESENT',
             'boxes':[[2,4,60,70]],'confirmed':True}
    response=colab_save_review_result(tmp_path,payload)
    assert response.data['ok'] is True and response.data['remaining_unreviewed']==0
    assert colab_review_status_result(tmp_path,'abc').data['saved'] is True
    assert (tmp_path/'review/annotations/abc.json').is_file()


def test_errors_are_structured_json_not_silent(tmp_path):
    images=tmp_path/'review'/'images';images.mkdir(parents=True)
    Image.new('RGB',(100,100),'white').save(images/'abc.png')
    pd.DataFrame([{'tile_id':'abc','name':'Test','candidate_class':'hospital_site_UNVERIFIED',
      'image':'review/images/abc.png','annotation':'review/annotations/abc.json',
      'review_status':'UNREVIEWED','reviewed_label':'UNKNOWN','reviewer':'','source_checked':'NO'}]).to_csv(tmp_path/'review_manifest.csv',index=False)
    payload={'tile_id':'abc','reviewer':'Reviewer','category':'HOSPITAL_PRESENT',
             'boxes':[[2,4,60,70]],'confirmed':False}
    response=colab_save_review_result(tmp_path,payload)
    assert response.data['ok'] is False and 'confirm' in response.data['error']
    assert colab_review_status_result(tmp_path,'abc').data['saved'] is False
    assert review_saved_status(tmp_path,'abc')['remaining_unreviewed']==1


def test_browser_does_not_claim_saved_without_manifest(tmp_path):
    folder=tmp_path/'demo';folder.mkdir()
    im=folder/'img.png';Image.new('RGB',(64,64),(12,40,100)).save(im)
    markup,script=build_labeler_ui('abc','Test','candidate',im)
    assert 'SAVED TO GOOGLE DRIVE.' in script
    assert 'oss4008_hospital_ai.review_status' in script
    assert "get('save')" in script
    assert 'Saved!' not in markup
