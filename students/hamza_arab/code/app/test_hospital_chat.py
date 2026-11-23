from hospital_chat import answer

records=[{'key':'0','name':'Ottawa Hospital General Campus','lat':45.398,'lon':-75.65,'type':'Hospital','er':'','ownership':'','address':'501 Smyth Rd, Ottawa','province':'ON'},
         {'key':'1','name':'Test clinic','lat':45.4,'lon':-75.65,'type':'Clinic','er':'no','ownership':'','address':'','province':'ON'}]

def test_hospital_questions_are_sourced():
    r=answer({'question':'What hospitals are nearby?','lat':45.4,'lon':-75.65,'radius_m':20000},records,key='')
    assert r['mode']=='reference-only' and 'Ottawa Hospital' in r['answer']

def test_select_forbids_invented_er():
    r=answer({'question':'Does it have an emergency department?','lat':45.4,'lon':-75.65,'radius_m':20000,'selected_key':'0'},records,key='')
    assert 'does not verify' in r['answer']

def test_venue_signals_not_exaggerated():
    r=answer({'question':'Is there a helicopter landing pad or parking?','lat':45.4,'lon':-75.65,'radius_m':20000,'selected_key':'0'},records,key='')
    assert 'not been verified' in r['answer']

def test_no_hospitals_not_absence():
    r=answer({'question':'Nearest hospital?','lat':53.0,'lon':-113.0,'radius_m':1000},records,key='')
    assert 'does not prove' in r['answer']

class FakeResp:
    def raise_for_status(self):pass
    def json(self):return {'candidates':[{'content':{'parts':[{'text':'The reference lists Ottawa Hospital General Campus, but has no verified ER field.'}]}}]}
class FakeSession:
    def post(self,*a,**kw):
        assert 'x-goog-api-key' in kw['headers']
        assert 'official_reference_records' in kw['json']['contents'][0]['parts'][0]['text']
        return FakeResp()

def test_optional_llm_success():
    r=answer({'question':'tell me about Ottawa Hospital','lat':45.4,'lon':-75.65,'radius_m':20000},records,key='example',session=FakeSession())
    assert r['mode']=='gemini' and r['llm_enabled']
