"""Regression: a within-X-m request must never include hospitals outside X metres."""
from hospital_chat import answer, distance_m

PARLIAMENT=(45.4247, -75.6994)
RECORDS=[
    {'key':'1','name':'Close hospital','type':'Hospital','lat':45.4260,'lon':-75.6994,'address':'Near test marker'},
    {'key':'2','name':'Outside 500m hospital','type':'Hospital','lat':45.4320,'lon':-75.6994,'address':'Further marker'},
    {'key':'3','name':'Clinic','type':'Clinic','lat':45.4250,'lon':-75.6994,'address':'Clinic'},
]

def _ask(question):
    return answer({'question':question,'lat':45.4215,'lon':-75.6972,'radius_m':25000},RECORDS)

def test_explicit_500_m_filters_results():
    res=_ask('I am looking for a hospital within 500 meters of Parliament Hill')
    assert res['ok'] and [x['key'] for x in res['nav_results']]==['1']
    assert '0.5 km' in res['answer']

def test_explicit_half_km_decimal():
    res=_ask('Hospital within 0.5 km of the Parliament Hill')
    assert [x['key'] for x in res['nav_results']]==['1']

def test_explicit_one_kilometer():
    res=_ask('Hospital within one kilometers of the Parliament Hill')
    assert [x['key'] for x in res['nav_results']]==['1','2']

def test_no_matches_must_not_return_nearest_outside_radius():
    res=_ask('Hospital within 25 meters of Parliament Hill')
    assert not res['nav_results'] and 'No listed hospital reference within' in res['answer']

def test_curated_place_coordinates_are_not_hospitals():
    assert 100<distance_m(*PARLIAMENT,RECORDS[0]['lat'],RECORDS[0]['lon'])<500
