"""Integration/regression checks to execute in Google Colab, without EE downloads."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class CanadaDashboardChecks(unittest.TestCase):
    def test_scene_selection_tries_next_candidate(self):
        from ship_map import fetch_observation, HALIFAX_BOUNDS, _RejectedObservation

        candidates = [{'id': 'missing-cloud'}, {'id': 'too-cloudy'}, {'id': 'clear'}]
        with patch('ship_map.ee') as ee, patch(
                'ship_map._fetch_candidate') as fetch:
            ee.ImageCollection.return_value.filterBounds.return_value.filterDate.return_value.filter.return_value.sort.return_value.limit.return_value.getInfo.return_value = {
                'features': candidates}
            fetch.side_effect = [_RejectedObservation('No matching pixel-cloud assessment.'),
                                 _RejectedObservation('Less than 80% clear.'),
                                 ('rgb', 'valid', 'transform', 'crs', {'image_id': 'clear'})]
            *_, metadata = fetch_observation(HALIFAX_BOUNDS, '2025-06-01', '2025-09-01')
            self.assertEqual(metadata['image_id'], 'clear')
            self.assertEqual([item['scene'] for item in metadata['skipped_scenes']], ['missing-cloud', 'too-cloudy'])
            self.assertEqual(fetch.call_count, 3)

    def test_scene_selection_reports_exhausted_candidates(self):
        from ship_map import fetch_observation, HALIFAX_BOUNDS, _RejectedObservation

        with patch('ship_map.ee') as ee, patch(
                'ship_map._fetch_candidate', side_effect=_RejectedObservation('Area coverage below 98%.')):
            ee.ImageCollection.return_value.filterBounds.return_value.filterDate.return_value.filter.return_value.sort.return_value.limit.return_value.getInfo.return_value = {
                'features': [{'id': 'partial'}]}
            with self.assertRaisesRegex(ValueError, r'No usable scene among 1 candidates.*Area coverage'):
                fetch_observation(HALIFAX_BOUNDS, '2025-06-01', '2025-09-01')

    def test_review_gate(self):
        from canadian_validation import checked_annotation
        sample={'imageWidth':128,'imageHeight':128,'annotation_status':'UNANNOTATED',
                'flags':{'review_required':True},'shapes':[]}
        self.assertFalse(checked_annotation(sample))
        sample.update(annotation_status='reviewer_checked',reviewer='test reviewer',reviewed_utc='2026-11-09T00:00:00Z',
                      flags={'review_required':False})
        self.assertFalse(checked_annotation(sample))
        sample['negative_confirmed']=True
        self.assertTrue(checked_annotation(sample))
        sample['annotation_status']='draft_unreviewed'
        self.assertFalse(checked_annotation(sample))

    def test_request_rejection_before_download(self):
        from dashboard_adapter import DetectionRequest,run_detection
        with patch('ship_map.fetch_observation') as fetch:
            with self.assertRaises(ValueError):
                run_detection('.',None,None,DetectionRequest((-140,42,-53,83),'2025-06-01','2025-09-01'))
            with self.assertRaises(ValueError):
                DetectionRequest((-63.58,44.615,-63.545,44.64),'2025-09-01','2025-06-01').validate()
            with self.assertRaises(ValueError):
                DetectionRequest((-63.58,44.615,-63.545,44.64),'2025-06-01','2025-09-01',float('nan')).validate()
            fetch.assert_not_called()

    def test_reference_filter(self):
        from national_data import filter_ports
        rows=[dict(name='Test Harbour',province='ON',subtype='Port'),dict(name='Other',province='BC',subtype='Marina')]
        self.assertEqual(filter_ports(rows,'ON','Port','HARBOUR'),rows[:1])
        self.assertEqual(filter_ports(rows,'ON','Marina'),[])

    def test_geographic_plan_keeps_halifax_together(self):
        from canadian_validation import CASE_PLAN
        groups={}
        for _,_,group,split,_ in CASE_PLAN:
            groups.setdefault(group,set()).add(split)
        self.assertTrue(all(len(s)==1 for s in groups.values()))

    def test_widget_filters_and_stale_result(self):
        from ship_map import build_detection_app
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'data').mkdir()
            (root/'results').mkdir()
            (root/'data/canada_ports_reference.csv').write_text(
                'reference_id,name,province,subtype,longitude,latitude\n1,Test Harbour,ON,Port,-79.86,43.27\n2,Other,BC,Marina,-123.1,49.3\n')
            panel,controls=build_detection_app(root,None,None,{})
            self.assertEqual(controls['map'].zoom,3)
            self.assertEqual(len(controls['national_layer'].data['features']),2)
            controls['province'].value='ON'
            self.assertEqual(len(controls['national_layer'].data['features']),1)
            controls['preset'].value='Iqaluit (Arctic)'
            self.assertAlmostEqual(controls['fields'][1].value,63.725)
            controls['state']['last']=('stale','stale')
            controls['threshold'].value=.65
            self.assertIsNone(controls['state']['last'])
            self.assertTrue(controls['download_button'].disabled)
            controls['ports'].value=False
            self.assertEqual(controls['national_layer'].data['features'],[])


def run_checks(project):
    import io
    stream=io.StringIO()
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(CanadaDashboardChecks))
    report={'execution_environment':'Google Colab','tests':result.testsRun,'failures':len(result.failures),
            'errors':len(result.errors),'passed':result.wasSuccessful(),'log':stream.getvalue(),
            'scope':'Synthetic data, real widget construction and callbacks; no live EE inference or accuracy measurement.'}
    (Path(project)/'results/canada_dashboard_checks.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(stream.getvalue())
    if not result.wasSuccessful():
        raise AssertionError('Dashboard regression checks failed.')
    return report
