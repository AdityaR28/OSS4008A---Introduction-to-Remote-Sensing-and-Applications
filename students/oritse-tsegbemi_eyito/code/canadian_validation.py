"""Canadian review eligibility and locked geographic split plan. Run in Colab."""
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

# Assign regions BEFORE new label review or model experiments. Both Halifax
# seasons stay together. Aerial imagery is a distinct domain, never mixed into S2.
CASE_PLAN = [
    ('halifax_summer','Atlantic','Halifax','train','sentinel2_l1c_10m'),
    ('halifax_autumn','Atlantic','Halifax','train','sentinel2_l1c_10m'),
    ('vancouver','Pacific','Vancouver','val','sentinel2_l1c_10m'),
    ('montreal','St Lawrence','Montreal','test','sentinel2_l1c_10m'),
    ('hamilton','Great Lakes','Hamilton','train','ontario_aerial'),
    ('toronto','Great Lakes','Toronto','val','ontario_aerial'),
    ('kingston','Great Lakes','Kingston','test','ontario_aerial'),
    ('churchill','Hudson Bay','Churchill','test','sentinel2_l1c_10m'),
    ('iqaluit','Arctic','Iqaluit','test','sentinel2_l1c_10m'),
    ('prince_rupert','Pacific North','Prince Rupert','train','sentinel2_l1c_10m'),
    ('st_johns','Newfoundland','St Johns','val','sentinel2_l1c_10m')]


def checked_annotation(label):
    """A draft/empty template cannot become a negative example accidentally."""
    if label.get('annotation_status') != 'reviewer_checked':
        return False
    if label.get('flags',{}).get('review_required',True):
        return False
    if not label.get('reviewer','').strip() or not label.get('reviewed_utc'):
        return False
    width,height=label.get('imageWidth',0),label.get('imageHeight',0)
    if not label.get('shapes') and not label.get('negative_confirmed',False):
        return False
    for shape in label.get('shapes',[]):
        if shape.get('label')!='ship' or shape.get('shape_type')!='rectangle':
            return False
        try:
            (x1,y1),(x2,y2)=shape['points']
            if not (0<=x1<x2<=width and 0<=y1<y2<=height):
                return False
        except (KeyError,TypeError,ValueError):
            return False
    return width>0 and height>0


def build_review_manifest(project):
    project=Path(project)
    rows=[]
    for case,region,group,split,domain in CASE_PLAN:
        folder='aerial_review' if domain=='ontario_aerial' else 'canadian_review'
        annotation=project/f'data/{folder}/{case}.json'
        image=annotation.with_suffix('.png')
        label=json.loads(annotation.read_text()) if annotation.exists() else {}
        eligible=image.exists() and checked_annotation(label)
        rows.append(dict(case=case,region=region,geographic_group=group,split=split,domain=domain,
                         image_path=str(image.relative_to(project)),annotation_path=str(annotation.relative_to(project)),
                         review_status=label.get('annotation_status','not_acquired'),eligible=eligible,
                         boxes=len(label.get('shapes',[])) if eligible else '',
                         image_sha256=hashlib.sha256(image.read_bytes()).hexdigest() if image.exists() else '',
                         source_scene=label.get('source',{}).get('image_id','')))
    # Fail closed if identical imagery or scene is assigned to different splits.
    for key in ['geographic_group','image_sha256','source_scene']:
        assigned={}
        for row in rows:
            value=row[key]
            if value and value in assigned and assigned[value]!=row['split']:
                raise ValueError('Canadian split leakage: '+key)
            if value:
                assigned[value]=row['split']
    destination=project/'data/canadian_review_manifest.csv'
    with destination.open('w',newline='',encoding='utf-8') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report={'generated_utc':datetime.now(timezone.utc).isoformat(),
            'eligible_images':sum(r['eligible'] for r in rows),
            'planned_cases':len(rows),'canadian_accuracy_measured':False,
            'status':'Review queue only; do not claim representative national accuracy.',
            'test_policy':'Previously inspected predictions make existing cases exploratory. Acquire new scenes for final independent testing.',
            'requirements':['Freeze acquisition IDs and geographic groups before training.',
                            'Add independent scenes, seasons and negatives in every region; a few harbours are insufficient.',
                            'Train only on train; select epochs/thresholds only on val; unlock new test scenes once.',
                            'Do not infer exact acquisition dates from aerial mosaic collection years.',
                            'Use separate compatible training and evaluation for aerial and 10 m imagery.']}
    (project/'results/canadian_review_readiness.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return rows,report
