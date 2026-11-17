"""Colab image-and-box review; only an explicit reviewer can mark labels checked."""
import json
from pathlib import Path
import ipywidgets as widgets
from IPython.display import display, clear_output
from PIL import Image, ImageDraw


def annotation_review(project):
    root = Path(project)
    choices = sorted((root/'data/canadian_review').glob('*.png')) + sorted((root/'data/aerial_review').glob('*.png'))
    pilot_image = root/'data/canadian_pilot/saint_john_pilot.png'
    if pilot_image.with_suffix('.json').is_file() and pilot_image.is_file():
        choices.insert(0, pilot_image)
    if not choices:
        raise ValueError('Run the Canadian imagery cells first.')
    select = widgets.Dropdown(options=[(p.stem,str(p)) for p in choices],description='Image:')
    coords = [widgets.IntText(value=0,description=s,layout=widgets.Layout(width='170px')) for s in ['x1','y1','x2','y2']]
    add=widgets.Button(description='Add ship box')
    undo=widgets.Button(description='Undo last box')
    save=widgets.Button(description='Save checked labels',button_style='success')
    draft=widgets.Button(description='Save draft (unreviewed)')
    negative=widgets.Checkbox(description='Reviewed image contains no ships',value=False,indent=False)
    notes=widgets.Textarea(description='Review notes:',placeholder='Ambiguities, cloud/ice, dry-stored boats or resolution limitations')
    reviewer=widgets.Text(description='Reviewer:')
    checked=widgets.Checkbox(description='All visible ships checked; ambiguous objects excluded',value=False,indent=False)
    blind=widgets.Checkbox(description='I drew these boxes before seeing model predictions',value=False,indent=False)
    out=widgets.Output()
    state={'shapes':[]}

    def show(*_):
        with out:
            clear_output(wait=True)
            image=Image.open(select.value).convert('RGB')
            draw=ImageDraw.Draw(image)
            for shape in state['shapes']:
                draw.rectangle(tuple(shape['points'][0]+shape['points'][1]),outline='red',width=3)
            print('Coordinates use the original image grid:',image.size,'| boxes:',len(state['shapes']))
            image.thumbnail((1000,750))
            display(image)

    def change(_):
        label=Path(select.value).with_suffix('.json')
        draft_path=Path(select.value).with_suffix('.draft.json')
        loaded=json.loads((draft_path if draft_path.exists() else label).read_text()) if label.exists() else {}
        state['shapes']=loaded.get('shapes',[])
        checked.value=False
        negative.value=False
        blind.value=False
        blind.layout.display='' if Path(select.value) == pilot_image else 'none'
        notes.value=loaded.get('review_notes','')
        show()

    def add_box(_):
        x1,y1,x2,y2=[w.value for w in coords]
        width,height=Image.open(select.value).size
        if not (0<=x1<x2<=width and 0<=y1<y2<=height):
            with out:
                print('Box must have positive size and remain inside the original image grid.')
            return
        state['shapes'].append(dict(label='ship',points=[[x1,y1],[x2,y2]],group_id=None,shape_type='rectangle',flags={}))
        checked.value=False
        negative.value=False
        show()

    def remove(_):
        if state['shapes']:
            state['shapes'].pop()
        checked.value=False
        show()

    def save_labels(_):
        if not reviewer.value.strip() or not checked.value:
            with out:
                print('Enter the reviewer name and confirm the image has been checked before saving.')
            return
        if not state['shapes'] and not negative.value:
            with out:
                print('For an empty checked label, explicitly confirm that the reviewed image contains no ships.')
            return
        from datetime import datetime,timezone
        path=Path(select.value).with_suffix('.json')
        label=json.loads(path.read_text())
        if Path(select.value) == pilot_image:
            prior_result = (root/'results/saint_john_pilot_metrics.json').exists()
            label['reviewed_before_predictions'] = bool(blind.value and not prior_result)
        history=path.parent/'review_history'
        history.mkdir(exist_ok=True)
        (history/(path.stem+'_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')+'.json')).write_text(path.read_text(),encoding='utf-8')
        label.update(shapes=state['shapes'],flags={**label.get('flags',{}),'review_required':False},annotation_status='reviewer_checked',
                     review_notes=notes.value,negative_confirmed=negative.value if not state['shapes'] else False,
                     reviewer=reviewer.value.strip(),reviewed_utc=datetime.now(timezone.utc).isoformat())
        path.write_text(json.dumps(label,indent=2)+'\n',encoding='utf-8')
        with out:
            print('Saved:',path.name,'| checked ship boxes:',len(state['shapes']))

    def save_draft(_):
        path=Path(select.value).with_suffix('.draft.json')
        original=json.loads(Path(select.value).with_suffix('.json').read_text())
        original.update(shapes=state['shapes'],annotation_status='draft_unreviewed',
                        flags={**original.get('flags',{}),'review_required':True},review_notes=notes.value)
        path.write_text(json.dumps(original,indent=2)+'\n',encoding='utf-8')
        with out:
            print('Saved draft separately; excluded from training and evaluation:',path.name)

    select.observe(change,names='value')
    add.on_click(add_box)
    undo.on_click(remove)
    save.on_click(save_labels)
    draft.on_click(save_draft)
    change(None)
    return widgets.VBox([widgets.HTML('<b>Canadian annotation review</b><br>Draw tight axis-aligned boxes using original pixel coordinates. Do not label docks, wakes or uncertain objects as ships. Aerial imagery must be trained and evaluated separately from the 10 m model.'),
                         select,widgets.HBox(coords),widgets.HBox([add,undo,draft]),notes,reviewer,checked,negative,blind,save,out])
