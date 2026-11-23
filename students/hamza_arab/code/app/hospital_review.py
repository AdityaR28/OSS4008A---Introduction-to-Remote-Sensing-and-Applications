"""Human annotation helper for the *separate* hospital AI training notebook.

This does NOT modify the working Hospital Infrastructure Canada map UI. It
requires a person to verify each selected roof and its building function.
"""
from __future__ import annotations
import base64
import html
import json
import uuid
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image
from IPython.display import HTML, Javascript, JSON, display


def save_annotation_review(root: str | Path, payload: dict[str, Any]) -> dict[str, Any]:
    root = Path(root).resolve()
    csv_path = root / "review_manifest.csv"
    manifest = pd.read_csv(csv_path, keep_default_na=False)
    token = str(payload.get("tile_id", ""))
    indexes = manifest.index[manifest.tile_id.eq(token)].tolist()
    if len(indexes) != 1:
        raise ValueError("Unknown tile ID")
    index = indexes[0]
    row = manifest.loc[index]
    image_path = (root / row.image).resolve()
    annotation_path = (root / row.annotation).resolve()
    if not image_path.is_relative_to(root) or not annotation_path.is_relative_to(root):
        raise ValueError("Invalid image/annotation path")
    reviewer = str(payload.get("reviewer", "")).strip()
    if not reviewer or len(reviewer) > 90:
        raise ValueError("A named reviewer is required")
    category = str(payload.get("category", ""))
    if category == "SKIP_UNCERTAIN":
        if payload.get("boxes"):
            raise ValueError("Undo all boxes before excluding an ambiguous tile")
        manifest.loc[index, "review_status"] = "REJECTED"
        manifest.loc[index, "reviewed_label"] = "UNKNOWN"
        manifest.loc[index, "reviewer"] = reviewer
        manifest.loc[index, "source_checked"] = "NO"
        manifest.to_csv(csv_path, index=False)
        return {"ok": True, "saved": "EXCLUDED: no training label created",
                "remaining_unreviewed": int(manifest.review_status.eq("UNREVIEWED").sum())}
    if payload.get("confirmed") is not True:
        raise ValueError("You must confirm imagery rights, location and building function were reviewed")
    if category not in ("HOSPITAL_PRESENT", "NO_HOSPITAL"):
        raise ValueError("Explicit hospital present / no hospital choice required")
    boxes = payload.get("boxes")
    if not isinstance(boxes, list) or len(boxes) > 100:
        raise ValueError("Invalid number of boxes")
    if category == "HOSPITAL_PRESENT" and not boxes:
        raise ValueError("Positive tile requires at least one drawn hospital roof rectangle")
    if category == "NO_HOSPITAL" and boxes:
        raise ValueError("Negative tile cannot contain hospital rectangles; undo boxes first")
    with Image.open(image_path) as im:
        width, height = im.size
    shapes = []
    for box in boxes:
        if not isinstance(box, list) or len(box) != 4:
            raise ValueError("Each drawn box requires four numeric coordinates")
        x1, y1, x2, y2 = [float(v) for v in box]
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height and (x2-x1)>=3 and (y2-y1)>=3):
            raise ValueError("Box corners must be inside the image with nonzero area")
        shapes.append({"label": "hospital", "points": [[x1,y1],[x2,y2]],
                       "group_id": None, "description": "Human-reviewed hospital rooftop", "shape_type": "rectangle", "flags": {}})
    labelme = {"version": "5.5.0", "flags": {"verified": True}, "shapes": shapes,
               "imagePath": image_path.name, "imageData": None, "imageHeight": height,
               "imageWidth": width, "reviewed_by": reviewer,
               "review_note": "Building use checked by a human. Data source and license verified."}
    annotation_path.parent.mkdir(parents=True, exist_ok=True)
    annotation_path.write_text(json.dumps(labelme, indent=2), encoding="utf-8")
    manifest.loc[index, "review_status"] = "APPROVED"
    manifest.loc[index, "reviewed_label"] = category
    manifest.loc[index, "reviewer"] = reviewer
    manifest.loc[index, "source_checked"] = "YES"
    manifest.to_csv(csv_path, index=False)
    remaining = int(manifest.review_status.eq("UNREVIEWED").sum())
    return {"ok": True, "saved": str(annotation_path), "remaining_unreviewed": remaining}



def review_saved_status(root: str | Path, tile_id: str) -> dict[str, Any]:
    """Read the real manifest and annotation from disk; do not trust a UI message."""
    root = Path(root).resolve()
    manifest = pd.read_csv(root / "review_manifest.csv", keep_default_na=False)
    rows = manifest.loc[manifest.tile_id.astype(str).eq(str(tile_id))]
    if len(rows) != 1:
        return {"ok": False, "saved": False, "error": "Review tile not found"}
    row = rows.iloc[0]
    status = str(row.review_status)
    path = (root / str(row.annotation)).resolve()
    safe_path = path.is_relative_to(root)
    saved = status == "REJECTED" or (status == "APPROVED" and safe_path and path.is_file())
    return {"ok": True, "saved": bool(saved), "review_status": status,
            "remaining_unreviewed": int(manifest.review_status.eq("UNREVIEWED").sum())}


def colab_save_review_result(root: str | Path, payload: dict[str, Any]) -> JSON:
    """Return IPython.display.JSON, which Colab's invokeFunction actually transports.

    Returning a plain Python dict is not enough for Colab's documented
    response.data['application/json'] contract.
    """
    try:
        result = save_annotation_review(root, payload)
        verified = review_saved_status(root, str(payload.get("tile_id", "")))
        if not verified.get("saved"):
            result = {"ok": False, "error": "Save returned but the annotation did not persist."}
        else:
            result["remaining_unreviewed"] = verified["remaining_unreviewed"]
        return JSON(result)
    except Exception as exc:
        return JSON({"ok": False, "error": f"{type(exc).__name__}: {exc}"[:450]})


def colab_review_status_result(root: str | Path, tile_id: str) -> JSON:
    """Independent read-back for an ambiguous or lost Colab browser response."""
    try:
        return JSON(review_saved_status(root, tile_id))
    except Exception as exc:
        return JSON({"ok": False, "saved": False,
                     "error": f"{type(exc).__name__}: {exc}"[:450]})

def next_unreviewed_payload(root: str | Path) -> dict[str, Any]:
    """Fetch the next *actual* pending image without rerunning a notebook cell.

    Includes cached OSM proposals but never converts proposals into approval.
    """
    root = Path(root).resolve()
    df = pd.read_csv(root / 'review_manifest.csv', keep_default_na=False)
    waiting = df.loc[df.review_status.eq('UNREVIEWED')]
    if waiting.empty:
        return {'ok': True, 'done': True, 'remaining': 0}
    row = waiting.iloc[0]
    path = (root / str(row.image)).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise FileNotFoundError('Next review image missing or outside project root')
    with Image.open(path) as im:
        width, height = im.size
        mime = Image.MIME.get(im.format, 'image/png')
    if width>1500 or height>1500 or mime not in ('image/png','image/jpeg','image/webp'):
        raise ValueError('Unsupported annotation image')
    from osm_auto_annotations import load_draft_boxes
    return {'ok':True, 'done':False, 'remaining':int(len(waiting)),
            'tile_id':str(row.tile_id), 'name':str(row['name']),
            'candidate_class':str(row.candidate_class), 'width':int(width),
            'height':int(height), 'image_src':f'data:{mime};base64,'+
            base64.b64encode(path.read_bytes()).decode('ascii'),
            'suggested_boxes':load_draft_boxes(root,str(row.tile_id)),
            'draft_status':'OPENSTREETMAP DRAFT — verify roof and hospital use'}


def colab_next_review_result(root: str | Path) -> JSON:
    try:
        return JSON(next_unreviewed_payload(root))
    except Exception as exc:
        return JSON({'ok':False,'error':f'{type(exc).__name__}: {exc}'[:400]})


def build_labeler_ui(tile_id: str, name: str, candidate_class: str, image_path: Path, suggested_boxes=None):
    """Return static HTML and executable JavaScript as separate notebook outputs.

    Colab may render HTML while not executing embedded <script> elements.
    The image is therefore a normal <img> element, visible even when JavaScript
    fails, and we execute event bindings explicitly with display(Javascript).
    """
    image_path = Path(image_path)
    with Image.open(image_path) as src:
        width, height = src.size
        mime = Image.MIME.get(src.format, "image/png")
    if width > 1500 or height > 1500:
        raise ValueError("Review UI supports images up to 1500 pixels per side")
    if mime not in ("image/png", "image/jpeg", "image/webp"):
        raise ValueError("Review UI supports PNG/JPEG/WebP imagery")

    root_id = "oss4008-review-" + uuid.uuid4().hex[:12]
    img_uri = f"data:{mime};base64,{base64.b64encode(image_path.read_bytes()).decode('ascii')}"
    title = html.escape(f"{name} · {candidate_class} · {tile_id}", quote=True)
    # Safe static HTML: never depend on a script to display the aerial photograph.
    markup = f'''<div id="{root_id}" style="background:#f6f8fb;color:#203449;padding:14px;border:1px solid #c8d2df;border-radius:10px;font:14px system-ui;max-width:950px">
      <div style="font-weight:750;font-size:16px">Hospital AI · Manual roof review</div>
      <p data-role="title" style="color:#435a76;font-weight:600">{title}</p>
      <p style="color:#435a76">The image below is real overhead imagery. Outline only roofs of buildings that you have independently confirmed are hospitals. A reference location alone is not proof.</p>
      <div style="position:relative;width:100%;max-width:640px;border:1px solid #b9c7dc;background:#e9eef4;line-height:0">
        <img data-role="image" src="{img_uri}" width="{width}" height="{height}" alt="Aerial photograph to label" style="width:100%;height:auto;display:block;"/>
        <canvas data-role="canvas" width="{width}" height="{height}" style="position:absolute;inset:0;width:100%;height:100%;cursor:crosshair;touch-action:none"></canvas>
      </div>
      <p><button data-role="undo" type="button">Undo last box</button> <button data-role="use-drafts" type="button">Use suggested OSM boxes</button> <span data-role="count">0 rooftop rectangles</span> <span data-role="draft-count" style="color:#2563eb">Drafts are unverified</span></p>
      <div style="display:flex;flex-wrap:wrap;gap:12px;align-items:center">
        <label style="color:#203449">Verified classification
          <select data-role="category" style="color:#203449;background:white;border:1px solid #9aaac0;padding:5px">
            <option value="">Select...</option>
            <option value="HOSPITAL_PRESENT">Hospital rooftop(s) drawn</option>
            <option value="NO_HOSPITAL">No hospital rooftop in this image</option>
            <option value="SKIP_UNCERTAIN">Cannot verify: exclude image</option>
          </select>
        </label>
        <label style="color:#203449">Reviewer <input data-role="reviewer" placeholder="Your initials/name" style="color:#203449;background:white;border:1px solid #9aaac0;padding:5px"/></label>
      </div>
      <p><label style="color:#203449"><input data-role="confirm" type="checkbox"/> I personally checked the building function, image source and permission to use the image for training.</label></p>
      <button data-role="save" type="button" style="background:#235bdd;color:white;padding:10px;border:0;border-radius:6px">Save reviewed annotation</button>
      <p data-role="status" aria-live="polite" style="color:#314860;white-space:pre-wrap">Image displayed. Initializing drawing controls…</p>
    </div>'''

    config = json.dumps({"root_id": root_id, "tile_id": str(tile_id),
                         "width": width, "height": height, "drafts": suggested_boxes or []}, ensure_ascii=False)
    # Put JS in an actual Javascript MIME output rather than static HTML.
    # No untrusted filename or user's free text is interpolated as JS source.
    js = r'''(() => {
      const config = __CONFIG__;
      const root = document.getElementById(config.root_id);
      if (!root) throw new Error('Annotation HTML not ready in Colab output frame');
      const get = role => root.querySelector('[data-role="' + role + '"]');
      const canvas = get('canvas');
      const img = get('image');
      const status = get('status');
      const ctx = canvas.getContext('2d');
      if (!ctx) { status.textContent = 'Drawing unavailable: browser 2D canvas unsupported.'; return; }
      const boxes = [];
      const drafts = [...config.drafts];
      let start = null, cursor = null;
      const coords = event => {
        const r = canvas.getBoundingClientRect();
        return [Math.max(0, Math.min(config.width, (event.clientX-r.left)*config.width/r.width)),
                Math.max(0, Math.min(config.height,(event.clientY-r.top)*config.height/r.height))];
      };
      const paint = () => {
        ctx.clearRect(0, 0, config.width, config.height);
        ctx.lineWidth = 3;
        ctx.strokeStyle = '#ffe000';
        for (const b of boxes) ctx.strokeRect(b[0], b[1], b[2]-b[0], b[3]-b[1]);
        ctx.save();ctx.setLineDash([8,5]);ctx.strokeStyle='#2d70ff';
        for (const b of drafts) ctx.strokeRect(b[0],b[1],b[2]-b[0],b[3]-b[1]);
        ctx.restore();
        get('draft-count').textContent=drafts.length+' OSM draft boxes (BLUE DASHED = unverified)';
        if (start && cursor) {
          const b = [Math.min(start[0],cursor[0]),Math.min(start[1],cursor[1]),
                     Math.max(start[0],cursor[0]),Math.max(start[1],cursor[1])];
          ctx.strokeStyle = '#2ce2b8';
          ctx.strokeRect(b[0], b[1], b[2]-b[0], b[3]-b[1]);
        }
        get('count').textContent = boxes.length + ' rooftop rectangles';
      };
      canvas.addEventListener('pointerdown', event => {
        if (get('save').disabled) return;
        canvas.setPointerCapture(event.pointerId);
        start = coords(event); cursor = start; paint();
      });
      canvas.addEventListener('pointermove', event => {
        if (start) { cursor = coords(event); paint(); }
      });
      canvas.addEventListener('pointerup', event => {
        if (!start) return;
        cursor = coords(event);
        const b = [Math.min(start[0],cursor[0]),Math.min(start[1],cursor[1]),
                   Math.max(start[0],cursor[0]),Math.max(start[1],cursor[1])];
        if (b[2]-b[0]>=3 && b[3]-b[1]>=3) boxes.push(b);
        start = null; cursor = null; paint();
      });
      canvas.addEventListener('pointercancel', () => { start=null; cursor=null; paint(); });
      get('undo').onclick = () => { boxes.pop(); paint(); };
      get('use-drafts').onclick = () => {
        if(!drafts.length){status.textContent='No mapped building footprint proposals for this image. Draw manually or exclude uncertain image.';return;}
        boxes.push(...drafts.map(b=>b.slice()));drafts.length=0;paint();
        status.textContent='OSM draft outlines copied. CHECK EACH BOX and remove incorrect boxes (Undo) before approving. OSM geometry is NOT proof of hospital use.';
      };
      const readJson = reply => {
        let value=reply?.data?.['application/json'];
        if (typeof value==='string'){try{value=JSON.parse(value);}catch(_){}}
        if(Array.isArray(value)&&value.length===1)value=value[0];
        return value;
      };
      async function openNext(api){
        status.textContent='Saved to Google Drive. Loading next aerial image…';
        try {
          const payload=readJson(await api.invokeFunction('oss4008_hospital_ai.next_review',[],{}));
          if(!payload?.ok){status.textContent='Saved, but next image unavailable: '+String(payload?.error||'unknown error');return;}
          if(payload.done){
            status.textContent='ALL REVIEWS FINISHED. Rerun the training cell to check training requirements.';
            get('save').disabled=true;get('use-drafts').disabled=true;return;
          }
          config.tile_id=payload.tile_id; config.width=payload.width;config.height=payload.height;
          img.src=payload.image_src;img.width=payload.width;img.height=payload.height;
          canvas.width=payload.width;canvas.height=payload.height;
          get('title').textContent=payload.name+' · '+payload.candidate_class+' · '+payload.tile_id;
          boxes.length=0;drafts.length=0;
          drafts.push(...(payload.suggested_boxes||[]).map(b=>b.slice()));
          get('category').value='';get('confirm').checked=false;
          start=null;cursor=null;get('save').disabled=false;paint();
          status.textContent='NEXT IMAGE OPENED automatically. '+payload.remaining+' remain. Blue dashed OSM proposals require verification.';
        }catch(error){status.textContent='Saved, but could not advance automatically: '+String(error).slice(0,160);}
      }
      get('save').onclick = async () => {
        const button = get('save');
        if (!get('reviewer').value.trim()) { status.textContent = 'Enter reviewer name or initials first.'; return; }
        const category = get('category').value;
        if (!category) { status.textContent = 'Select hospital, no hospital, or exclude this image.'; return; }
        if (category === 'HOSPITAL_PRESENT' && boxes.length===0) {
          status.textContent = 'Draw at least one rectangle around a verified hospital roof.'; return;
        }
        if (category !== 'HOSPITAL_PRESENT' && boxes.length>0) {
          status.textContent = 'Undo rectangles before choosing no hospital or exclude.'; return;
        }
        if (category !== 'SKIP_UNCERTAIN' && !get('confirm').checked) {
          status.textContent = 'Confirm the building use, imagery source and training rights before saving.'; return;
        }
        const api = window.google?.colab?.kernel;
        if (!api || typeof api.invokeFunction !== 'function') {
          status.textContent = 'Colab callback unavailable. Rerun the annotation cell in an active Colab runtime.';
          return;
        }
        button.disabled = true;
        status.textContent = 'Saving review to Google Drive…';
        const payload = {tile_id:config.tile_id, category,
          reviewer:get('reviewer').value.trim(), confirmed:get('confirm').checked,
          boxes:boxes.map(box=>box.slice())};
        try {
          // Google Colab's official callback format is JSON data in application/json.
          const response = await api.invokeFunction('oss4008_hospital_ai.save_review',[payload],{});
          const result = readJson(response);
          if (result?.ok) {
            await openNext(api);
          } else if (result?.error) {
            status.textContent = 'Could not save: ' + result.error;
            button.disabled = false;
          } else {
            // Older Colab outputs can return an empty browser message even
            // after Python finished. Verify the actual files before reporting failure.
            const verifyResponse = await api.invokeFunction('oss4008_hospital_ai.review_status',[config.tile_id],{});
            const verified = readJson(verifyResponse);
            if (verified?.saved) {
              await openNext(api);
            } else {
              status.textContent = 'Save not confirmed: ' + (verified?.error || 'No reviewed annotation found on disk.');
              button.disabled = false;
            }
          }
        } catch (error) {
          status.textContent = 'Could not save: ' + String(error);
          button.disabled = false;
        }
      };
      status.textContent = 'Ready. Drag across a verified hospital roof to draw a rectangle.';
      paint();
    })();'''.replace('__CONFIG__', config)
    return markup, js


def launch_labeler(root: str | Path) -> dict[str, Any]:
    """Show one image in Colab and bind its drawing controls with Javascript MIME.

    All annotation data remains in Drive. Nothing is auto-approved or labeled.
    """
    root = Path(root).resolve()
    manifest = pd.read_csv(root / "review_manifest.csv", keep_default_na=False)
    pending = manifest.loc[manifest.review_status.eq("UNREVIEWED")]
    if pending.empty:
        print("No unreviewed images remain. Check approved/omitted site coverage before training.")
        return {"remaining": 0}
    row = pending.iloc[0]
    image_path = (root / str(row.image)).resolve()
    if not image_path.is_relative_to(root) or not image_path.is_file():
        raise FileNotFoundError(f"Missing candidate image: {row.image}")
    from google.colab import output
    output.register_callback("oss4008_hospital_ai.save_review",
                             lambda payload: colab_save_review_result(root, payload))
    output.register_callback("oss4008_hospital_ai.review_status",
                             lambda tile_id: colab_review_status_result(root, tile_id))
    output.register_callback("oss4008_hospital_ai.next_review",
                             lambda: colab_next_review_result(root))
    from osm_auto_annotations import load_draft_boxes
    suggested = load_draft_boxes(root,str(row.tile_id))
    markup, script = build_labeler_ui(str(row.tile_id), str(row["name"]),
                                     str(row.candidate_class), image_path, suggested_boxes=suggested)
    display(HTML(markup))
    display(Javascript(script))
    return {"tile_id": str(row.tile_id), "remaining": len(pending),
            "human_review_required": True}
