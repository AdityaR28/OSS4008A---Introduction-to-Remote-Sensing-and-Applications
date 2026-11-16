"""Reproducible Sentinel-2 vessel detection backend. Run in Google Colab."""
from pathlib import Path
from collections import OrderedDict
import hashlib
import io
import json
import random
import re
import time
import zipfile

import numpy as np
import pandas as pd
import requests
import torch
import torchvision
from PIL import Image, ImageDraw
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models import ResNet18_Weights
from torchvision.models.detection.faster_rcnn import TwoMLPHead

DATA_URL = 'https://zenodo.org/records/15571607/files/SEN2MS_Vessel_BBoxes.zip?download=1'
DATA_MD5 = '830a5e741425c736b3971055ff2e9a9f'
DATA_DOI = 'https://doi.org/10.5281/zenodo.15571607'
DATA_LICENSE = 'CC-BY-4.0'


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def checksum(path, algorithm='md5'):
    digest = hashlib.new(algorithm)
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def download_dataset(project):
    """Download data only; never execute code from the archive."""
    raw = Path(project) / 'data/raw'
    raw.mkdir(parents=True, exist_ok=True)
    destination = raw / 'SEN2MS_Vessel_BBoxes.zip'
    if not destination.exists() or checksum(destination) != DATA_MD5:
        temporary = destination.with_suffix('.partial')
        with requests.get(DATA_URL, stream=True, timeout=120) as response:
            response.raise_for_status()
            with temporary.open('wb') as handle:
                for chunk in response.iter_content(4 * 1024 * 1024):
                    handle.write(chunk)
        if checksum(temporary) != DATA_MD5:
            raise ValueError('Dataset checksum differs from the published archive.')
        temporary.replace(destination)
    print('Dataset archive verified:', destination.stat().st_size, 'bytes')
    return destination


def prepare_dataset(project, archive_path, seed=42):
    """Convert original labels to Labelme JSON and split by MGRS tile.

    Image identity includes country AND source category: the original archive
    reuses numeric image names in vessel/non-vessel directories.
    """
    project = Path(project)
    image_dir, annotation_dir = project / 'data/images', project / 'data/annotations'
    image_dir.mkdir(parents=True, exist_ok=True)
    annotation_dir.mkdir(parents=True, exist_ok=True)
    rows, seen, exclusions = [], set(), {'cloud': 0, 'duplicate_rgb': 0}
    with zipfile.ZipFile(archive_path) as archive:
        members = set(archive.namelist())
        for name in sorted(n for n in members if n.endswith('.csv')):
            table = pd.read_csv(io.BytesIO(archive.read(name)))
            parent = name.rsplit('/', 1)[0]
            country, category = parent.split('/')[-2:]
            for _, source in table.iterrows():
                if int(source['Clouds']) > 1:
                    exclusions['cloud'] += 1
                    continue
                stem = Path(source['Image_name']).stem
                member = f'{parent}/MS/{stem}.npy'
                if member not in members:
                    raise FileNotFoundError(member)
                array = np.load(io.BytesIO(archive.read(member)), allow_pickle=False)
                if array.shape != (13, 128, 128) or not np.isfinite(array).all():
                    raise ValueError(f'Unexpected source array: {member}')
                # Source documentation indexes B04/B03/B02 at 3/2/1 and stretches
                # each band to 0..255. Do not divide these stored values by 10000.
                rgb = np.rint(np.clip(array[[3, 2, 1]], 0, 255)).astype(np.uint8)
                digest = hashlib.sha256(rgb.tobytes()).hexdigest()
                if digest in seen:
                    exclusions['duplicate_rgb'] += 1
                    continue
                seen.add(digest)
                label_member = f'{parent}/BBoxes/{stem}.txt'
                if category == 'vessel' and label_member not in members:
                    raise FileNotFoundError(f'Missing vessel labels: {label_member}')
                boxes = []
                if label_member in members:
                    for line in archive.read(label_member).decode().splitlines():
                        if not line.strip():
                            continue
                        label, cx, cy, width, height = map(float, line.split())
                        if label != 0 or not (0 < width <= 1 and 0 < height <= 1):
                            raise ValueError(f'Invalid YOLO label: {label_member}')
                        box = np.array([cx-width/2, cy-height/2, cx+width/2, cy+height/2]) * 128
                        box = np.clip(box, 0, 128).tolist()
                        if box[2] <= box[0] or box[3] <= box[1]:
                            raise ValueError('Degenerate bounding box.')
                        boxes.append(box)
                tile_match = re.search(r'_T(\d{2}[A-Z]{3})_', str(source['Picture_title']))
                if not tile_match:
                    raise ValueError('Source MGRS tile is missing.')
                identity = f'{country}_{category}_{stem}'
                image_path = image_dir / f'{identity}.png'
                Image.fromarray(rgb.transpose(1, 2, 0)).save(image_path)
                annotation = {
                    'version': '5.0.1', 'flags': {},
                    'shapes': [{'label': 'ship', 'points': [[b[0], b[1]], [b[2], b[3]]],
                                'group_id': None, 'shape_type': 'rectangle', 'flags': {}} for b in boxes],
                    'imagePath': f'../images/{identity}.png', 'imageData': None,
                    'imageHeight': 128, 'imageWidth': 128,
                }
                annotation_path = annotation_dir / f'{identity}.json'
                write_json(annotation_path, annotation)
                rows.append({'image_id': identity, 'country': country,
                             'source_tile': tile_match.group(1), 'source_scene': source['Picture_title'],
                             'image_path': str(image_path.relative_to(project)),
                             'annotation_path': str(annotation_path.relative_to(project)),
                             'boxes': len(boxes), 'rgb_sha256': digest})
    manifest = pd.DataFrame(rows)
    # Fixed geographic groups, chosen without looking at model performance.
    # Stratify tile lists by whether any ship labels are available in that tile.
    tile_has_ship = manifest.groupby('source_tile')['boxes'].sum().gt(0)
    assignment = {}
    rng = random.Random(seed)
    for positive in [True, False]:
        tiles = sorted(tile_has_ship[tile_has_ship == positive].index)
        rng.shuffle(tiles)
        count = max(1, round(len(tiles) * 0.2)) if len(tiles) >= 3 else 0
        for index, tile in enumerate(tiles):
            assignment[tile] = 'test' if index < count else 'val' if index < 2*count else 'train'
    manifest['split'] = manifest.source_tile.map(assignment)
    for split in ['train', 'val', 'test']:
        subset = manifest[manifest.split == split]
        if subset.empty or subset.boxes.sum() == 0:
            raise ValueError(f'{split} has no usable ship examples.')
    assert manifest.groupby('source_tile').split.nunique().max() == 1
    assert manifest.rgb_sha256.is_unique
    manifest.to_csv(project / 'data/dataset_manifest.csv', index=False)
    summary = {'source': DATA_DOI, 'license': DATA_LICENSE, 'archive_md5': DATA_MD5,
               'source_processing': 'Sentinel-2 L1C, source per-band percentile stretch to 0..255',
               'input_bands': ['B4', 'B3', 'B2'], 'seed': seed,
               'split_method': 'MGRS tile groups, 60/20/20 approximate; exact RGB duplicates removed',
               'exclusions': exclusions,
               'splits': {s: {'images': int((manifest.split == s).sum()),
                              'boxes': int(manifest.loc[manifest.split == s, 'boxes'].sum()),
                              'tiles': sorted(manifest.loc[manifest.split == s, 'source_tile'].unique())}
                          for s in ['train', 'val', 'test']}}
    write_json(project / 'results/dataset_audit.json', summary)
    print(json.dumps(summary, indent=2))
    return manifest


class VesselDataset(Dataset):
    def __init__(self, project, table, augment=False):
        self.project, self.rows, self.augment = Path(project), table.to_dict('records'), augment

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        image = np.array(Image.open(self.project / row['image_path']).convert('RGB'))
        image = torch.from_numpy(image.copy()).permute(2, 0, 1).float() / 255
        annotation = json.loads((self.project / row['annotation_path']).read_text())
        boxes = torch.tensor([s['points'][0]+s['points'][1] for s in annotation['shapes']],
                             dtype=torch.float32).reshape(-1, 4)
        if self.augment and random.random() < 0.5:
            image = image.flip(-1)
            boxes[:, [0, 2]] = 128 - boxes[:, [2, 0]]
        target = {'boxes': boxes, 'labels': torch.ones(len(boxes), dtype=torch.int64),
                  'image_id': torch.tensor(index),
                  'area': (boxes[:, 2]-boxes[:, 0])*(boxes[:, 3]-boxes[:, 1]),
                  'iscrowd': torch.zeros(len(boxes), dtype=torch.int64)}
        return image, target


def collate(batch):
    return tuple(zip(*batch))


def build_mobilenet_detector(pretrained=True):
    """Transfer a COCO MobileNet detector to one vessel class and small boxes."""
    model = fasterrcnn_mobilenet_v3_large_fpn(
        weights='DEFAULT' if pretrained else None, weights_backbone=None,
        min_size=256, max_size=256, box_score_thresh=0.01,
        rpn_pre_nms_top_n_train=1000, rpn_post_nms_top_n_train=300,
        rpn_pre_nms_top_n_test=600, rpn_post_nms_top_n_test=200,
        box_detections_per_img=100,
    )
    original = model.roi_heads.box_predictor
    predictor = FastRCNNPredictor(original.cls_score.in_features, 2)
    if pretrained:
        # COCO category 9 is boat. Preserve its learned classifier/regressor as
        # the initial ship class instead of starting the output layer randomly.
        with torch.no_grad():
            predictor.cls_score.weight.copy_(original.cls_score.weight[[0, 9]])
            predictor.cls_score.bias.copy_(original.cls_score.bias[[0, 9]])
            predictor.bbox_pred.weight.copy_(torch.cat([original.bbox_pred.weight[:4], original.bbox_pred.weight[36:40]]))
            predictor.bbox_pred.bias.copy_(torch.cat([original.bbox_pred.bias[:4], original.bbox_pred.bias[36:40]]))
    model.roi_heads.box_predictor = predictor
    # Preserve 15 anchors/location so the pretrained RPN output dimensions match.
    model.rpn.anchor_generator = AnchorGenerator(((4, 8, 16, 32, 64),)*3, ((0.5, 1., 2.),)*3)
    return model


def build_detector(pretrained=True):
    """ImageNet-pretrained ResNet18 with stride-4 FPN features for small ships.

    A smaller 512-unit ROI head keeps the full half-precision model below 50 MB.
    This is a transfer-learning detector; the RPN and detection heads are learned
    from the vessel annotations, while the backbone starts from ImageNet weights.
    """
    backbone = resnet_fpn_backbone('resnet18',
                                  weights=ResNet18_Weights.DEFAULT if pretrained else None,
                                  trainable_layers=3)
    anchors = AnchorGenerator(((4,), (8,), (16,), (32,), (64,)), ((0.5,1.,2.),)*5)
    head = TwoMLPHead(256*7*7, 512)
    predictor = FastRCNNPredictor(512, 2)
    return FasterRCNN(backbone, num_classes=None, min_size=256, max_size=256,
                      rpn_anchor_generator=anchors, box_head=head, box_predictor=predictor,
                      box_score_thresh=0.01, box_detections_per_img=100,
                      rpn_pre_nms_top_n_train=1000, rpn_post_nms_top_n_train=300,
                      rpn_pre_nms_top_n_test=600, rpn_post_nms_top_n_test=200)


def evaluate_detector(model, dataset, device, score_threshold=0.3):
    """COCO AP plus greedy one-to-one precision/recall at IoU 0.50."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    model.eval()
    annotations, detections, images = [], [], []
    tp = fp = fn = 0
    started = time.perf_counter()
    with torch.inference_mode():
        for index in range(len(dataset)):
            image, target = dataset[index]
            prediction = model([image.to(device)])[0]
            boxes, scores = prediction['boxes'].cpu(), prediction['scores'].cpu()
            images.append({'id': index, 'width': 128, 'height': 128})
            for box in target['boxes'].tolist():
                x1, y1, x2, y2 = box
                annotations.append({'id': len(annotations)+1, 'image_id': index,
                                    'category_id': 1, 'bbox': [x1, y1, x2-x1, y2-y1],
                                    'area': (x2-x1)*(y2-y1), 'iscrowd': 0})
            for box, score in zip(boxes.tolist(), scores.tolist()):
                x1, y1, x2, y2 = box
                detections.append({'image_id': index, 'category_id': 1,
                                   'bbox': [x1, y1, x2-x1, y2-y1], 'score': score})
            selected = boxes[scores >= score_threshold]
            gt = target['boxes']
            matched = set()
            overlaps = torchvision.ops.box_iou(selected, gt)
            for overlap in overlaps:
                options = [(float(v), j) for j, v in enumerate(overlap) if j not in matched]
                best = max(options, default=(0., -1))
                if best[0] >= 0.5:
                    tp += 1
                    matched.add(best[1])
                else:
                    fp += 1
            fn += len(gt)-len(matched)
    ground_truth = COCO()
    ground_truth.dataset = {'images': images, 'annotations': annotations,
                            'categories': [{'id': 1, 'name': 'ship'}], 'info': {}}
    ground_truth.createIndex()
    if detections:
        result = ground_truth.loadRes(detections)
        evaluation = COCOeval(ground_truth, result, 'bbox')
        evaluation.evaluate()
        evaluation.accumulate()
        evaluation.summarize()
        ap, ap50 = map(float, evaluation.stats[:2])
    else:
        ap = ap50 = 0.
    return {'images': len(dataset), 'ground_truth_boxes': len(annotations),
            'ap_50_95': ap, 'ap_50': ap50,
            'precision_at_threshold': tp/max(tp+fp, 1), 'recall_at_threshold': tp/max(tp+fn, 1),
            'score_threshold': score_threshold, 'iou_threshold': 0.5,
            'true_positives': tp, 'false_positives': fp, 'false_negatives': fn,
            'seconds': time.perf_counter()-started}


def train_detector(project, manifest, epochs=20, seed=42, architecture='resnet18'):
    project = Path(project)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if architecture not in ['resnet18', 'mobilenet']:
        raise ValueError('Unknown detector architecture.')
    model = (build_detector(pretrained=True) if architecture=='resnet18'
             else build_mobilenet_detector(pretrained=True)).to(device)
    training = manifest[manifest.split == 'train']
    # Balance the training sample only. Validation/test retain all eligible data.
    positive, negative = training[training.boxes > 0], training[training.boxes == 0]
    training = pd.concat([positive, negative.sample(min(len(negative), len(positive)), random_state=seed)])
    train_set = VesselDataset(project, training, augment=True)
    val_set = VesselDataset(project, manifest[manifest.split == 'val'])
    test_set = VesselDataset(project, manifest[manifest.split == 'test'])
    loader = DataLoader(train_set, batch_size=8 if device.type == 'cuda' else 2,
                        shuffle=True, collate_fn=collate, num_workers=0)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=0.0001 if architecture=='resnet18' else 0.00015, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.3)
    best_ap, history, best_state = -1., [], None
    for epoch in range(epochs):
        model.train()
        losses, started = [], time.perf_counter()
        for images, targets in loader:
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss = sum(model(images, targets).values())
            if not torch.isfinite(loss):
                raise ValueError('Non-finite training loss.')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            losses.append(float(loss.detach()))
        validation = evaluate_detector(model, val_set, device)
        scheduler.step()
        record = {'epoch': epoch+1, 'loss': float(np.mean(losses)), 'validation': validation,
                  'epoch_seconds': time.perf_counter()-started}
        history.append(record)
        print('Epoch', epoch+1, '| loss', round(record['loss'], 4), '| validation AP50', round(validation['ap_50'], 4), flush=True)
        if validation['ap_50'] > best_ap:
            best_ap = validation['ap_50']
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        write_json(project / 'results/training_history.json', history)
    model.load_state_dict(best_state)
    # Evaluate the untouched test partition once after validation selects a model.
    metrics = evaluate_detector(model, test_set, device)
    metrics.update({'partition': 'test', 'selection': 'best validation AP50',
                    'epochs': epochs, 'training_images': len(train_set),
                    'validation_best_ap50': best_ap,
                    'dataset': DATA_DOI, 'seed': seed,
                    'architecture': architecture,
                    'canadian_accuracy_measured': False,
                    'torch': str(torch.__version__), 'torchvision': str(torchvision.__version__)})
    model_path = project / 'results/models/vessel_detector.pt'
    model_path.parent.mkdir(parents=True, exist_ok=True)
    # Half precision storage keeps the complete checkpoint below 50 MB.
    compact = {k: v.half() if v.is_floating_point() else v for k, v in best_state.items()}
    architecture_name = 'fasterrcnn_resnet18_fpn_small' if architecture=='resnet18' else 'fasterrcnn_mobilenet_v3_large_fpn'
    torch.save({'state_dict': compact, 'architecture': architecture_name,
                'input_size': 256, 'class_names': ['background', 'ship'], 'metrics': metrics}, model_path)
    assert model_path.stat().st_size < 50_000_000
    write_json(project / 'results/test_metrics.json', metrics)
    print('Checkpoint:', model_path, '| bytes:', model_path.stat().st_size, flush=True)
    print(json.dumps(metrics, indent=2), flush=True)
    return model, device, metrics


def load_detector(path, device=None):
    device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    saved = torch.load(path, map_location='cpu', weights_only=True)
    if saved['architecture'] not in ['fasterrcnn_resnet18_fpn_small','fasterrcnn_mobilenet_v3_large_fpn']:
        raise ValueError('Unsupported checkpoint architecture.')
    model = (build_detector(pretrained=False) if saved['architecture']=='fasterrcnn_resnet18_fpn_small'
             else build_mobilenet_detector(pretrained=False))
    model.load_state_dict(saved['state_dict'], strict=True)
    return model.to(device).eval(), device, saved['metrics']


def predict_rgb(model, device, rgb, threshold=0.3):
    """Return x1,y1,x2,y2 boxes and scores in the supplied image pixel grid."""
    array = np.asarray(rgb)
    if array.ndim != 3 or array.shape[2] != 3 or array.dtype != np.uint8:
        raise ValueError('Expected an HxWx3 uint8 RGB image.')
    tensor = torch.from_numpy(array.copy()).permute(2, 0, 1).float().to(device)/255
    model.eval()
    with torch.inference_mode():
        prediction = model([tensor])[0]
    keep = prediction['scores'] >= threshold
    return prediction['boxes'][keep].cpu().numpy(), prediction['scores'][keep].cpu().numpy()
