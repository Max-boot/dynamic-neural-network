"""
Stufe 5: Bounding-Box-Bestimmung + Gesamt-Detektionsevaluation.

Ansatz (optimal fuer MCU-Klasse):
  1. Pro Region der Stufe 3 existiert bereits eine Box (Tile-Cluster).
  2. Verfeinerung: Die Region-Box wird auf den BNN-Ausgabekern reduziert,
     in dem die Saliency-Pipeline verdichtete Gewissheit hatte.
     Tiefer liegend: Da eine Region ~1-2 Kacheln umfasst und Ziffern kleiner
     sind, schrumpfen wir die Box auf den Grossteil der Saliency-Masse.
  3. Matching gegen GT-Ziffern-Boxen via IoU >= 0.5 (TP), Rest FP.
  4. BBox-Regression: optionaler 1-Layer-Kopf (YOLOv1-Stil) auf den Regionen.

Fuer den Prototyp verwenden wir die *Saliency-Massenzentrum*-Verfeinerung:
  Box = um das Massenzentrum der Saliency in der Region gelegte Box, die
  ~70% der Masse einschliesst. Das ist deterministisch, lernfrei und
  robust im Vergleich zur vollen Tile-Cluster-Box.
"""
import numpy

from data_common import boxes_iou


def refine_box_from_saliency(region, saliency_map, size=1.0, min_side=8):
    """
    region: dict mit x0,y0,x1,y1 (Tiles)
    saliency_map: [8,8] ANFIS-Saliency im [0,1] (global)
    -> verfeinerte (x0,y0,x1,y1). Nutzt Massenzentrum + quantil-umschliessende Box.
    """
    # Saliency im Region-Ausschnitt
    i0 = region["x0"] // 16
    i1 = min(7, region["x1"] // 16 - 1)
    j0 = region["y0"] // 16
    j1 = min(7, region["y1"] // 16 - 1)
    patch = saliency_map[j0:j1 + 1, i0:i1 + 1]
    if patch.size == 0 or patch.max() <= 0:
        return (region["x0"], region["y0"], region["x1"], region["y1"])
    # Massenzentrum in Pixel
    ys, xs = numpy.mgrid[j0 * 16:(j1 + 1) * 16, i0 * 16:(i1 + 1) * 16]
    w = patch  # als Gewicht wiederholen je Kachel
    w2 = numpy.repeat(numpy.repeat(w, 16, axis=0), 16, axis=1)
    cy = (ys * w2).sum() / w2.sum()
    cx = (xs * w2).sum() / w2.sum()
    # Hälfte der Seitenlaenge aus der Masse (sqrt(Masse))
    mass = w2.sum()
    side = size * numpy.sqrt(mass)
    side = min(max(side, min_side), region["x1"] - region["x0"])
    hx = side / 2
    x0 = int(round(cx - hx))
    y0 = int(round(cy - hx))
    x1 = int(round(cx + hx))
    y1 = int(round(cy + hx))
    # Clamp an Szenengrenze
    x0 = max(0, x0); y0 = max(0, y0)
    x1 = min(128, x1); y1 = min(128, y1)
    return (x0, y0, x1, y1)


def match_detections(dets, gt_boxes, gt_labels, iou_thr=0.5):
    """
    dets: list of (x0,y0,x1,y1, cls, conf)
    gt_boxes: [K,4]  gt_labels: [K]
    -> (tp, fp, n_gt, matches) mit einem Detzentrum je GT (greedy).
    """
    gt_idx = [k for k in range(len(gt_boxes)) if gt_boxes[k, 0] >= 0]
    assigned = set()
    tp = 0
    for d in dets:
        best_iou, best_k = 0.0, None
        for k in gt_idx:
            iou = boxes_iou(d[:4], gt_boxes[k])
            if iou > best_iou:
                best_iou, best_k = iou, k
        if best_k is not None and best_iou >= iou_thr and best_k not in assigned:
            assigned.add(best_k)
            # Klassenkorrektheit zaehlt fuer die Acc, TP bleibt IoU-basiert
            tp += 1
    fp = len(dets) - tp
    return tp, fp, len(gt_idx), assigned