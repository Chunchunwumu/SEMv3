import torch
from .utils import  find_unmatch_segment_spans
from .teds import TEDS

def match_segment_spans(segments, begins, ends):
    matched_segments = list()
    matched_spans = list()

    for segment_idx, segment in enumerate(segments):
        for span_idx, span in enumerate(zip(begins, ends)):
            # print(span,segment)
            # if  (segment_idx in matched_segments):
            #     break

            if (span_idx not in matched_spans):
                if (segment >= span[0][0]) and (segment < span[1][0]):
                    matched_segments.append(segment_idx)
                    matched_spans.append(span_idx)
                    break

    # print()
    # print(matched_segments)
    # print(matched_spans)
    # print(segments)
    return matched_segments, matched_spans

def cal_segment_pr(point_c, begins, ends):
    correct_nums = 0
    segment_nums = 0
    span_nums = 0
    matched_segments_idx, _ = match_segment_spans(point_c, begins, ends)
        # unmatched_segments_idx = find_unmatch_segment_spans(pred_segments_pi, fg_spans_pi + bg_spans_pi)
    # print(point_c)
    correct_nums += len(matched_segments_idx)
    segment_nums += len(point_c) #  - len(unmatched_segments_idx) 
    span_nums += len(begins)
    # print(correct_nums, segment_nums, span_nums)
    return correct_nums, segment_nums, span_nums


class CellMergeAcc:
    def __call__(self, preds, labels, labels_mask):
        preds = preds & labels_mask
        labels = labels & labels_mask
        flag = preds == labels
        flag = flag.reshape(flag.shape[0], flag.shape[1], -1).min(-1)[0]
        mask = labels.reshape(labels.shape[0], labels.shape[1], -1).max(-1)[0]
        correct_nums = float(torch.sum(flag & mask).detach().cpu().item())
        total_nums = max(float(torch.sum(mask).detach().cpu().item()), 1e-6)
        return correct_nums, total_nums


class AccMetric:
    def __call__(self, preds, labels, labels_mask):
        mask = (labels_mask != 0) & (labels != -1)
        correct_nums = float(torch.sum((preds == labels) & mask).detach().cpu().item())
        total_nums = max(float(torch.sum(mask).detach().cpu().item()), 1e-6)
        return correct_nums, total_nums


def cal_cls_acc(cls_preds, cls_labels):
    mask = (cls_labels != -1)
    total_nums = float(torch.sum(mask).item())
    pred_nums = float(torch.sum((cls_preds == cls_labels) & mask).item())
    return pred_nums, total_nums





class TEDSMetric:
    def __init__(self, num_workers=1, structure_only=False):
        self.evaluator = TEDS(n_jobs=num_workers, structure_only=structure_only)

    def __call__(self, pred_htmls, label_htmls):
        assert len(pred_htmls) == len(label_htmls)
        pred_jsons = {idx: pred_html for idx, pred_html in enumerate(pred_htmls)}
        label_jsons = {idx: dict(html=label_html) for idx, label_html in enumerate(label_htmls)}
        scores = self.evaluator.batch_evaluate(pred_jsons, label_jsons)
        scores = [scores[idx] for idx in range(len(pred_htmls))]
        return scores
