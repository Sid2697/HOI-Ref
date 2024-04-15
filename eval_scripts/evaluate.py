import os
import json
import argparse
from tqdm import tqdm

from helper import calculate_iou, extract_bounding_boxes, bb_normalize, multiple_boxes_iou


parser = argparse.ArgumentParser()
parser.add_argument(
    '--test_json',
    type=str,
    default='hoiqa_dataset/epic-test.json',
    help='Path to the test JSON file',
)
parser.add_argument(
    '--pred_json',
    type=str,
    default='/path/to/saved/predictions.json',
    help='Path to the predictions JSON file',
)
parser.add_argument(
    '--hoi_json',
    type=str,
    default='hoiqa_dataset/epic-visor-test.json',
    help='Path to the HOI JSON file',
)
parser.add_argument(
    '--hoi_pred_json',
    type=str,
    default='/path/to/saved/predictions_hoi.json',
    help='Path to the HOI predictions JSON file',
)
parser.add_argument(
    '--ego4d_test_data',
    type=str,
    default='hoiqa_dataset/ego4d-test.json',
    help='Path to the Ego4D test JSON file',
)
parser.add_argument(
    '--iou_thresh',
    type=float,
    default=0.5,
    help='IOU threshold value to select the bounding box as correct or incorrect',
)
parser.add_argument(
    '--new_width',
    type=int,
    default=1920,
    help='New width of the image',
)
parser.add_argument(
    '--new_height',
    type=int,
    default=1080,
    help='New height of the image',
)
args = parser.parse_args()

assert os.path.isfile(args.test_json), f"Wrong test JSON path: {args.test_json}"
assert os.path.isfile(args.pred_json), f"Wrong predictions JSON path: {args.pred_json}"
assert os.path.isfile(args.hoi_json), f"Wrong HOI JSON path: {args.hoi_json}"
assert os.path.isfile(args.hoi_pred_json), f"Wrong HOI predictions JSON path: {args.hoi_pred_json}"
assert 0 <= args.iou_thresh <= 1, f'Invalid IoU threshold: {args.iou_thresh}'
ego4d_pred_path = args.pred_json.replace('.json', '_ego4d.json')
assert os.path.isfile(ego4d_pred_path), f"Wrong Ego4D predictions JSON path: {ego4d_pred_path}"

test_data = json.load(open(args.test_json, 'r'))
pred_data = json.load(open(args.pred_json, 'r'))
hoi_test_data = json.load(open(args.hoi_json, 'r'))
ego4d_test_data = json.load(open(args.ego4d_test_data, 'r'))
ego4d_pred_data = json.load(open(ego4d_pred_path, 'r'))
hoi_pred_data = json.load(open(args.hoi_pred_json, 'r'))
assert len(test_data) == len(pred_data), "Length of test and predictions JSON files are not equal"
assert len(hoi_test_data) == len(hoi_pred_data), "Length of HOI test and predictions JSON files are not equal"

test_keys = list(test_data.keys())
hoi_test_keys = list(hoi_pred_data.keys())
ego4d_test_keys = list(ego4d_test_data.keys())

hoi_noun_correct = 0
hoi_noun_wrong = 0
hoi_noun_total = 0
hoi_bbox_correct = 0
hoi_bbox_total = 0
hoi_bbox_wrong = 0
hoi_wrong_bb_cases = 0
hoi_iou_tracker = list()

# Evaluating the benchmarks
i_ref_bbox_total = 0
i_ref_noun_total = 0
ho_ref_bbox_total = 0
ho_ref_noun_total = 0
i_ref_bbox_correct = 0
i_ref_noun_correct = 0
ho_ref_bbox_correct = 0
ho_ref_noun_correct = 0
i_ref_iou_score = list()
ho_ref_iou_score = list()


def eval(test_data, pred_data, test_keys, width=args.new_width, height=args.new_height):
    noun_correct = 0
    noun_wrong = 0
    noun_total = 0
    bbox_correct = 0
    bbox_total = 0
    bbox_wrong = 0
    iou_tracker = list()
    wrong_bb_cases = 0
    # Evaluating the subsets in the benchmark
    objN2objBB_total = 0
    objN2objBB_correct = 0
    handN2handBB_total = 0
    handN2handBB_correct = 0
    objBB2objN_total = 0
    objBB2objN_correct = 0
    handBB2handN_total = 0
    handBB2handN_correct = 0
    subset_dict = dict()
    for key in tqdm(test_keys, desc='VISOR Evaluation'):
        gt = test_data[key]
        pred = pred_data[key]
        # Checking for noun
        gt_noun = gt['noun']
        pred_noun = pred['iden_answer']
        noun_total += 1
        noun_present = False
        if gt_noun == 'right hand':
            if 'right hand' in pred_noun.lower().replace('.', ''):
                noun_correct += 1
                noun_present = True
                handBB2handN_correct += 1
            handBB2handN_total += 1
        elif gt_noun == 'left hand':
            if 'left hand' in pred_noun.lower().replace('.', ''):
                noun_correct += 1
                noun_present = True
                handBB2handN_correct += 1
            handBB2handN_total += 1
        else:
            for item in pred_noun.lower().replace('.', '').split(' '):
                if item == gt_noun:
                    noun_correct += 1
                    noun_present = True
                    objBB2objN_correct += 1
                    break
            objBB2objN_total += 1
        if not noun_present:
            noun_wrong += 1
        # Checking for bounding box
        gt_bbox = gt['bbox'] # NOTE: There are no multiple bounding boxes in the test data
        pred_bbox = pred['ref_answer']
        parsed_pred_bbox = extract_bounding_boxes(pred_bbox)
        bbox_total += 1
        multiple_bboxes = False
        hand_bbox = False
        if gt_noun == 'right hand' or gt_noun == 'left hand':
            handN2handBB_total += 1
            hand_bbox = True
        else:
            objN2objBB_total += 1
        if len(parsed_pred_bbox) > 1:
            multiple_bboxes = True
        elif len(parsed_pred_bbox) == 1:
            parsed_pred_bbox = parsed_pred_bbox[0]
        else:
            bbox_wrong += 1
            wrong_bb_cases += 1
            continue
        assert len(gt_bbox) == 4, f"Ground truth has more than 1 BB. {gt_bbox}"
        if multiple_bboxes:
            iou_score = max(multiple_boxes_iou([gt_bbox], parsed_pred_bbox))
        else:
            parsed_pred_bbox = bb_normalize(
                parsed_pred_bbox,
                image_original_size=(100, 100),
                image_new_size=(width, height),
            )
            assert len(parsed_pred_bbox) == 4, f"Predictions has more than 1 BB. {parsed_pred_bbox}"
            iou_score = calculate_iou(gt_bbox, parsed_pred_bbox)
        iou_tracker.append(iou_score)
        if iou_score > args.iou_thresh:
            bbox_correct += 1
            if hand_bbox:
                handN2handBB_correct += 1
            else:
                objN2objBB_correct += 1
        else:
            bbox_wrong += 1
    bbox_accuracy = round((bbox_correct/bbox_total)*100, 2)
    noun_accuracy = round((noun_correct/noun_total)*100, 2)
    subset_dict['handBB2handN'] = {
        'accuracy': round((handBB2handN_correct/handBB2handN_total)*100, 2),
        'correct': handBB2handN_correct,
        'total': handBB2handN_total,
    }
    subset_dict['objBB2objN'] = {
        'accuracy': round((objBB2objN_correct/objBB2objN_total)*100, 2),
        'correct': objBB2objN_correct,
        'total': objBB2objN_total,
    }
    subset_dict['objN2objBB'] = {
        'accuracy': round((objN2objBB_correct/objN2objBB_total)*100, 2),
        'correct': objN2objBB_correct,
        'total': objN2objBB_total,
    }
    subset_dict['handN2handBB'] = {
        'accuracy': round((handN2handBB_correct/handN2handBB_total)*100, 2),
        'correct': handN2handBB_correct,
        'total': handN2handBB_total,
    }
    print(f'Noun Accuracy: {noun_accuracy}; {noun_correct}/{noun_total}')
    print(f'BBox Accuracy: {bbox_accuracy}; {bbox_correct}/{bbox_total}')
    print(f'Mean IoU Score: {round(sum(iou_tracker)/len(iou_tracker)*100, 2)}')
    print(f'Wrong BB cases: {wrong_bb_cases}')
    return iou_tracker, bbox_accuracy, noun_accuracy, noun_correct, noun_total, bbox_correct, bbox_total, subset_dict

epic_iou, epic_bbox_acc, epic_noun_acc, epic_noun_correct, epic_noun_total, epic_bbox_correct, epic_bbox_total, epic_subset_dict = eval(test_data, pred_data, test_keys, width=1920, height=1080)
ego4d_iou, ego4d_bbox_acc, ego4d_noun_acc, ego4d_noun_correct, ego4d_noun_total, ego4d_bbox_correct, ego4d_bbox_total, ego4d_subset_dict = eval(ego4d_test_data, ego4d_pred_data, ego4d_test_keys, width=1440, height=1080)

ho_ref_noun_total = epic_noun_total + ego4d_noun_total
ho_ref_bbox_total = epic_bbox_total + ego4d_bbox_total
ho_ref_noun_correct = epic_noun_correct + ego4d_noun_correct
ho_ref_bbox_correct = epic_bbox_correct + ego4d_bbox_correct
ho_ref_iou_score = epic_iou + ego4d_iou

ho_ref_question_list = ['[refer] Where is the right hand of the person?', '[refer] Where is the left hand of the person?', '[refer] Where are the hands of the person?']

handN2objBB_total = 0
handN2objBB_correct = 0
objN2handBB_total = 0
objN2handBB_correct = 0
objN2handN_total = 0
objN2handN_correct = 0
handN2objN_total = 0
handN2objN_correct = 0

for key in tqdm(hoi_test_keys, desc='VISOR HOI Evaluation'):
    for item in hoi_test_data:
        if item['id'] == key:
            gt = item['answer']
            question = item['question']
            break
    pred = hoi_pred_data[key]['answer']
    ho_ref_question = False
    i_ref_question = False
    handN2objN_flag = False
    objN2handN_flag = False
    objN2handBB_flag = False
    handN2objBB_flag = False
    if '[refer]' in question:
        if question in ho_ref_question_list:
            ho_ref_question = True
            ho_ref_bbox_total += 1
            epic_subset_dict['handN2handBB']['total'] += 1
        else:
            i_ref_bbox_total += 1
            i_ref_question = True
        if '[refer] Locate the object being manipulated by' in question:
            handN2objBB_total += 1
            handN2objBB_flag = True
        elif '[refer] Which hand has the' in question:
            objN2handBB_total += 1
            objN2handBB_flag = True
        else:
            if not ho_ref_question:
                raise ValueError(f"Invalid question: {question}")
        multiple_bb = False
        if len(gt) > 4:
            multiple_bb = True
            parsed_gt_bbox = extract_bounding_boxes(gt)
            assert len(parsed_gt_bbox) == 2, f"Multiple BBs but not 2. {gt} {key}"
        # Checking for bounding box
        parsed_pred_bbox = extract_bounding_boxes(pred)
        hoi_bbox_total += 1
        if multiple_bb:
            hoi_bbox_total += 1
            if ho_ref_question:
                ho_ref_bbox_total += 1
                epic_subset_dict['handN2handBB']['total'] += 1
            elif i_ref_question:
                i_ref_bbox_total += 1
            else:
                raise ValueError(f"Invalid question: {question}")
            if handN2objBB_flag:
                handN2objBB_total += 1
            if objN2handBB_flag:
                objN2handBB_total += 1
        if len(parsed_pred_bbox) > 1 and not multiple_bb:
            # Multiple bounding boxes are present for single bounding box question
            iou_score = max(multiple_boxes_iou([gt], parsed_pred_bbox))
            hoi_iou_tracker.append(iou_score)
            if ho_ref_question:
                ho_ref_iou_score.append(iou_score)
            elif i_ref_question:
                i_ref_iou_score.append(iou_score)
            else:
                raise ValueError(f"Invalid question: {question}")
            if iou_score > args.iou_thresh:
                hoi_bbox_correct += 1
                if ho_ref_question:
                    ho_ref_bbox_correct += 1
                    epic_subset_dict['handN2handBB']['correct'] += 1
                elif i_ref_question:
                    i_ref_bbox_correct += 1
                else:
                    raise ValueError(f"Invalid question: {question}")
                if handN2objBB_flag:
                    handN2objBB_correct += 1
                if objN2handBB_flag:
                    objN2handBB_correct += 1
            else:
                hoi_bbox_wrong += 1
        elif len(parsed_pred_bbox) > 1 and multiple_bb:
            # Multiple bounding boxes are present for multiple bounding box question
            if len(parsed_pred_bbox) == 2:
                # We want only two predicted bounding boxes
                # calculate the IOU for the nearest pairs of bounding boxes
                multiple_iou = multiple_boxes_iou(parsed_gt_bbox, parsed_pred_bbox)
                assert len(multiple_iou) == 4, f"More than 4 IOU scores. {multiple_iou}"
                first_gt_bbox_iou = max(multiple_iou[:2])
                hoi_iou_tracker.append(first_gt_bbox_iou)
                if ho_ref_question:
                    ho_ref_iou_score.append(first_gt_bbox_iou)
                elif i_ref_question:
                    i_ref_iou_score.append(first_gt_bbox_iou)
                else:
                    raise ValueError(f"Invalid question: {question}")
                second_gt_bbox_iou = max(multiple_iou[2:4])
                hoi_iou_tracker.append(second_gt_bbox_iou)
                if ho_ref_question:
                    ho_ref_iou_score.append(second_gt_bbox_iou)
                elif i_ref_question:
                    i_ref_iou_score.append(second_gt_bbox_iou)
                else:
                    raise ValueError(f"Invalid question: {question}")
                if second_gt_bbox_iou > args.iou_thresh and first_gt_bbox_iou > args.iou_thresh:
                    hoi_bbox_correct += 2
                    if ho_ref_question:
                        ho_ref_bbox_correct += 2
                        epic_subset_dict['handN2handBB']['correct'] += 2
                    elif i_ref_question:
                        i_ref_bbox_correct += 2
                    else:
                        raise ValueError(f"Invalid question: {question}")
                    if handN2objBB_flag:
                        handN2objBB_correct += 2
                    if objN2handBB_flag:
                        objN2handBB_correct += 2
                else:
                    hoi_bbox_wrong += 2
            else:
                hoi_bbox_wrong += 2
                print(f"More than 2 bounding boxes present. {parsed_pred_bbox}")
        elif len(parsed_pred_bbox) == 1 and not multiple_bb:
            # Single bounding box is present for single bounding box question
            parsed_pred_bbox = parsed_pred_bbox[0]
            parsed_pred_bbox = bb_normalize(parsed_pred_bbox, image_original_size=(100, 100), image_new_size=(1920, 1080))
            assert len(gt) == 4, f"Ground truth has more than 1 BB. {gt} {key}"
            assert len(parsed_pred_bbox) == 4, f"Predictions has more than 1 BB. {parsed_pred_bbox}"
            iou_score = calculate_iou(gt, parsed_pred_bbox)
            hoi_iou_tracker.append(iou_score)
            if ho_ref_question:
                ho_ref_iou_score.append(iou_score)
            elif i_ref_question:
                i_ref_iou_score.append(iou_score)
            else:
                raise ValueError(f"Invalid question: {question}")
            if iou_score > args.iou_thresh:
                hoi_bbox_correct += 1
                if ho_ref_question:
                    ho_ref_bbox_correct += 1
                    epic_subset_dict['handN2handBB']['correct'] += 1
                elif i_ref_question:
                    i_ref_bbox_correct += 1
                else:
                    raise ValueError(f"Invalid question: {question}")
                if handN2objBB_flag:
                    handN2objBB_correct += 1
                if objN2handBB_flag:
                    objN2handBB_correct += 1
            else:
                hoi_bbox_wrong += 1
        elif len(parsed_pred_bbox) == 1 and multiple_bb:
            # # Single bounding box is present for multiple bounding box question
            # multiple_iou = multiple_boxes_iou(parsed_gt_bbox, parsed_pred_bbox)
            # # As it didn't predict one box, we will consider the entire thing to be wrong
            hoi_bbox_wrong += 2
            hoi_iou_tracker.append(0)
            hoi_iou_tracker.append(0)
            if ho_ref_question:
                ho_ref_iou_score.append(0)
                ho_ref_iou_score.append(0)
            elif i_ref_question:
                i_ref_iou_score.append(0)
                i_ref_iou_score.append(0)
            else:
                raise ValueError(f"Invalid question: {question}")
        else:
            hoi_bbox_wrong += 1
            hoi_wrong_bb_cases += 1
            hoi_iou_tracker.append(0)
            if ho_ref_question:
                ho_ref_iou_score.append(0)
            elif i_ref_question:
                i_ref_iou_score.append(0)
            else:
                raise ValueError(f"Invalid question: {question}")
            continue
    elif '[vqa]' in question:
        # Checking for noun
        hoi_noun_total += 1
        noun_present = False
        if '[vqa] What hand is holding the' in question:
            objN2handN_total += 1
            objN2handN_flag = True
        elif '[vqa] What is the object in the ' in question:
            handN2objN_total += 1
            handN2objN_flag = True
        if gt == 'right hand':
            if 'right hand' in pred.lower().replace('.', ''):
                hoi_noun_correct += 1
                noun_present = True
                if objN2handN_flag:
                    objN2handN_correct += 1
                if handN2objN_flag:
                    handN2objN_correct += 1
        elif gt == 'left hand':
            if 'left hand' in pred.lower().replace('.', ''):
                hoi_noun_correct += 1
                noun_present = True
                if objN2handN_flag:
                    objN2handN_correct += 1
                if handN2objN_flag:
                    handN2objN_correct += 1
        elif gt == 'right glove' or gt == 'left glove':
            if 'glove' in pred.lower().replace('.', ''):
                hoi_noun_correct += 1
                noun_present = True
                if objN2handN_flag:
                    objN2handN_correct += 1
                if handN2objN_flag:
                    handN2objN_correct += 1
        else:
            for item in pred.lower().replace('.', '').split(' '):
                if item == gt:
                    hoi_noun_correct += 1
                    noun_present = True
                    if objN2handN_flag:
                        objN2handN_correct += 1
                    if handN2objN_flag:
                        handN2objN_correct += 1
                    break
            # print(gt, pred, noun_present)
        if not noun_present:
            hoi_noun_wrong += 1
    else:
        raise ValueError(f"Invalid question: {question}")
print(f'HOI Noun Accuracy: {round((hoi_noun_correct/hoi_noun_total)*100, 2)}; {hoi_noun_correct}/{hoi_noun_total}')
print(f'HOI BBox Accuracy: {round((hoi_bbox_correct/hoi_bbox_total)*100, 2)}; {hoi_bbox_correct}/{hoi_bbox_total}')
print(f'HOI Mean IoU Score: {round(sum(hoi_iou_tracker)/len(hoi_iou_tracker)*100, 2)}')
print(f'HOI Wrong BB cases: {hoi_wrong_bb_cases}')

final_iou_list = epic_iou + hoi_iou_tracker + ego4d_iou
total_noun = epic_noun_total + hoi_noun_total + ego4d_noun_total
total_noun_pred = epic_noun_correct + hoi_noun_correct + ego4d_noun_correct
total_bbox = epic_bbox_total + hoi_bbox_total + ego4d_bbox_total
total_bbox_pred = epic_bbox_correct + hoi_bbox_correct + ego4d_bbox_correct
final_bbox_acc = round((total_bbox_pred/total_bbox)*100, 2)
final_noun_acc = round((total_noun_pred/total_noun)*100, 2)
final_iou = round(sum(final_iou_list)/len(final_iou_list)*100, 2)
print(f'Final Noun Accuracy: {final_noun_acc}; {total_noun_pred}/{total_noun}')
print(f'Final BBox Accuracy: {final_bbox_acc}; {total_bbox_pred}/{total_bbox}')
print(f'Final Mean IoU Score: {final_iou}; {len(final_iou_list)}')

i_ref_noun_total = hoi_noun_total
i_ref_noun_correct = hoi_noun_correct
print('-'*50)
print(f'ho_ref Noun Accuracy: {round((ho_ref_noun_correct/ho_ref_noun_total)*100, 2)}; {ho_ref_noun_correct}/{ho_ref_noun_total}')
print(f'ho_ref BBox Accuracy: {round((ho_ref_bbox_correct/ho_ref_bbox_total)*100, 2)}; {ho_ref_bbox_correct}/{ho_ref_bbox_total}')
print(f'ho_ref Mean IoU Score: {round(sum(ho_ref_iou_score)/len(ho_ref_iou_score)*100, 2)}')
print(f'i_ref Noun Accuracy: {round((i_ref_noun_correct/i_ref_noun_total)*100, 2)}; {i_ref_noun_correct}/{i_ref_noun_total}')
print(f'i_ref BBox Accuracy: {round((i_ref_bbox_correct/i_ref_bbox_total)*100, 2)}; {i_ref_bbox_correct}/{i_ref_bbox_total}')
print(f'i_ref Mean IoU Score: {round(sum(i_ref_iou_score)/len(i_ref_iou_score)*100, 2)}')

print('-'*50)
print('Subset Evaluation')
handBB2handN_total_final = epic_subset_dict['handBB2handN']['total'] + ego4d_subset_dict['handBB2handN']['total']
handBB2handN_correct_final = epic_subset_dict['handBB2handN']['correct'] + ego4d_subset_dict['handBB2handN']['correct']
handBB2handN_acc = round((handBB2handN_correct_final/handBB2handN_total_final)*100, 2)

objBB2objN_total_final = epic_subset_dict['objBB2objN']['total'] + ego4d_subset_dict['objBB2objN']['total']
objBB2objN_correct_final = epic_subset_dict['objBB2objN']['correct'] + ego4d_subset_dict['objBB2objN']['correct']
objBB2objN_acc = round((objBB2objN_correct_final/objBB2objN_total_final)*100, 2)

objN2objBB_total_final = epic_subset_dict['objN2objBB']['total'] + ego4d_subset_dict['objN2objBB']['total']
objN2objBB_correct_final = epic_subset_dict['objN2objBB']['correct'] + ego4d_subset_dict['objN2objBB']['correct']
objN2objBB_acc = round((objN2objBB_correct_final/objN2objBB_total_final)*100, 2)

handN2handBB_total_final = epic_subset_dict['handN2handBB']['total'] + ego4d_subset_dict['handN2handBB']['total']
handN2handBB_correct_final = epic_subset_dict['handN2handBB']['correct'] + ego4d_subset_dict['handN2handBB']['correct']
handN2handBB_acc = round((handN2handBB_correct_final/handN2handBB_total_final)*100, 2)

handN2objbb_acc = round((handN2objBB_correct/handN2objBB_total)*100, 2)
objN2handbb_acc = round((objN2handBB_correct/objN2handBB_total)*100, 2)
objN2handN_acc = round((objN2handN_correct/objN2handN_total)*100, 2)
handN2objN_acc = round((handN2objN_correct/handN2objN_total)*100, 2)

print(f'handBB2handN Accuracy: {handBB2handN_acc}; {handBB2handN_correct_final}/{handBB2handN_total_final}')
print(f'objBB2objN Accuracy: {objBB2objN_acc}; {objBB2objN_correct_final}/{objBB2objN_total_final}')
print(f'objN2objBB Accuracy: {objN2objBB_acc}; {objN2objBB_correct_final}/{objN2objBB_total_final}')
print(f'handN2handBB Accuracy: {handN2handBB_acc}; {handN2handBB_correct_final}/{handN2handBB_total_final}')
print(f'handN2objBB Accuracy: {handN2objbb_acc}; {handN2objBB_correct}/{handN2objBB_total}')
print(f'objN2handBB Accuracy: {objN2handbb_acc}; {objN2handBB_correct}/{objN2handBB_total}')
print(f'objN2handN Accuracy: {objN2handN_acc}; {objN2handN_correct}/{objN2handN_total}')
print(f'handN2objN Accuracy: {handN2objN_acc}; {handN2objN_correct}/{handN2objN_total}')
