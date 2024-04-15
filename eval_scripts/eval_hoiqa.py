
import os
import json
from tqdm import tqdm

from torch.utils.data import DataLoader
from vlm4hoi.common.config import Config
from vlm4hoi.conversation.conversation import CONV_VISION_vlm4hoi
from vlm4hoi.common.eval_utils import eval_parser, init_model, prepare_texts
from vlm4hoi.datasets.datasets.epic_conversation import EPICEvalDataREF, EPICEvalDataIDEN, EPICEvalDataHOI


parser = eval_parser()
parser.add_argument(
    '--gt_json',
    type=str,
    default='hoiqa_dataset/epic-test.json',
    help='Path to the test JSON file',
)
parser.add_argument(
    '--ego4d_gt_json',
    type=str,
    default='hoiqa_dataset/ego4d-test.json',
    help='Path to the test JSON file for Ego4D dataset',
)
parser.add_argument(
    '--pred_json',
    type=str,
    default='/path/to/save/predictions.json',
    help='Path to save the predictions',
)
parser.add_argument(
    '--eval_batch_size',
    type=int,
    default=32,
    help='Batch size for evaluation',
)
parser.add_argument(
    '--hoi_gt_json',
    type=str,
    default='hoiqa_dataset/epic-visor-test.json',
    help='Path to the test JSON file for VISOR portion of EPIC-Kitchens dataset',
)
args = parser.parse_args()
cfg = Config(args)
print(args)

assert os.path.exists(args.gt_json), f'GT JSON file not found at {args.gt_json}'
assert os.path.exists(args.hoi_gt_json), f'GT JSON file not found at {args.hoi_gt_json}'
assert os.path.exists(args.ego4d_gt_json), f'GT JSON file not found at {args.ego4d_gt_json}'

# Output file paths
assert not os.path.exists(args.pred_json), f'Predictions JSON file already exists at {args.pred_json}'
hoi_json_save_path = args.pred_json.replace('.json', '_hoi.json')
assert not os.path.exists(hoi_json_save_path), f'Predictions JSON file already exists at {hoi_json_save_path}'
ego_4d_json_save_path = args.pred_json.replace('.json', '_ego4d.json')
assert not os.path.exists(ego_4d_json_save_path), f'Predictions JSON file already exists at {ego_4d_json_save_path}'

model, vis_processor = init_model(args)
model.eval()

CONV_VISION = CONV_VISION_vlm4hoi
conv_temp = CONV_VISION.copy()
conv_temp.system = ""

print('Preparing DataLoaders...')
gt_data = json.load(open(args.gt_json, 'r'))
ref_data = EPICEvalDataREF(gt_data, vis_processor)
ref_eval_dataloader = DataLoader(
    ref_data,
    batch_size=args.eval_batch_size,
    shuffle=False,
    num_workers=10,
)

ego4d_data = json.load(open(args.ego4d_gt_json, 'r'))
ego4d_ref_data = EPICEvalDataREF(ego4d_data, vis_processor)
ego4d_ref_eval_dataloader = DataLoader(
    ego4d_ref_data,
    batch_size=int(args.eval_batch_size/2),
    shuffle=False,
    num_workers=10,
)

iden_data = EPICEvalDataIDEN(gt_data, vis_processor)
iden_eval_dataloader = DataLoader(
    iden_data,
    batch_size=args.eval_batch_size,
    shuffle=False,
    num_workers=10,
)

ego4d_iden_data = EPICEvalDataIDEN(ego4d_data, vis_processor, image_original_size=(1440, 1080))
ego4d_iden_eval_dataloader = DataLoader(
    ego4d_iden_data,
    batch_size=int(args.eval_batch_size/2),
    shuffle=False,
    num_workers=10,
)

hoi_gt_data = json.load(open(args.hoi_gt_json, 'r'))
hoi_data = EPICEvalDataHOI(hoi_gt_data, vis_processor)
hoi_eval_dataloader = DataLoader(
    hoi_data,
    batch_size=args.eval_batch_size,
    shuffle=False,
    num_workers=10,
)

print('Starting evaluation for hoi...')
hoi_pred_dict = dict()
for image, question, id in tqdm(hoi_eval_dataloader):
    texts = prepare_texts(question, conv_temp)
    answers = model.generate(image, texts, max_new_tokens=50, do_sample=False)
    for i in range(len(answers)):
        assert id[i] not in hoi_pred_dict.keys(), f'Duplicate ID {id[i]}'
        hoi_pred_dict[id[i]] = {
            "question": question[i],
            "answer": answers[i],
        }
json.dump(hoi_pred_dict, open(hoi_json_save_path, 'w'), indent=4)
print(f'HOI Predictions saved to {args.pred_json.replace(".json", "_hoi.json")}')
print('-'*50)

print('Starting evaluation for ref...')
pred_dict = dict()
for image, question, id in tqdm(ref_eval_dataloader):
    texts = prepare_texts(question, conv_temp)
    answers = model.generate(image, texts, max_new_tokens=50, do_sample=False)
    for i in range(len(answers)):
        assert id[i] not in pred_dict.keys(), f'Duplicate ID {id}'
        pred_dict[id[i]] = {
            "question": question[i],
            "ref_answer": answers[i],
            "iden_answer": "",
        }

print('Starting evaluation for iden...')
for image, question, id in tqdm(iden_eval_dataloader):
    texts = prepare_texts(question, conv_temp)
    answers = model.generate(image, texts, max_new_tokens=50, do_sample=False)
    for i in range(len(answers)):
        pred_dict[id[i]]["iden_answer"] = answers[i]

json.dump(pred_dict, open(args.pred_json, 'w'), indent=4)
print(f'EPIC-Kitchens Predictions saved to {args.pred_json}')
print('-'*50)

print('Starting evaluation on Ego4D dataset...')
print('-'*50)
print('Starting evaluation for ego4d ref...')
ego4d_pred_dict = dict()
for image, question, id in tqdm(ego4d_ref_eval_dataloader):
    texts = prepare_texts(question, conv_temp)
    answers = model.generate(image, texts, max_new_tokens=50, do_sample=False)
    for i in range(len(answers)):
        assert id[i] not in ego4d_pred_dict.keys(), f'Duplicate ID {id[i]}'
        ego4d_pred_dict[id[i]] = {
            "question": question[i],
            "ref_answer": answers[i],
            "iden_answer": "",
        }

json.dump(ego4d_pred_dict, open(ego_4d_json_save_path, 'w'), indent=4)
print(f'Ego4D Predictions saved to {ego_4d_json_save_path}')

print('Starting evaluation for ego4d iden...')
for image, question, id in tqdm(ego4d_iden_eval_dataloader):
    texts = prepare_texts(question, conv_temp)
    answers = model.generate(image, texts, max_new_tokens=50, do_sample=False)
    for i in range(len(answers)):
        ego4d_pred_dict[id[i]]["iden_answer"] = answers[i]

json.dump(ego4d_pred_dict, open(ego_4d_json_save_path, 'w'), indent=4)
print(f'Ego4D Predictions saved to {ego_4d_json_save_path}')
