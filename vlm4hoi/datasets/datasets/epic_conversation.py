import os
import re
import json
import pickle
import logging

from PIL import Image
from torch.utils.data import Dataset


class EpicConversationDataset(Dataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_path, question_len, answer_len):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        self.vis_root = vis_root
        self.vis_processor = vis_processor
        self.text_processor = text_processor
        self.question_len = question_len
        self.answer_len = answer_len

        if 'json' in ann_path:
            logging.info('Loading data from json file')
            with open(ann_path, 'r') as f:
                self.ann = json.load(f)
        elif 'pkl' in ann_path:
            logging.info('Loading data from pkl file')
            self.ann = pickle.load(open(ann_path, 'rb'))
        else:
            raise ValueError(f'Unknown annotation file type: {ann_path}')

        self.connect_sym = "!@#"

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        info = self.ann[index]

        image_path = os.path.join(self.vis_root, f"{info['image']}.jpg")
        assert os.path.exists(image_path), f"{image_path} does not exist"
        image = Image.open(image_path).convert("RGB")
        image = self.vis_processor(image)

        first_instruction = info['conversations'][0]['value'].replace('<image>', '').replace('\n', '').strip()
        first_instruction = '<Img><ImageHere></Img> {} '.format(first_instruction)

        # Here we assume that the bounding boxes are already normalised to (0, 100)
        questions = [first_instruction]
        answers = []

        for i, item in enumerate(info["conversations"][1:]):
            if i % 2 ==0:  # assistant
                assistant_answer = item["value"]
                answers.append(assistant_answer)
            else:
                human_instruction = item["value"]+" "
                questions.append(human_instruction)

        questions = self.connect_sym.join(questions)
        answers = self.connect_sym.join(answers)

        if len(answers) > self.answer_len:
            logging.warning(f'Warning: epic answer length more than {self.answer_len}, truncating the data')
            answers = answers[:self.answer_len]
        if len(questions) > self.question_len:
            logging.warning(f'Warning: epic question length more than {self.question_len}, truncating the data')
            questions = questions[:self.question_len]
        
        return {
            "image": image,
            "conv_q": questions,
            'conv_a': answers,
            "image_id": info['id'],
            "connect_sym": self.connect_sym
        }


class EPICEvalDataHOI(Dataset):
    def __init__(self, qa_list, vis_processor):
        self.qa_list = qa_list
        self.vis_processor = vis_processor

    def __len__(self):
        return len(self.qa_list)
    
    def __getitem__(self, idx):
        data = self.qa_list[idx]
        image_path = data['image_path']
        image = Image.open(image_path).convert('RGB')
        image = self.vis_processor(image)
        question = data['question']
        pattern = r'\[.*?\] '
        question = re.sub(pattern, '', question)
        id = data['id']
        return image, question, id


class EPICEvalDataREF(Dataset):
    def __init__(self, loaded_json, vis_processor):
        self.loaded_json = loaded_json
        self.vis_processor = vis_processor
        self.keys = list(self.loaded_json.keys())

    def __len__(self):
        return len(self.keys)
    
    def __getitem__(self, idx):
        key = self.keys[idx]
        data = self.loaded_json[key]
        image_path = data['image_path']
        image = Image.open(image_path).convert('RGB')
        image = self.vis_processor(image)
        noun = data['noun']
        question = f"Where is the {noun}"
        id = key
        return image, question, id


class EPICEvalDataIDEN(Dataset):
    def __init__(self, loaded_json, vis_processor, image_original_size=(1920, 1080)):
        self.loaded_json = loaded_json
        self.vis_processor = vis_processor
        self.keys = list(self.loaded_json.keys())
        self.image_original_size = image_original_size

    def __len__(self):
        return len(self.keys)
    
    def bb_normalise(
            self,
            bbox,
            image_original_size=(1920, 1080),
            image_new_size=(100, 100),
            ):
        """
        This function normalises the bounding box coordinates to the new image size.
        It is based on the minigpt-v2 implementation. Where the authors hardcode the
        new image size to 100x100.
        For Ego4D dataset, the image size is 1440x1080.
        For EPIC-KITCHENS-100 dataset, the image size is 456x256. This is because of
        the resizing done when generating the HOI information.
        For VISOR dataset, the image size is 1920x1080.
        """
        image_original_size = self.image_original_size
        if type(bbox[0]) is str:
            bbox = [float(x) for x in bbox]
        normalised_bbox = [
            bbox[0] / image_original_size[0] * image_new_size[0],
            bbox[1] / image_original_size[1] * image_new_size[1],
            bbox[2] / image_original_size[0] * image_new_size[0],
            bbox[3] / image_original_size[1] * image_new_size[1]
        ]
        return [int(x) for x in normalised_bbox]
    
    def __getitem__(self, idx):
        key = self.keys[idx]
        data = self.loaded_json[key]
        # As VISOR annotations are done on the image size of 1920x1080, we can
        # hardcode and obtain the bouding box coordinates for the new image size
        bbox = self.bb_normalise(data['bbox'])
        image_path = data['image_path']
        image = Image.open(image_path).convert('RGB')
        image = self.vis_processor(image)
        question = "{{<{}><{}><{}><{}>}}".format(*bbox)
        id = key
        return image, question, id
