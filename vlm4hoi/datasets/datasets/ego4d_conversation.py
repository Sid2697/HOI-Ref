import os
import json

from PIL import Image
from torch.utils.data import Dataset


class Ego4DConversationDataset(Dataset):
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

        with open(ann_path, 'r') as f:
            self.ann = json.load(f)

        self.connect_sym = "!@#"

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        info = self.ann[index]

        image_path = info['image_path']
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
            print(f'Warning: ego4d answer length more than {self.answer_len}, truncating the data')
            answers = answers[:self.answer_len]
        if len(questions) > self.question_len:
            print(f'Warning: ego4d question length more than {self.question_len}, truncating the data')
            questions = questions[:self.question_len]

        return {
            "image": image,
            "conv_q": questions,
            'conv_a': answers,
            "image_id": info['id'],
            "connect_sym": self.connect_sym
        }
