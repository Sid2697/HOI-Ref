import logging

from vlm4hoi.common.registry import registry
from vlm4hoi.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from vlm4hoi.datasets.datasets.epic_conversation import EpicConversationDataset
from vlm4hoi.datasets.datasets.ego4d_conversation import Ego4DConversationDataset


@registry.register_builder("ego4d_conversation")
class Ego4DConversationBuilder(BaseDatasetBuilder):
    train_dataset_cls = Ego4DConversationDataset
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/ego4d_conversation/default.yaml",
    }

    def build_datasets(self):
        logging.info("Building datasets...")
        self.build_processors()

        build_info = self.config.build_info
        datasets = dict()

        # create datasets
        dataset_cls = self.train_dataset_cls
        datasets['train'] = dataset_cls(
            vis_processor=self.vis_processors["train"],
            text_processor=self.text_processors["train"],
            vis_root=build_info.image_path,
            ann_path=build_info.ann_path,
            question_len=build_info.question_len,
            answer_len=build_info.answer_len,
        )
        return datasets


@registry.register_builder("epic_conversation")
class EpicConversationBuilder(BaseDatasetBuilder):
    train_dataset_cls = EpicConversationDataset
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/epic_conversation/default.yaml",
    }

    def build_datasets(self):
        logging.info("Building datasets...")
        self.build_processors()

        build_info = self.config.build_info
        datasets = dict()

        # create datasets
        dataset_cls = self.train_dataset_cls
        datasets['train'] = dataset_cls(
            vis_processor=self.vis_processors["train"],
            text_processor=self.text_processors["train"],
            vis_root=build_info.image_path,
            ann_path=build_info.ann_path,
            question_len=build_info.question_len,
            answer_len=build_info.answer_len,
        )
        return datasets
