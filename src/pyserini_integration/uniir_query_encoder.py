import yaml
import random
from typing import Optional

import pandas as pd
from torch.utils.data import DataLoader

from uniir_for_pyserini.pyserini_integration.uniir_base_encoder import UniIRBaseEncoder
from uniir_for_pyserini.pyserini_integration.mbeir_datasets import MBEIRQueryDataset
from uniir_for_pyserini.data.mbeir_dataset import MBEIRInferenceOnlyCollator
from uniir_for_pyserini.common.mbeir_embedder import generate_embeds_and_ids_for_dataset_with_gather
from uniir_for_pyserini.data.preprocessing.utils import format_string, hash_qid


class QueryEncoder(UniIRBaseEncoder):
    def __init__(
        self,
        model_name: str,
        device="cuda:0",
    ):
        super().__init__(model_name, device)

    def encode(
        self,
        qid: int,
        query_txt: str,
        query_img_path: str,
        query_modality: str,
        instruction_prompt: Optional[str] = None,
        fp16: bool = False,
    ):
        if instruction_prompt is not None:
            query_txt = f"{instruction_prompt} {query_txt}" if query_txt else instruction_prompt

        query_info = [
            {
                "qid": hash_qid(qid),
                "query_txt": format_string(query_txt),
                "query_img_path": query_img_path,
                "query_modality": query_modality,
            }
        ]

        dataset = MBEIRQueryDataset(query_info, self.img_preprocess_fn)
        collator = MBEIRInferenceOnlyCollator(tokenizer=self.tokenizer, image_size=(224, 224))
        dataloader = DataLoader(dataset, batch_size=1, collate_fn=collator)

        query_embeddings, _ = generate_embeds_and_ids_for_dataset_with_gather(
            self.model,
            dataloader,
            device=self.device,
            use_fp16=fp16,
        )

        return query_embeddings
