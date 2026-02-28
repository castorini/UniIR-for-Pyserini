from PIL import Image
from torch.utils.data import Dataset


class MBEIRCorpusDataset(Dataset):
    def __init__(self, batch_info, img_preprocess_fn, **kwargs):
        data = []
        num_records = len(batch_info["did"])
        for i in range(num_records):
            record = {
                "did": batch_info["did"][i],
                "img_path": batch_info["img_path"][i],
                "modality": batch_info["modality"][i],
                "txt": batch_info["txt"][i],
            }
            data.append(record)
        self.data = data
        self.img_preprocess_fn = img_preprocess_fn
        self.kwargs = kwargs

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        img_path = entry.get("img_path", None)
        if img_path:
            try:
                img = Image.open(img_path).convert("RGB")
                img = self.img_preprocess_fn(img)
            except Exception:
                raise ValueError(
                    f"Could not open image at {img_path}. Please check the file path or format."
                )
        else:
            img = None

        instance = {
            "did": entry["did"],
            "txt": entry.get("txt", ""),
            "img": img,
            "modality": entry["modality"],
        }
        return instance


class MBEIRQueryDataset(Dataset):
    def __init__(self, batch_info, img_preprocess_fn, **kwargs):
        data = []
        num_records = len(batch_info["qid"])
        for i in range(num_records):
            record = {
                "qid": batch_info["qid"][i],
                "query_txt": batch_info["query_txt"][i],
                "img_path": batch_info["query_img_path"][i],
            }
            data.append(record)
        self.data = data
        self.img_preprocess_fn = img_preprocess_fn
        self.kwargs = kwargs

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        img_path = entry.get("img_path", None)
        if img_path:
            try:
                img = Image.open(img_path).convert("RGB")
                img = self.img_preprocess_fn(img)
            except Exception:
                raise ValueError(
                    f"Could not open image at {img_path}. Please check the file path or format."
                )
        else:
            img = None

        query = {"txt": entry["query_txt"], "img": img, "qid": entry["qid"]}

        instance = {
            "query": query,
            "qid": entry["qid"],
        }
        return instance
