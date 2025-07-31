import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel

class LocalEmbeddingFunction:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def __call__(self, texts):
        embeddings = []
        for text in texts:
            with torch.no_grad():
                encoded_input = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
                encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}
                model_output = self.model(**encoded_input)

                # Mean Pooling
                token_embeddings = model_output.last_hidden_state
                input_mask_expanded = encoded_input['attention_mask'].unsqueeze(-1).expand(token_embeddings.size())
                sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
                sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
                mean_pooled = sum_embeddings / sum_mask

                # Normalize
                normalized = torch.nn.functional.normalize(mean_pooled, p=2, dim=1)
                embeddings.append(normalized.squeeze().cpu().numpy())
        return embeddings
