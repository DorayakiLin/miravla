from transformers import AutoModel, AutoProcessor
from PIL import Image
import torch

class InternVL3Embedder:
    def __init__(self, device="cuda"):
        self.device = device
        self.processor = AutoProcessor.from_pretrained("OpenGVLab/InternVL3-2B", trust_remote_code=True)
        self.model = AutoModel.from_pretrained("OpenGVLab/InternVL3-2B", trust_remote_code=True).to(device)

    def get_embedding(self, image_path, text_prompt):
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, text=text_prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            if hasattr(outputs, "last_hidden_state"):
                return outputs.last_hidden_state  # B x N x D
            else:
                print("[WARN] No last_hidden_state in output. Returning raw output.")
                return outputs
