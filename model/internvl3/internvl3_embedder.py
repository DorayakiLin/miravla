# model/internvl3/internvl3_embedder.py
import torch
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_ar = ratio[0] / ratio[1]
        diff = abs(aspect_ratio - target_ar)
        if diff < best_ratio_diff:
            best_ratio_diff = diff
            best_ratio = ratio
        elif diff == best_ratio_diff and area > 0.5 * image_size**2 * ratio[0] * ratio[1]:
            best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=True):
    w, h = image.size
    aspect_ratio = w / h
    target_ratios = sorted(set(
        (i, j) for n in range(min_num, max_num+1) for i in range(1, n+1) for j in range(1, n+1) if min_num <= i*j <= max_num
    ), key=lambda x: x[0]*x[1])

    tile_x, tile_y = find_closest_aspect_ratio(aspect_ratio, target_ratios, w, h, image_size)
    resized = image.resize((tile_x * image_size, tile_y * image_size))
    blocks = tile_x * tile_y
    tiles = [
        resized.crop((j*image_size, i*image_size, (j+1)*image_size, (i+1)*image_size))
        for i in range(tile_y) for j in range(tile_x)
    ]
    if use_thumbnail and blocks != 1:
        tiles.append(image.resize((image_size, image_size)))
    return tiles

class InternVL3Embedder:
    def __init__(self, model_name="OpenGVLab/InternVL3-2B", image_size=448, device="cuda"):
        self.device = device
        self.image_size = image_size
        self.transform = build_transform(image_size)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)
        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            use_flash_attn=True,
            low_cpu_mem_usage=True,
            device_map="auto"
        ).eval()

    def encode_image(self, image_path):
        image = Image.open(image_path).convert('RGB')
        tiles = dynamic_preprocess(image, image_size=self.image_size)
        pixels = torch.stack([self.transform(tile) for tile in tiles])
        return pixels.to(torch.bfloat16).to(self.device)

    def chat(self, image_path, text_prompt, generation_config=None):
        pixel_values = self.encode_image(image_path)
        prompt = "<image>\n" + text_prompt
        if generation_config is None:
            generation_config = dict(max_new_tokens=512, do_sample=True, output_hidden_states=True, return_dict=True)
        outputs = self.model.chat(self.tokenizer, pixel_values, prompt, generation_config)
        fused_hidden = outputs.hidden_states[-1]  # [1, T+L, D]
        return fused_hidden[:, 0, :] if return_cls_only else fused_hidden
        

    def get_image_embedding(self, image_path):
        pixel_values = self.encode_image(image_path)
        with torch.no_grad():
            outputs = self.model.vision_model(pixel_values)
            return outputs.last_hidden_state  # [B, N, D]

    def get_fused_image_text_embedding(self, image_path, text_prompt, return_cls_only=True):
        pixel_values = self.encode_image(image_path)  # [T, 3, 448, 448]
        num_tiles = pixel_values.shape[0]

        # 构造 prompt：必须包含 <image> 占位符
        prompt = "<image>\n" + text_prompt

        # 构造 image_flags: 每个 tile = 1
        image_flags = torch.ones((num_tiles,), dtype=torch.long).to(self.device)

        # 构造模板 prompt，替换 <image> 为 <img><IMG_CONTEXT>*n</img>
        IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"
        IMG_START_TOKEN = "<img>"
        IMG_END_TOKEN = "</img>"
        image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.model.num_image_token * 1 + IMG_END_TOKEN
        prompt = prompt.replace("<image>", image_tokens, 1)

        # 构建 tokenizer 输出
        model_inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs["attention_mask"]

        # 设置必须字段
        self.model.img_context_token_id = self.tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)

        # 模型 forward 推理（返回 hidden_states）
        with torch.no_grad():
            outputs = self.model(
                pixel_values=pixel_values,
                image_flags=image_flags,
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True
            )

            fused_hidden = outputs.hidden_states[-1]  # [1, T+L, D]
            return fused_hidden[:, 0, :] if return_cls_only else fused_hidden
