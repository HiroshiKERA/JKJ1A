"""Small VLM inference and controlled input generation."""
from pathlib import Path
import json
import math
import random
from PIL import Image, ImageDraw, ImageFont
import torch


def counting_image(n=5, *, layout='grid', seed=0, size=512, radius=24):
    if not isinstance(n, int) or not 1 <= n <= 25:
        raise ValueError('n must be an integer between 1 and 25')
    if layout not in ('grid', 'jitter'):
        raise ValueError('layout must be grid or jitter')
    side = math.ceil(math.sqrt(n))
    spacing = size / (side + 1)
    if not 0 < radius < spacing * .3:
        raise ValueError('radius is too large for separate circles')
    rng = random.Random(seed)
    image = Image.new('RGB', (size, size), 'white')
    draw = ImageDraw.Draw(image)
    for i in range(n):
        x, y = spacing*(i % side+1), spacing*(i//side+1)
        if layout == 'jitter':
            x += rng.uniform(-.15, .15)*spacing
            y += rng.uniform(-.15, .15)*spacing
        draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill='#2064b0')
    return image


def add_text(image, text, *, font_size=32):
    """Add a white label band, without covering the original image."""
    image = image.convert('RGB')
    font = ImageFont.load_default(size=font_size)
    bbox = font.getbbox(text)
    width = max(image.width, bbox[2]-bbox[0]+24)
    band_height = bbox[3]-bbox[1]+32
    output = Image.new('RGB', (width, image.height+band_height), 'white')
    output.paste(image, ((width-image.width)//2, 0))
    draw = ImageDraw.Draw(output)
    draw.text(((width-(bbox[2]-bbox[0]))//2, image.height+16-bbox[1]), text, font=font, fill='black')
    return output


class MiniVLM:
    def __init__(self, model_id='HuggingFaceTB/SmolVLM-500M-Instruct', *, device=None):
        from transformers import AutoProcessor, AutoModelForImageTextToText
        self.model_id = model_id
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        dtype = torch.float16 if self.device.type == 'cuda' else torch.float32
        print(f'モデルを読み込み中: {model_id}', flush=True)
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id, torch_dtype=dtype, attn_implementation='eager',
        ).to(self.device).eval()
        self.revision = getattr(self.model.config, '_commit_hash', None)

    @torch.inference_mode()
    def ask(self, image, question, *, max_new_tokens=64):
        messages = [{'role': 'user', 'content': [
            {'type': 'image'}, {'type': 'text', 'text': question}]}]
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=prompt, images=[image.convert('RGB')], return_tensors='pt')
        inputs = inputs.to(self.device)
        for key, value in inputs.items():
            if torch.is_floating_point(value):
                inputs[key] = value.to(self.model.dtype)
        tokens = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        answer = self.processor.batch_decode(tokens[:, inputs['input_ids'].shape[1]:],
                                             skip_special_tokens=True)[0].strip()
        return answer


def save_observation(vlm, image, question, answer, *, name, expected, condition,
                     directory='results/day2-vlm', max_new_tokens=64):
    import transformers
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if not name or Path(name).name != name or name in ('.', '..'):
        raise ValueError('name must be a file name, not a path')
    image_path = directory / f'{name}.png'
    record_path = directory / f'{name}.json'
    if image_path.exists() or record_path.exists():
        raise FileExistsError(f'{name} は保存済みです。別のnameを指定してください。')
    image.save(image_path)
    record = dict(model=vlm.model_id, revision=vlm.revision, question=question, answer=answer,
                  expected=expected, condition=condition, image=image_path.name,
                  do_sample=False, max_new_tokens=max_new_tokens,
                  torch_version=torch.__version__, transformers_version=transformers.__version__)
    record_path.write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n')
    return record_path
