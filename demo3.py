import torch
import requests
from PIL import Image
from transformers import BlipProcessor, BlipForQuestionAnswering

processor = BlipProcessor.from_pretrained("ybelkada/blip-vqa-base")
model = BlipForQuestionAnswering.from_pretrained(
    "ybelkada/blip-vqa-base", torch_dtype=torch.float16
).to("cuda")

img_url = "https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg"
raw_image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")

question = "question: this picture depict a man holding a cat.Does this picture depict a woman holding a dog?answer:"
question = "Known information: This picture depicts a man holding a cat. Question: Does this picture depict a woman holding a dog? Answer: ."
inputs = processor(raw_image, question, return_tensors="pt").to("cuda", torch.float16)

out = model.generate(**inputs)
print(processor.decode(out[0], skip_special_tokens=True))
