import io
import json

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T

device = "cuda" if torch.cuda.is_available() else "cpu"

transforms = T.Compose(
    [
        T.ToTensor(),
        T.Resize((224, 224)),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

class_name = ["buildings", "forest", "glacier", "mountain", "sea", "street"]


def model_fn(model_dir):
    model = torch.jit.load(f"{model_dir}/model.scripted.pt")
    model.to(device).eval()

    return model


# data preprocessing
def input_fn(request_body, request_content_type):
    assert request_content_type == "application/x-npy"
    data = np.load(io.BytesIO(request_body))
    data = transforms(data)
    return data.unsqueeze(0)


# inference
def predict_fn(input_object, model):
    with torch.no_grad():
        prediction = model(input_object)
        prediction = F.softmax(prediction, dim=-1)
    return prediction


# postprocess
def output_fn(predictions, content_type):
    assert content_type == "application/json"

    top_prob, top_classes = torch.topk(predictions, 5, dim=1)
    top_prob, top_classes = top_prob[0].cpu().tolist(), top_classes[0].cpu().tolist()
    result = {class_name[classes]: probs * 100 for classes, probs in zip(top_classes, top_prob)}

    return json.dumps(result)
