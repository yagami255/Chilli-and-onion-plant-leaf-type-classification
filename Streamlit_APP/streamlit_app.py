import warnings
from pathlib import Path

import numpy as np
import streamlit as st
import timm
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

from lime import lime_image
from skimage.segmentation import mark_boundaries, quickshift

from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, EigenCAM, AblationCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

warnings.filterwarnings("ignore")

# ==============================
# CONFIG
# ==============================
CLASS_NAMES = ["Cercospora", "Healthy", "Mites", "Nutritional", "Powdery"]
NUM_CLASSES = len(CLASS_NAMES)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = Path(__file__).resolve().parent

def resolve_path(p: str | Path) -> Path:
    """Return usable Path from absolute or relative input."""
    p = Path(p)
    if p.is_absolute():
        return p
    return (BASE_DIR / p).resolve()

MODEL_CONFIGS = {
    "Custom CNN": {
        "path": resolve_path(r"C:\Users\Asus\Downloads\CSE 366_Project\models\custom_cnn_best.pth"),
        "input_size": (224, 224),
    },
    "ConvNeXt-Tiny": {
        "path": resolve_path(r"C:\Users\Asus\Downloads\CSE 366_Project\models\Convne_best.pth"),
        "input_size": (224, 224),
    },
    "DenseNet-121": {
        "path": resolve_path(r"C:\Users\Asus\Downloads\CSE 366_Project\models\densenet121_best.pth"),
        "input_size": (224, 224),
    },
    "EfficientNet-B0": {
        "path": resolve_path(r"C:\Users\Asus\Downloads\CSE 366_Project\models\efficientnet_b0_best.pth"),
        "input_size": (224, 224),
    },
    "Inception-V3": {
        "path": resolve_path(r"C:\Users\Asus\Downloads\CSE 366_Project\models\inception_v3_best.pth"),
        "input_size": (299, 299),
    },
}

SAMPLE_IMAGES = {
    "Cercospora": resolve_path(r"C:\Users\Asus\Downloads\CSE 366_Project\sample_Images\Cercospora.jpg"),
    "Healthy": resolve_path(r"C:\Users\Asus\Downloads\CSE 366_Project\sample_Images\Healthy.jpg"),
    "Mites": resolve_path(r"C:\Users\Asus\Downloads\CSE 366_Project\sample_Images\Mites_trips.jpg"),
    "Nutritional": resolve_path(r"C:\Users\Asus\Downloads\CSE 366_Project\sample_Images\Nutritional.jpg"),
    "Powdery": resolve_path(r"C:\Users\Asus\Downloads\CSE 366_Project\sample_Images\Powdery mildew.jpg"),
}

# ==============================
# CUSTOM CSS
# ==============================
st.set_page_config(page_title="🌿 Chilli and Onion Plant Leaf Disease Classifier", layout="wide")
st.markdown("""
<style>
body { background: #f4f9f4; }
.main-header {
    font-size: 2.5rem; color: white; text-align: center;
    background: linear-gradient(90deg, #228B22, #32CD32);
    padding: 15px; border-radius: 10px; margin-bottom: 20px;
}
.card {
    background: black; padding: 20px; border-radius: 12px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.1); margin-bottom: 20px;
}
.prediction-box {
    background: black; padding: 20px; border-left: 5px solid #228B22;
    border-radius: 10px; margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🌿 Chilli and Onion Plant Leaf Disease Classifier</div>', unsafe_allow_html=True)

# ==============================
# MODEL CLASS
# ==============================
class CustomCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.Dropout(0.25), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.Dropout(0.25), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.Dropout(0.3), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 7 * 7, 512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))

# ==============================
# HELPERS
# ==============================
def get_transform(size):
    return transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

def preprocess(image, size):
    return get_transform(size)(image).unsqueeze(0)

def get_last_conv_layer(model):
    """Find last Conv2d layer for Grad-CAM."""
    for name, m in reversed(list(model.named_modules())):
        if isinstance(m, nn.Conv2d):
            return [m]
    return [model]

def torch_predict_fn(model, size):
    """Return prediction function for LIME."""
    transform = get_transform(size)
    def _predict(imgs):
        batch = []
        for arr in imgs:
            arr = (arr * 255).astype(np.uint8)
            pil = Image.fromarray(arr)
            batch.append(transform(pil))
        batch = torch.stack(batch).to(DEVICE)
        with torch.no_grad():
            outputs = model(batch)
            return torch.softmax(outputs, dim=1).cpu().numpy()
    return _predict

# ==============================
# LOAD MODEL
# ==============================
@st.cache_resource
def load_model(model_name, path, num_classes):
    path = resolve_path(path)
    ckpt = torch.load(path, map_location=DEVICE)
    state_dict = ckpt.get("model_state_dict", ckpt)

    if model_name == "Custom CNN":
        model = CustomCNN(num_classes)
    elif model_name == "ConvNeXt-Tiny":
        model = timm.create_model("convnext_tiny", pretrained=False, num_classes=num_classes)
    elif model_name == "DenseNet-121":
        model = models.densenet121(weights=None)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif model_name == "EfficientNet-B0":
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name == "Inception-V3":
        model = models.inception_v3(weights=None, aux_logits=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        model.load_state_dict(state_dict, strict=False)

    return model.to(DEVICE).eval()

# ==============================
# SIDEBAR
# ==============================
st.sidebar.header("⚙️ Settings")
model_name = st.sidebar.selectbox("Choose Model", list(MODEL_CONFIGS.keys()))
uploaded_file = st.sidebar.file_uploader("Upload Image", type=["jpg","jpeg","png"])
sample_choice = st.sidebar.selectbox("Or use sample image", ["None"] + list(SAMPLE_IMAGES.keys()))
xai_methods = st.sidebar.multiselect("XAI Methods", ["Grad-CAM","Grad-CAM++","Eigen-CAM","Ablation-CAM","LIME"], default=["Grad-CAM","LIME"])
xai_mode = st.sidebar.radio("XAI Engine", ["Basic","Advanced (Generator)"], index=1)

# ==============================
# IMAGE SELECTION
# ==============================
image = None
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
elif sample_choice != "None":
    image = Image.open(SAMPLE_IMAGES[sample_choice]).convert("RGB")

# ==============================
# MAIN PREDICTION + XAI
# ==============================
if image:
    st.markdown('<div class="card"><h3>📷 Input Image</h3></div>', unsafe_allow_html=True)
    st.image(image, width=300)

    config = MODEL_CONFIGS[model_name]
    model = load_model(model_name, config["path"], NUM_CLASSES)
    size = config["input_size"]
    tensor = preprocess(image, size).to(DEVICE)

    # Prediction
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    top_idxs = np.argsort(probs)[::-1][:3]
    pred_class = CLASS_NAMES[top_idxs[0]]
    st.markdown(f'<div class="prediction-box"><h3>✅ Predicted: {pred_class}</h3><p>Confidence: {probs[top_idxs[0]]*100:.2f}%</p></div>', unsafe_allow_html=True)

    for i in top_idxs:
        st.progress(int(probs[i]*100))
        st.write(f"**{CLASS_NAMES[i]}**: {probs[i]*100:.2f}%")

    # XAI
    if xai_methods:
        st.subheader("Explainable AI Visualizations")
        img_np = np.array(image.resize(size)).astype(np.float32) / 255.0
        results = {}

        if xai_mode == "Advanced (Generator)":
            target_layers = get_last_conv_layer(model)

        for m in xai_methods:
            if m == "LIME":
                explainer = lime_image.LimeImageExplainer()
                explanation = explainer.explain_instance(
                    img_np,
                    torch_predict_fn(model, size),
                    top_labels=1,
                    hide_color=0,
                    num_samples=1000,
                    segmentation_fn=lambda x: quickshift(x, kernel_size=4, max_dist=200, ratio=0.2)
                )
                label = int(np.argmax(probs))
                lime_img, mask = explanation.get_image_and_mask(label=label, positive_only=True, num_features=10, hide_rest=False)
                results[m] = mark_boundaries(lime_img, mask)

            else:
                cam_class = {"Grad-CAM": GradCAM,"Grad-CAM++": GradCAMPlusPlus,"Eigen-CAM": EigenCAM,"Ablation-CAM": AblationCAM}[m]
                target_layers = get_last_conv_layer(model)
                cam = cam_class(model=model, target_layers=target_layers)
                grayscale_cam = cam(input_tensor=tensor, targets=[ClassifierOutputTarget(int(top_idxs[0]))])[0]
                cam_img = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)
                results[m] = cam_img

        cols = st.columns(len(results))
        for i,(name,img) in enumerate(results.items()):
            cols[i].image(img, caption=name, use_container_width=True)

else:
    st.info("👆 Upload an image or select a sample from the sidebar to get started.")
