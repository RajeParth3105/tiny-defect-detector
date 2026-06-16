import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

# Set page configuration for a premium, wide dashboard look
st.set_page_config(
    page_title="AI Defect Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom dark theme details
st.markdown("""
<style>
    .reportview-container {
        background: #111216;
    }
    .main {
        background-color: #0e1117;
        color: #ffffff;
    }
    .stAlert {
        border-radius: 10px;
    }
    h1 {
        font-family: 'Outfit', 'Inter', sans-serif;
        font-weight: 700;
        background: linear-gradient(45deg, #FF4B4B, #FF8F8F);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: #1e222b;
        border-radius: 10px;
        padding: 15px;
        border: 1px solid #3e4451;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# 1. Global Configurations
MODEL_PATH = r"C:\Users\abhij\AntigravityProjects\defect_detector\defect_model.pth"
CLASS_NAMES = ['Crazing', 'Inclusion', 'Patches', 'Pitted', 'Rolled', 'Scratches']

# Initialize GPU/CPU Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model (cached to run fast)
@st.cache_resource
def load_defect_model():
    model = models.resnet18()
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(CLASS_NAMES))
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
        model.to(device)
        return model
    else:
        st.error(f"Checkpoint not found at: {MODEL_PATH}. Please run training first!")
        return None

# Load model
model = load_defect_model()

# Image Transformations
img_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 2. Grad-CAM Class to generate defect heatmaps
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_handles = []

        # Hook functions to capture activations and gradients
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        # Register hooks on layer4 (for ResNet18 convolutional output)
        self.hook_handles.append(target_layer.register_forward_hook(forward_hook))
        self.hook_handles.append(target_layer.register_full_backward_hook(backward_hook))

    def generate_heatmap(self, input_tensor, class_idx):
        self.model.zero_grad()
        output = self.model(input_tensor)
        
        # Target output class logit
        loss = output[0, class_idx]
        loss.backward()

        # Compute channel weights by averaging gradients
        gradients = self.gradients.cpu().data.numpy()[0]
        activations = self.activations.cpu().data.numpy()[0]
        
        weights = np.mean(gradients, axis=(1, 2))
        
        # Calculate weighted sum of activations
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        # Apply ReLU (we only care about features that positively contribute to the target class)
        cam = np.maximum(cam, 0)
        
        # Normalize between 0 and 1
        if cam.max() != 0:
            cam = cam - cam.min()
            cam = cam / cam.max()
            
        # Remove hooks
        for handle in self.hook_handles:
            handle.remove()
            
        return cam

def generate_cam_overlay(image_pil, heatmap):
    # Resize heatmap to match image size
    img = np.array(image_pil)
    height, width, _ = img.shape
    heatmap_resized = cv2.resize(heatmap, (width, height))
    
    # Scale heatmap to [0, 255] and apply jet colormap
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Overlay heatmap onto original image
    overlay = cv2.addWeighted(img, 0.6, heatmap_colored, 0.4, 0)
    return overlay

# 3. Streamlit App Layout
st.sidebar.markdown("## ⚙️ Model Information")
st.sidebar.info(
    "**Network**: ResNet-18 (Pre-trained + Fine-tuned)\n\n"
    "**Input Resolution**: 224x224 RGB\n\n"
    "**Hardware Target**: Edge GPU / CUDA\n\n"
    "**Accuracy on Test Set**: 94.4%"
)

st.title("Metal Sheet Defect Detector")
st.markdown("A deep-learning inspection tool that automatically scans sheet metal components for structural surface defects and highlights suspicious areas.")

# Uploader widget
uploaded_file = st.file_uploader("Upload an image of a sheet metal surface (BMP, PNG, JPG):", type=["bmp", "png", "jpg"])

if uploaded_file is not None:
    # Read the image
    image = Image.open(uploaded_file).convert('RGB')
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Original Surface Image")
        st.image(image, use_container_width=True)
        
    with col2:
        if model is not None:
            # Prepare tensor
            tensor = img_transforms(image).unsqueeze(0).to(device)
            
            # Run forward pass to get prediction
            with torch.set_grad_enabled(True):
                # We need gradients enabled even for evaluation to run Grad-CAM
                tensor.requires_grad = True
                outputs = model(tensor)
                probabilities = torch.softmax(outputs, dim=1).detach().cpu().numpy()[0]
                pred_idx = np.argmax(probabilities)
                pred_class = CLASS_NAMES[pred_idx]
                confidence = probabilities[pred_idx] * 100
                
            # Perform Grad-CAM visualization
            # Target layer4 of Resnet18 (last convolutional layer block)
            target_layer = model.layer4[-1]
            grad_cam = GradCAM(model, target_layer)
            
            # Generate heatmap for the predicted category
            try:
                heatmap = grad_cam.generate_heatmap(tensor, pred_idx)
                cam_overlay = generate_cam_overlay(image, heatmap)
                st.subheader("Grad-CAM Defect Localization Map")
                st.image(cam_overlay, use_container_width=True, caption="Warm colors (red/orange) highlight where the neural network detected the defect.")
            except Exception as e:
                st.warning("Could not generate Grad-CAM visualization. Displaying raw image instead.")
                st.image(image, use_container_width=True)
                
    # 4. Result Metrics Section
    st.markdown("---")
    res_col1, res_col2 = st.columns([1, 2])
    
    with res_col1:
        st.subheader("Primary Classification")
        st.markdown(
            f"<div class='metric-card'>"
            f"<h2 style='color:#FF4B4B; margin:0;'>{pred_class}</h2>"
            f"<p style='margin:5px 0 0 0; font-size:1.1rem;'>Confidence: <b>{confidence:.2f}%</b></p>"
            f"</div>",
            unsafe_allow_html=True
        )
        
    with res_col2:
        st.subheader("Classification Probabilities Distribution")
        # Plot horizontal bar charts for all categories
        for i, (name, prob) in enumerate(zip(CLASS_NAMES, probabilities)):
            percentage = prob * 100
            st.write(f"**{name}** ({percentage:.1f}%)")
            st.progress(float(prob))

else:
    # Visual placeholder layout when no image is loaded
    st.info("Please upload a sheet metal surface image to start the visual inspection pipeline.")
    
    # Display some reference test files for quick testing
    st.markdown("### Sample Test Files Available:")
    valid_dir = r"C:\Users\abhij\AntigravityProjects\defect_detector\NEU Metal Surface Defects Data\valid"
    if os.path.exists(valid_dir):
        categories = os.listdir(valid_dir)
        st.write("You can find sample defect images on your system in these folders:")
        for cat in categories:
            cat_path = os.path.join(valid_dir, cat)
            if os.path.isdir(cat_path):
                files = os.listdir(cat_path)[:3]
                st.write(f"- **{cat}**: `.../valid/{cat}/{files[0]}`")
