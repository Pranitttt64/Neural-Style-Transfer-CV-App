import streamlit as st
import torch
import torch.optim as optim
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms, models
import numpy as np
import io
import tempfile
import os

# Set page config
st.set_page_config(page_title="Neural Style Transfer", page_icon="🎨", layout="wide")

# Title and description
st.title("🎨 Neural Style Transfer")
st.markdown("Transform your images by applying the artistic style of famous paintings or other images!")

# Initialize session state
if 'processed_image' not in st.session_state:
    st.session_state.processed_image = None

# Device setup
@st.cache_resource
def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device

device = get_device()

# Load VGG model
@st.cache_resource
def load_vgg_model():
    vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features.to(device).eval()
    # Freeze parameters
    for param in vgg.parameters():
        param.requires_grad_(False)
    return vgg

vgg = load_vgg_model()

def load_image(image_file, max_size=720, shape=None):
    """Load and preprocess image"""
    image = Image.open(image_file).convert('RGB')
    
    if max(image.size) > max_size:
        size = max_size
    else:
        size = max(image.size)
    
    if shape is not None:
        size = shape
    
    in_transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor()
    ])
    
    image = in_transform(image)[:3, :, :].unsqueeze(0)
    return image

def im_convert(tensor):
    """Convert tensor to PIL image"""
    image = tensor.to("cpu").clone().detach()
    image = image.squeeze(0)
    image = transforms.ToPILImage()(image.clamp(0, 1))
    return image

def get_features(image, model=None):
    """Extract features from VGG layers"""
    if model is None:
        model = vgg
    
    layers = {
        '0': 'conv1_1',
        '5': 'conv2_1',
        '10': 'conv3_1',
        '19': 'conv4_1',
        '21': 'conv4_2',  # Content layer
        '28': 'conv5_1'
    }
    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x
    return features

def gram_matrix(tensor):
    """Calculate Gram matrix for style loss"""
    b, c, h, w = tensor.size()
    tensor = tensor.view(c, h * w)
    gram = torch.mm(tensor, tensor.t())
    return gram

def train_style_transfer(content, style, steps=500, lr=0.003, content_weight=1, style_weight=1e6):
    """Perform neural style transfer"""
    
    # Style weights for different layers
    style_weights = {
        'conv1_1': 1.,
        'conv2_1': 0.75,
        'conv3_1': 0.2,
        'conv4_1': 0.2,
        'conv5_1': 0.2
    }
    
    # Get features and detach them to avoid graph retention
    with torch.no_grad():
        content_features = get_features(content.detach())
        style_features = get_features(style.detach())
        style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}
    
    # Initialize target image
    target = content.detach().clone().requires_grad_(True).to(device)
    optimizer = torch.optim.Adam([target], lr=lr)
    
    # Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for step in range(1, steps + 1):
        # Clear gradients
        optimizer.zero_grad()
        
        # Get target features
        target_features = get_features(target)
        
        # Content loss
        content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2'])**2)
        
        # Style loss
        style_loss = 0
        for layer in style_weights:
            target_feature = target_features[layer]
            target_gram = gram_matrix(target_feature)
            style_gram = style_grams[layer]
            layer_style_loss = style_weights[layer] * torch.mean((target_gram - style_gram)**2)
            style_loss += layer_style_loss / (target_feature.shape[1]**2)
        
        total_loss = content_weight * content_loss + style_weight * style_loss
        
        # Backward pass
        total_loss.backward()
        optimizer.step()
        
        # Clamp values to valid range
        with torch.no_grad():
            target.clamp_(0, 1)
        
        # Update progress
        progress = step / steps
        progress_bar.progress(progress)
        
        if step % 50 == 0:
            status_text.text(f"Step {step}/{steps}, Total loss: {total_loss.item():.4f}")
    
    progress_bar.empty()
    status_text.empty()
    
    return target

def upscale_image(image, target_size=1024):
    """Upscale image while maintaining aspect ratio"""
    orig_w, orig_h = image.size
    
    # Scale while keeping aspect ratio
    if orig_w > orig_h:
        new_w = target_size
        new_h = int(target_size * (orig_h / orig_w))
    else:
        new_h = target_size
        new_w = int(target_size * (orig_w / orig_h))
    
    # Resize without distorting ratio
    upscaled_img = image.resize((new_w, new_h), Image.LANCZOS)
    return upscaled_img

# Sidebar for parameters
st.sidebar.header("Parameters")
steps = st.sidebar.slider("Training Steps", 100, 1000, 500, 50)
learning_rate = st.sidebar.slider("Learning Rate", 0.001, 0.01, 0.003, 0.001)
content_weight = st.sidebar.slider("Content Weight", 1, 10, 1)
style_weight = st.sidebar.number_input("Style Weight", 1e4, 1e7, 1e6, format="%.0e")
upscale_size = st.sidebar.slider("Upscale Size", 512, 2048, 1024, 128)

# Main interface
col1, col2 = st.columns(2)

with col1:
    st.header("Content Image")
    content_file = st.file_uploader("Upload Content Image", type=['png', 'jpg', 'jpeg'], key="content")
    
    if content_file is not None:
        content_image = Image.open(content_file)
        st.image(content_image, caption="Content Image", use_column_width=True)

with col2:
    st.header("Style Image")
    style_file = st.file_uploader("Upload Style Image", type=['png', 'jpg', 'jpeg'], key="style")
    
    if style_file is not None:
        style_image = Image.open(style_file)
        st.image(style_image, caption="Style Image", use_column_width=True)

# Process button
if st.button("🎨 Apply Style Transfer", type="primary"):
    if content_file is not None and style_file is not None:
        with st.spinner("Processing... This may take a few minutes."):
            try:
                # Clear any existing gradients and cache
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
                # Load and preprocess images
                content_tensor = load_image(content_file).to(device)
                style_tensor = load_image(style_file, shape=[content_tensor.size(2), content_tensor.size(3)]).to(device)
                
                # Perform style transfer
                result_tensor = train_style_transfer(
                    content_tensor, 
                    style_tensor, 
                    steps=steps,
                    lr=learning_rate,
                    content_weight=content_weight,
                    style_weight=style_weight
                )
                
                # Convert to PIL image
                result_image = im_convert(result_tensor)
                
                # Upscale the result
                upscaled_result = upscale_image(result_image, target_size=upscale_size)
                
                # Store in session state
                st.session_state.processed_image = upscaled_result
                
                st.success("Style transfer completed!")
                
                # Clear cache after processing
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.error("Please make sure both images are uploaded and try again.")
                # Clear cache on error
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
    else:
        st.warning("Please upload both content and style images before processing.")

# Display result
if st.session_state.processed_image is not None:
    st.header("Result")
    
    # Display the processed image
    st.image(st.session_state.processed_image, caption="Style Transfer Result", use_column_width=True)
    
    # Download button
    buf = io.BytesIO()
    st.session_state.processed_image.save(buf, format="PNG")
    byte_im = buf.getvalue()
    
    st.download_button(
        label="📥 Download Result",
        data=byte_im,
        file_name="style_transfer_result.png",
        mime="image/png"
    )

# Information section
with st.expander("ℹ️ How it works"):
    st.markdown("""
    **Neural Style Transfer** uses a pre-trained VGG-19 network to separate and recombine the content and style of images:
    
    1. **Content Representation**: Captured from deeper layers of the network
    2. **Style Representation**: Captured using Gram matrices of feature maps from multiple layers
    3. **Optimization**: The algorithm iteratively updates a target image to minimize both content and style losses
    
    **Tips for better results:**
    - Use high-contrast style images for more dramatic effects
    - Adjust the style weight to control how strongly the style is applied
    - More training steps generally produce better results but take longer
    - The content weight controls how much the original image structure is preserved
    """)

with st.expander("⚙️ Technical Details"):
    st.markdown(f"""
    - **Device**: {device}
    - **Model**: VGG-19 (pre-trained on ImageNet)
    - **Content Layer**: conv4_2
    - **Style Layers**: conv1_1, conv2_1, conv3_1, conv4_1, conv5_1
    - **Optimization**: Adam optimizer
    """)

st.markdown("---")
st.markdown("Made By Pranit Saundankar")
