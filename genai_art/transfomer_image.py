# Add necessary imports for transformer implementation
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
from genai_art.image import display_rgb

# Define a simple transformer-based image fusion model
class ImageTransformer(nn.Module):
    def __init__(self, img_size=255, patch_size=15, in_channels=3, num_images=5, 
                 embed_dim=256, depth=4, num_heads=8, mlp_ratio=4):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.num_images = num_images
        
        # Calculate number of patches
        self.num_patches = (img_size // patch_size) ** 2
        
        # Patch embedding layer
        self.patch_embed = nn.Conv2d(
            in_channels, 
            embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size)
        
        # Position embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        
        # Image type embeddings (to differentiate between different input images)
        self.img_type_embed = nn.Parameter(torch.zeros(num_images, 1, embed_dim))
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=embed_dim*mlp_ratio
            )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=depth
        )
        
        # Output projection to reconstruct the image
        self.output_proj = nn.Linear(
            embed_dim, 
            patch_size * patch_size * in_channels
            )
        
    def forward(self, x_list):
        # x_list is a list of input images [batch_size, channels, height, width]
        b, c, h, w = x_list[0].shape
        
        # Process each image and concatenate
        embeddings = []
        for i, x in enumerate(x_list):
            # Convert image to patches and flatten: [B, C, H, W] -> [B, E, H/P, W/P] -> [B, H/P*W/P, E]
            x = self.patch_embed(x).flatten(2).permute(0, 2, 1)
            
            # Add position embeddings and image type embeddings
            x = x + self.pos_embed + self.img_type_embed[i]

            # Do something like:
            # x = x + self.pos_embed + self.img_type_embed[i].unsqueeze(1)
                
            embeddings.append(x)
        
        # Concatenate all embeddings along sequence dimension
        x = torch.cat(embeddings, dim=1)  # [B, num_images*num_patches, embed_dim]
        
        # Apply transformer (reshape for transformer: [seq_len, batch_size, embed_dim])
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # [batch_size, seq_len, embed_dim]
        
        # Take only the first num_patches embeddings for reconstruction
        x = x[:, :self.num_patches, :]
        
        # Project to patch pixels
        x = self.output_proj(x)  # [B, num_patches, P*P*C]
        
        # Reshape to image
        x = x.reshape(b, h // self.patch_size, w // self.patch_size, 
                     self.patch_size, self.patch_size, c)
        # Verify dimensions before permute and reshape
        print(f"Before permute - x shape: {x.shape}, expected: [b={b}, h/p={h//self.patch_size}, w/p={w//self.patch_size}, p={self.patch_size}, p={self.patch_size}, c={c}]")
        x = x.permute(0, 5, 1, 3, 2, 4)
        print(f"After permute - x shape: {x.shape}")
        # Calculate expected size and compare with actual size
        print(f"b={b}, c={c}, h={h}, w={w}")
        expected_size = b * c * h * w
        actual_size = x.numel()
        print(f"Expected size: {expected_size}, Actual size: {actual_size}")
        # Reshape with size verification
        x = x.reshape(b, c, h, w)
        
        return x

# Function to preprocess images for the transformer
def preprocess_images(image_list):
    processed_images = []
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((255, 255))  # Ensure all images are 255x255
    ])
    
    for image in image_list:
        # Convert numpy array to tensor
        if isinstance(image, np.ndarray):
            if len(image.shape) == 2:  # Grayscale
                # Convert to RGB by duplicating channels
                image = np.stack([image, image, image], axis=2)
            
            # Make sure values are in range [0, 1]
            if image.max() > 1.0:
                image = image / 255.0
                
            tensor_image = torch.from_numpy(image).permute(2, 0, 1).float()
        else:
            tensor_image = transform(image)
            
        processed_images.append(tensor_image)
    
    # Stack all images into a batch
    return torch.stack(processed_images)

# Function to apply transformer to a set of images
def apply_transformer_fusion(image_list, model=None, in_channels=3):
    # Create or load the model
    if model is None:
        model = ImageTransformer(num_images=len(image_list), in_channels=in_channels)
    
    # Process the images
    processed_batch = preprocess_images(image_list)
    
    # Split the batch back into individual images for model input
    individual_images = [processed_batch[i:i+1] for i in range(processed_batch.shape[0])]
    
    # Apply the model
    with torch.no_grad():
        output = model(individual_images)
    
    # Convert back to numpy array
    output_image = output[0].permute(1, 2, 0).numpy()
    
    # Scale back to 0-255 range if needed
    if output_image.max() <= 1.0:
        output_image = (output_image * 255).astype(np.uint8)
    
    return output_image

# Example of using the transformer with your existing code
def create_transformer_animation(image_list, num_frames=100, in_channels=3, **kwargs):
    # Initialize the transformer model
    model = ImageTransformer(num_images=len(image_list), in_channels=in_channels, **kwargs)
    
    frames = []
    T = num_frames
    
    for i in range(T):
        # You could modify the model parameters over time to create animation effects
        # For example, adjusting attention weights or position embeddings
        
        # Apply transformer to images
        fused_image = apply_transformer_fusion(image_list, model)
        
        # Display every 10th frame
        if i % 10 == 0:
            display_rgb(fused_image)
        
        frames.append(fused_image)
    
    return frames