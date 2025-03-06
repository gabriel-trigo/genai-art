import numpy as np
from colorsys import hsv_to_rgb

def generate_harmonious_colors(T):
    """
    Generate a color palette with harmonious color relationships
    """
    colors = []
    # Use color wheel principles for more pleasing color generation
    for i in range(T):
        # Adjust hue to create smooth color transitions
        hue = (0.618033988749895 * i) % 1.0  # Golden ratio method for even color distribution
        saturation = 0.7  # Maintain consistent saturation
        value = 0.9  # Maintain consistent brightness
        
        # Convert HSV to RGB
        rgb = hsv_to_rgb(hue, saturation, value)
        colors.append(np.array(rgb))
    return np.array(colors)
