import numpy as np
import tensorflow as tf

# activation functions
def sig(x): # smooth between -1 and 1
    return 1./(1. + np.exp(-x))

def tanh(x): # smooth between 0 and 1
    return np.tanh(x)
    # return tf.keras.activations.tanh(x).numpy()

def gelu(x): # soft clip negative
    # return 0.5*x*(1 + np.tanh(np.sqrt(2/np.pi)*(x + 0.044715*x**3)))
    return tf.keras.activations.gelu(x).numpy()

# remapping 
def remap_wave_function(
        image, 
        amp_x=15, 
        freq_x=2*np.pi/10, 
        phase_x=0,
        amp_y=15, 
        freq_y=2*np.pi/10, 
        phase_y=0):
    h, w = image.shape[:2]
    output = np.ones_like(image) * 255.
    
    for x in range(h):
        for y in range(w):
            # Remap the x and y coordinates using sine functions.
            new_x = int(np.clip(x + amp_x * np.sin(freq_x * y + phase_x), 0, h - 1))
            new_y = int(np.clip(y + amp_y * np.sin(freq_y * x + phase_y), 0, w - 1))
            
            # Assign the pixel from the original image to the new coordinates.
            output[new_x, new_y] = image[x, y]
    
    return output


def compute_attention_map(h, w, sigma=50):
  # attention map that dense in the center
    x = np.arange(w)
    y = np.arange(h)
    x, y = np.meshgrid(x, y)
    cx, cy = w / 2, h / 2
    attention = np.exp(-((x - cx)**2 + (y - cy)**2) / (2 * sigma**2))
    return attention

def attention_based_remap1(image, attention, amplitude=20):
    h, w = image.shape
    output = np.zeros_like(image)
    cx, cy = h / 2, w / 2

    # Remap each pixel based on its attention weight.
    for x in range(h):
        for y in range(w):
            # Compute a normalized displacement vector from the center.
            # Here, (x - cx, y - cy) indicates the direction from the center.
            dx = (x - cx) / cx  # Normalize to [-1, 1]
            dy = (y - cy) / cy  # Normalize to [-1, 1]
            
            # The attention weight determines how strong the displacement is.
            # Multiply by amplitude to control the maximum shift.
            shift_x = int(np.clip(x + amplitude * attention[x, y] * dx, 0, h - 1))
            shift_y = int(np.clip(y + amplitude * attention[x, y] * dy, 0, w - 1))
            
            # Map the original pixel to the new location.
            output[shift_x, shift_y] = image[x, y]
    
    return output

def attention_based_remap2(image, attention_map, amplitude=20):
    h, w = image.shape
    output = np.zeros_like(image)
    
    # For each pixel, adjust its displacement by the corresponding attention weight.
    for x in range(h):
        for y in range(w):
            # Compute a simple displacement based on attention weight.
            # Here, higher attention means a larger shift.
            shift = int(amplitude * attention_map[x, y])
            
            new_y = (y + shift) % w
            new_x = (x + shift) % h
            
            output[new_x, new_y] = image[x, y]
    return output

def linear_remap(x, y, h, w, w11, w12, w21, w22, b1, b2):
    x_hat = w11*x/h + w21*y/w + b1
    y_hat = w12*x/h + w22*y/w + b2
    return x_hat, y_hat


# color changes
def mix_rgb_channels(r, g, b, alpha, beta, gamma):
    r_hat = r + alpha * (g - r)
    g_hat = g + beta * (b - g)
    b_hat = b + gamma * (r - b)
    return r_hat, g_hat, b_hat

def oscillating_function(t, offset, amplitude, frequency):
    return amplitude*np.sin(frequency * t + offset)

def color_harmony_oscillation(T):
    """
    Create more controlled oscillating parameters
    """
    params = []
    for i in range(T):
        # Use smoother, more controlled oscillations
        w11 = 2 + np.sin(2 * np.pi * i / T) * 1.5  # Oscillate between 0.5 and 3.5
        w12 = 2 + np.cos(2 * np.pi * i / T) * 1.5  # Slightly out of phase
        w21 = 1.5 + np.sin(2 * np.pi * i / T) * 1
        w22 = 1.5 + np.cos(2 * np.pi * i / T) * 1
        
        # More controlled bias terms
        b1 = 25 * np.sin(2 * np.pi * i / T)
        b2 = 25 * np.cos(2 * np.pi * i / T)
        
        params.append((w11, w12, w21, w22, b1, b2))
    
    return params

# pixel transformation
def linear_transformation(image, w11, w12, w21, w22, b1, b2):
    return w11*image + w12*image.T + b1, w21*image + w22*image.T + b2

def clipped_linear_remap(image, w11, w12, w21, w22, b1, b2):
    
    w11 = np.clip(w11, -2, 2)
    w12 = np.clip(w12, -2, 2)
    w21 = np.clip(w21, -2, 2)
    w22 = np.clip(w22, -2, 2)
    
    # Apply transformations with controlled scaling
    X1 = w11 * image + w12 * image.T + b1
    X2 = w21 * image + w22 * image.T + b2
    
    return X1, X2