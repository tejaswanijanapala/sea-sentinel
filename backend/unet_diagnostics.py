import os
import sys
import time
import numpy as np
import cv2
import torch
import glob

sys.path.insert(0, os.path.join(os.getcwd(), 'backend'))
from ai.segmentation.unet_segmenter import UNetSegmenter

def find_test_image():
    project_dir = os.getcwd()
    images = glob.glob(os.path.join(project_dir, '**', '*.jpg'), recursive=True) + \
             glob.glob(os.path.join(project_dir, '**', '*.png'), recursive=True)
    images = [img for img in images if 'debug' not in img and 'mock' not in img]
    if not images:
        print("No test image found. Generating a synthetic one.")
        img = np.zeros((512, 512), dtype=np.uint8)
        cv2.rectangle(img, (100, 100), (400, 400), 255, -1)
        cv2.circle(img, (256, 256), 50, 0, -1)
        return img
    return cv2.imread(images[0], cv2.IMREAD_GRAYSCALE)

def run_diagnostics():
    print("===========================================================")
    print("PHASE 2 & 3: MODEL LOADING AND ARCHITECTURE")
    print("===========================================================")
    
    segmenter = UNetSegmenter(
        checkpoint_path='backend/models/unet/attention_unet_best.pt', 
        model_type="attention_unet", 
        device="cpu"
    )
    
    if not segmenter.is_model_loaded:
        print("U-NET CHECKPOINT INVALID OR INCOMPATIBLE")
        return

    print(f"MODEL PATH: {segmenter.checkpoint_path}")
    print(f"MODEL FILE SIZE: {os.path.getsize(segmenter.checkpoint_path) / 1024 / 1024:.2f} MB")
    print(f"MODEL ARCHITECTURE: {segmenter.model.__class__.__name__}")
    
    total_params = sum(p.numel() for p in segmenter.model.parameters())
    print(f"MODEL PARAMETERS: {total_params:,}")
    
    try:
        first_layer = next(segmenter.model.parameters())
        print(f"INPUT CHANNELS: {first_layer.shape[1]}")
        print(f"OUTPUT CHANNELS: 1 (Sigmoid)")
        print(f"MODEL DEVICE: {first_layer.device}")
        print(f"MODEL DTYPE: {first_layer.dtype}")
    except Exception as e:
        print(f"Could not inspect layers: {e}")

    print("\n===========================================================")
    print("PHASE 4, 5 & 6: INPUT TENSOR & PREPROCESSING")
    print("===========================================================")
    
    img_gray = find_test_image()
    if img_gray is None:
        print("Error: Could not load or generate image.")
        return

    h, w = img_gray.shape[:2]
    print(f"Original Image Size: {w}x{h}")

    img_size = segmenter.img_size
    resized = cv2.resize(img_gray, (img_size, img_size), interpolation=cv2.INTER_AREA)
    norm_img = resized.astype(np.float32) / 255.0
    
    print(f"Model Input Size (after resize): {img_size}x{img_size}")
    print(f"Interpolation: INTER_AREA")
    print(f"Normalization: / 255.0")
    
    tensor = torch.from_numpy(norm_img).unsqueeze(0).unsqueeze(0).float()
    print(f"\nINPUT TENSOR:")
    print(f"shape = {list(tensor.shape)}")
    print(f"dtype = {tensor.dtype}")
    print(f"min = {tensor.min().item():.4f}")
    print(f"max = {tensor.max().item():.4f}")
    print(f"mean = {tensor.mean().item():.4f}")
    print(f"std = {tensor.std().item():.4f}")

    print("\n===========================================================")
    print("PHASE 7 & 8: RAW MODEL OUTPUT STATISTICS")
    print("===========================================================")
    
    with torch.no_grad():
        logits = segmenter.model(tensor)
        probs = torch.sigmoid(logits).squeeze().cpu().numpy()
        
    print("RAW LOGITS:")
    print(f"min = {logits.min().item():.4f}")
    print(f"max = {logits.max().item():.4f}")
    print(f"mean = {logits.mean().item():.4f}")
    
    print("\nRAW PROBABILITY (Sigmoid):")
    print(f"min = {probs.min():.4f}")
    print(f"max = {probs.max():.4f}")
    print(f"mean = {probs.mean():.4f}")
    
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    for t in thresholds:
        pct = (probs > t).mean() * 100
        print(f"percentage > {t:.1f} = {pct:.2f}%")

    print("\n===========================================================")
    print("PHASE 9 & 10: SAVING MAPS TO debug/unet/")
    print("===========================================================")
    
    os.makedirs('debug/unet', exist_ok=True)
    cv2.imwrite('debug/unet/original.png', img_gray)
    cv2.imwrite('debug/unet/preprocessed.png', (norm_img * 255).astype(np.uint8))
    
    prob_img = (probs * 255).astype(np.uint8)
    cv2.imwrite('debug/unet/probability.png', prob_img)
    color_prob = cv2.applyColorMap(prob_img, cv2.COLORMAP_JET)
    cv2.imwrite('debug/unet/probability_color.png', color_prob)
    
    for t in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        binary = (probs >= t).astype(np.uint8) * 255
        cv2.imwrite(f'debug/unet/binary_{int(t*10):02d}.png', binary)
        
    print("Saved all debug masks to debug/unet/")
    
    print("\n===========================================================")
    print("PHASE 12, 13 & 14: MASK RESIZING & RECTANGULAR CHECK")
    print("===========================================================")
    
    full_prob = cv2.resize(probs, (w, h), interpolation=cv2.INTER_LINEAR)
    binary_mask = (full_prob >= 0.5).astype(np.uint8)
    
    print(f"Original: {w}x{h}")
    print(f"Model: {img_size}x{img_size}")
    print(f"Probability Map: {w}x{h}")
    print(f"Binary Mask: {binary_mask.shape[1]}x{binary_mask.shape[0]}")
    
    if binary_mask.shape[:2] != (h, w):
        print("FAIL: Mask dimensions do not match original image.")
    else:
        print("PASS: Mask dimensions match.")
        
    print("\nExtracting Candidate Objects (Checking for rectangular bug)...")
    objects = segmenter.extract_candidate_objects(binary_mask, full_prob)
    print(f"Found {len(objects)} candidate objects.")
    for idx, obj in enumerate(objects):
        print(f"  Object {idx+1}:")
        print(f"    Class: {obj['class']}")
        print(f"    Aspect Ratio: {obj['aspect_ratio']}")
        print(f"    Compactness: {obj['compactness']}")
        print(f"    Polygon points: {len(obj['polygon'])}")
        if len(obj['polygon']) == 4:
            print("    WARNING: Polygon simplified to 4 points (possible rectangle bug!)")

if __name__ == "__main__":
    run_diagnostics()
