from __future__ import annotations
import os, json
from typing import Dict, List
from imageio import imwrite, mimsave
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def _save_frames(frames, out_dir: str, prefix: str):
    _ensure_dir(out_dir)
    for i, fr in enumerate(frames):
        fn = os.path.join(out_dir, f"{prefix}_{i:03d}.png")
        imwrite(fn, fr)

def _add_monitor_text(frame: np.ndarray, scene_id: str, frame_num: int, total_frames: int, intervention_info: str = None, frame_symbols: dict = None) -> np.ndarray:
    """
    Add monitor information to the bottom of the frame, intervention info to the top (if provided),
    and object ID labels on each object (if frame_symbols provided).
    """
    # Convert numpy array to PIL Image
    if frame.dtype != np.uint8:
        if frame.max() <= 1.0:
            frame = (frame * 255).astype(np.uint8)
        else:
            frame = frame.astype(np.uint8)
    
    # Create PIL Image
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    
    # Try to load a font, fall back to default if not available
    try:
        # Try to use a system font (adjust path as needed)
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 12)
        small_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 10)
    except (OSError, IOError):
        try:
            # Try alternative font path
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
            small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
        except (OSError, IOError):
            # Fall back to default font
            font = ImageFont.load_default()
            small_font = ImageFont.load_default()
    
    # Get image dimensions
    img_width, img_height = img.size
    padding = 2
    
    # Add object ID labels if frame symbols are provided
    if frame_symbols and "objects" in frame_symbols:
        for obj in frame_symbols["objects"]:
            obj_id = obj.get("id", "?")
            pos = obj.get("pos", [0.5, 0.5])  # normalized coordinates [0, 1]
            
            # Convert normalized coordinates to pixel coordinates
            x_px = int(pos[0] * img_width)
            y_px = int(pos[1] * img_height)
            
            # Get text dimensions for centering
            bbox = draw.textbbox((0, 0), obj_id, font=small_font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            # Position label above the object center
            label_x = x_px - text_width // 2
            label_y = y_px - 15  # Position above the object
            
            # Ensure label stays within image bounds
            label_x = max(2, min(label_x, img_width - text_width - 2))
            label_y = max(2, label_y)
            
            # Draw background circle/rectangle for object ID
            draw.ellipse([label_x-3, label_y-2, label_x+text_width+3, label_y+text_height+2], 
                        fill=(255, 255, 255, 200))  # Semi-transparent white background
            draw.ellipse([label_x-3, label_y-2, label_x+text_width+3, label_y+text_height+2], 
                        outline=(0, 0, 0), width=1)  # Black border
            
            # Draw object ID in black
            draw.text((label_x, label_y), obj_id, fill=(0, 0, 0), font=small_font)
    
    # Add intervention information at the top (if provided)
    if intervention_info:
        intervention_text = f"Intervention: {intervention_info}"
        bbox = draw.textbbox((0, 0), intervention_text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Position at top center
        x = (img_width - text_width) // 2
        y = 5
        
        # Draw background rectangle
        draw.rectangle([x-padding, y-padding, x+text_width+padding, y+text_height+padding], 
                       fill=(0, 0, 0, 180))  # Semi-transparent black
        
        # Draw text in bright yellow for intervention
        draw.text((x, y), intervention_text, fill=(255, 255, 0), font=font)
    
    # Add monitor information at the bottom
    monitor_text = f"{scene_id} | Frame {frame_num+1}/{total_frames}"
    bbox = draw.textbbox((0, 0), monitor_text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    # Position text at bottom left with some padding
    x = 5
    y = img_height - text_height - 5
    
    # Draw black background rectangle for better readability
    draw.rectangle([x-padding, y-padding, x+text_width+padding, y+text_height+padding], 
                   fill=(0, 0, 0, 180))  # Semi-transparent black
    
    # Draw white text
    draw.text((x, y), monitor_text, fill=(255, 255, 255), font=font)
    
    # Convert back to numpy array
    return np.array(img)

def _format_intervention_info(do_info: dict) -> str:
    """
    Format intervention information for display in GIF.
    """
    if not do_info:
        return ""
    
    op = do_info.get("op", "unknown")
    obj = do_info.get("obj", "?")
    t_intervene = do_info.get("t_intervene", None)
    
    # Format the intervention operation
    if op == "freeze":
        action_desc = f"{op} {obj}"
    elif op == "move":
        dx = do_info.get("dx", 0)
        dy = do_info.get("dy", 0)
        action_desc = f"{op} {obj} by ({dx:.2f}, {dy:.2f})"
    elif op == "change_color":
        color = do_info.get("color", [0, 0, 0])
        action_desc = f"{op} {obj} to RGB({color[0]}, {color[1]}, {color[2]})"
    elif op == "remove":
        action_desc = f"{op} {obj}"
    elif op == "set_pos":
        pos = do_info.get("pos", [0, 0])
        action_desc = f"{op} {obj} to ({pos[0]:.2f}, {pos[1]:.2f})"
    else:
        action_desc = f"{op} {obj}"
    
    # Add timing information if available
    if t_intervene is not None:
        return f"{action_desc} @ t={t_intervene}"
    else:
        return action_desc

def _save_gif(frames: List[np.ndarray], out_path: str, fps: int = 5, scene_id: str = None, intervention_info: dict = None, frame_symbols: List[dict] = None):
    """
    Save frames as an animated GIF with optional monitor information, intervention info, and object labels
    """
    processed_frames = []
    total_frames = len(frames)
    
    # Format intervention info for display
    formatted_intervention = _format_intervention_info(intervention_info) if intervention_info else None
    
    for i, frame in enumerate(frames):
        # Get symbols for this frame if available
        current_frame_symbols = frame_symbols[i] if frame_symbols and i < len(frame_symbols) else None
        
        # Add monitor information if scene_id is provided
        if scene_id:
            frame = _add_monitor_text(frame, scene_id, i, total_frames, formatted_intervention, current_frame_symbols)
        
        # Ensure frames are in the right format (uint8)
        if frame.dtype != np.uint8:
            if frame.max() <= 1.0:
                frame = (frame * 255).astype(np.uint8)
            else:
                frame = frame.astype(np.uint8)
        processed_frames.append(frame)
    
    mimsave(out_path, processed_frames, fps=fps, loop=0)

def export_pair(pair, out_dir: str, export_gif: bool = False, gif_fps: int = 5):
    """
    导出：
    - baseline/*.png
    - intervention/*.png
    - baseline.gif (if export_gif=True)
    - intervention.gif (if export_gif=True)
    - meta.json（含每帧符号与基本信息）
    """
    base_dir = os.path.join(out_dir, "baseline")
    intv_dir = os.path.join(out_dir, "intervention")
    _ensure_dir(out_dir)
    _save_frames(pair.baseline.frames, base_dir, "frame")
    _save_frames(pair.intervention.frames, intv_dir, "frame")

    # Export GIFs if requested
    if export_gif:
        # Extract scene_id from output directory path
        scene_id = os.path.basename(out_dir)
        
        # Get intervention information from metadata
        intervention_do = pair.meta.get("intervention", {})
        t_intervene = pair.meta.get("t_intervene", None)
        
        # Add timing information to intervention details
        if t_intervene is not None:
            intervention_do_with_timing = dict(intervention_do)
            intervention_do_with_timing["t_intervene"] = t_intervene
        else:
            intervention_do_with_timing = intervention_do
        
        baseline_gif_path = os.path.join(out_dir, "baseline.gif")
        intervention_gif_path = os.path.join(out_dir, "intervention.gif")
        
        # Baseline GIF without intervention info but with object labels
        _save_gif(pair.baseline.frames, baseline_gif_path, fps=gif_fps, 
                 scene_id=f"{scene_id}_baseline", frame_symbols=pair.baseline.symbols)
        
        # Intervention GIF with intervention info and object labels
        _save_gif(pair.intervention.frames, intervention_gif_path, fps=gif_fps, 
                 scene_id=f"{scene_id}_intervention", intervention_info=intervention_do_with_timing,
                 frame_symbols=pair.intervention.symbols)
        
        print(f"GIFs exported: {baseline_gif_path}, {intervention_gif_path}")

    meta: Dict = dict(pair.meta)
    meta["baseline_symbols"] = pair.baseline.symbols
    meta["intervention_symbols"] = pair.intervention.symbols

    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
        