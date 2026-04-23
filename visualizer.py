import cv2
import json
import os
import argparse
from tqdm import tqdm

def draw_hud(frame, action_text, confidence, timestamp):
    h, w, _ = frame.shape
    bar_h = int(h * 0.15) # Slimmer bar
    
    bg_color = (0, 0, 0)
    
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - bar_h), (w, h), bg_color, -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    
    font = cv2.FONT_HERSHEY_DUPLEX
    scale = h / 1100.0
    color_white = (255, 255, 255)
    color_grey = (180, 180, 180)
    
    # Render Command Context
    cmd_text = f"COMMAND: {action_text.upper()}"
    cv2.putText(frame, cmd_text, (30, h - int(bar_h * 0.55)), font, scale * 1.1, (0, 255, 0), 2, cv2.LINE_AA)
    
    stats_label = f"TIMESTAMP: {timestamp:06.2f}s | CONFIDENCE: {confidence:.2%}"
    cv2.putText(frame, stats_label, (30, h - int(40 * scale)), font, scale * 0.6, color_grey, 1, cv2.LINE_AA)

def get_action_at_time(segments, current_time):
    for seg in segments:
        if seg['start'] <= current_time <= seg['end']:
            return seg
    return {"label": "Idle/Transition", "robot_command": "Wait", "confidence": 1.0}

def run_visualization(video_path, master_json_path, output_path):
    if not os.path.exists(video_path):
        print(f"Error: Video {video_path} not found.")
        return
        
    # Load segments
    with open(master_json_path, "r") as f:
        data = json.load(f)
    
    if isinstance(data, list):
        # Direct list of segments (our high-precision JSON)
        segments = data
    else:
        # Master ledger format
        video_key = os.path.basename(video_path)
        if video_key not in data:
            print(f"Error: No labels found for {video_key} in {master_json_path}")
            return
        segments = data[video_key]
    
    # Video setup
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Use MP4V for compatibility
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    
    print(f"Rendering visualized video to {output_path}...")
    print(f"Video: {w}x{h} @ {fps} FPS")
    
    # Read and process frames
    pbar = tqdm(total=total_frames)
    last_action = None
    change_pulse_frames = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_idx = cap.get(cv2.CAP_PROP_POS_FRAMES)
        current_time = frame_idx / fps
        
        # Get labeling data
        segment = get_action_at_time(segments, current_time)
        action = segment.get('label', segment.get('action', 'Unknown'))
        confidence = segment.get('confidence', 1.0)
        
        # Overlay HUD
        draw_hud(frame, action, confidence, current_time)
        
        # Write frame
        out.write(frame)
        pbar.update(1)
        
    cap.release()
    out.release()
    pbar.close()
    print(f"Visualization complete: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True, help="Input video filename")
    parser.add_argument("--master", type=str, default="master_results.json", help="Path to master results JSON")
    parser.add_argument("--output", type=str, help="Output video filename")
    args = parser.parse_args()
    
    output_filename = args.output if args.output else args.video.replace(".", "_labeled.")
    run_visualization(args.video, args.master, output_filename)
