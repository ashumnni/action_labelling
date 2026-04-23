import os
import cv2
import base64
import json
import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm

# Load environment variables (API Key)
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found. Please set it in the .env file.")

client = OpenAI(api_key=api_key)

ACTIVE_KEYWORDS = ["cut", "wash", "mix", "pick", "place", "clean", "open", "close", "pour", "stir", "sew", "trim", "measure", "scrub", "wipe"]

SYSTEM_PROMPT = """You are an expert temporal action labeling system for long videos.
Your task is to analyze structured video data and produce accurate, human-quality action segments.

## INPUT
You will receive a sequence of frames from a video. Each frame is followed by its timestamp in seconds.

## OUTPUT FORMAT (STRICT JSON ONLY)
{
  "segments": [
    {
      "action": "<descriptive_action_name_exactly_10_to_12_words>",
      "start": <start_time_in_seconds>,
      "end": <end_time_in_seconds>,
      "confidence": <0 to 1>
    }
  ]
}
No explanation. No extra text. Each action name MUST be between 10 and 12 words long.

## CORE OBJECTIVE
Convert noisy, fragmented, or descriptive temporal data into clean, continuous action segments with 100% TEMPORAL COVERAGE.

## RULES (STRICT)
1. 100% Coverage: There MUST be no gaps in the timeline. Every second must be accounted for.
2. Contiguous Segments: The end of one segment should be the start of the next.
3. Density: If no specific action is visible, label it as "Transition state" or "Preparation" with a 10-12 word description.
4. Temporal Consistency: Merge consecutive frames with the same action.
5. Multi-Label Support: Multiple actions can happen at the same time.
6. Confidence Estimation: Assign higher confidence to longer, stable segments.
"""

TAL_SYSTEM_PROMPT = """You are a highly accurate Robot Command Context Temporal Action Localization (TAL) engine.
Your task is to identify meaningful human action segments and return a precise command context.

## OBJECTIVE
Identify NON-OVERLAPPING atomic activity segments: (start, end, command) with 100% TEMPORAL COVERAGE.

## RULE 1: COMMAND CONTEXT
- Every label MUST be a command in imperative form (e.g., "Adjust fabric", "Sew fabric").
- Every label MUST be exactly 6 to 8 words long.
- DO NOT use subject words like "person" or "user".
- DO NOT use continuous tense (no "ing").
- Example: "Carefully guide the white fabric through machine" (7 words)
- Example: "Pick up the metal scissors from table" (7 words)

## RULE 2: TEMPORAL PRECISION
- No gaps. 100% temporal coverage.
- Every second must be labeled. 

## OUTPUT FORMAT (STRICT JSON ONLY)
{
  "segments": [
    {
      "label": "<exactly_6_to_8_word_command>",
      "start": <start_time_in_seconds>,
      "end": <end_time_in_seconds>
    }
  ]
}
No explanations. No extra text.
"""

def encode_image(image):
    _, buffer = cv2.imencode(".jpg", image)
    return base64.b64encode(buffer).decode("utf-8")

def get_labels_for_window(video_path, start_time, end_time, sampling_interval=10, system_prompt=SYSTEM_PROMPT):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Setup screenshot directory for this video
    video_basename = os.path.basename(video_path)
    screenshot_dir = os.path.join("dataset", "screenshots", video_basename)
    os.makedirs(screenshot_dir, exist_ok=True)
    
    frames_for_api = []
    for t in range(int(start_time), int(end_time), sampling_interval):
        frame_idx = int(t * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret: break
        
        # Save screenshot for the dataset
        screenshot_path = os.path.join(screenshot_dir, f"frame_{frame_idx}_ts{t}.jpg")
        cv2.imwrite(screenshot_path, frame)
        
        # Resize for API efficiency
        resized_frame = cv2.resize(frame, (512, 288))
        base64_img = encode_image(resized_frame)
        frames_for_api.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}", "detail": "low"}})
        frames_for_api.append({"type": "text", "text": f"Timestamp: {t}s"})
    
    cap.release()
    
    print(f"  Calling GPT-4o for window {start_time}s - {end_time}s ({len(frames_for_api)//2} frames)...")
    
    max_retries = 3
    retry_delay = 5
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": [
                        {"type": "text", "text": f"Analyze this {start_time}s to {end_time}s segment of a video. Provide temporal action labeling following the system instructions."},
                        *frames_for_api
                    ]}
                ],
                response_format={"type": "json_object"}
            )
            content = response.choices[0].message.content
            data = json.loads(content)
            if "segments" in data: 
                # Normalize key to 'action' for internal consistency if needed, 
                # but we'll try to keep them as is and handle in merger
                return data["segments"]
            return data
        except Exception as e:
            if "rate_limit_exceeded" in str(e).lower() and attempt < max_retries - 1:
                print(f"  Rate limit hit for {start_time}-{end_time}. Retrying in {retry_delay}s (Attempt {attempt+1}/{max_retries})...")
                time.sleep(retry_delay)
                retry_delay *= 2 # Exponential backoff
                continue
            print(f"  Error in window {start_time}-{end_time}: {e}")
            return []

def update_master_results(video_name, segments, master_file="master_results.json"):
    data = {}
    if os.path.exists(master_file):
        try:
            with open(master_file, "r") as f:
                loaded = json.load(f)
                if isinstance(loaded, dict):
                    data = loaded
                else:
                    data = {} # Reset to dict if it was a list
        except json.JSONDecodeError:
            data = {}
    
    data[video_name] = segments
    
    with open(master_file, "w") as f:
        json.dump(data, f, indent=4)
    print(f"Master results updated for {video_name} in {master_file}")

def merge_segments(all_segments, duration, sparse=False):
    merged = []
    if not all_segments:
        return []
        
    # Normalize keys and validate timestamps strictly
    valid_segments = []
    for s in all_segments:
        try:
            start = float(s.get('start', 0))
            end = float(s.get('end', 0))
            if start >= end:
                continue # Drop invalid segments
            s['start'], s['end'] = start, end
            
            if 'label' in s and 'action' not in s:
                s['action'] = s['label']
            elif 'action' in s and 'label' not in s:
                s['label'] = s['action']
            
            # Ensure robot_command exists
            if 'robot_command' not in s:
                s['robot_command'] = ""
                
            valid_segments.append(s)
        except (ValueError, TypeError):
            continue

    if not valid_segments:
        return []
        
    valid_segments.sort(key=lambda x: x['start'])
    
    if not sparse:
        # Full coverage mode: ensure first segment starts at 0
        valid_segments[0]['start'] = 0
    
    curr = valid_segments[0]
    for i in range(1, len(valid_segments)):
        next_seg = valid_segments[i]
        
        # If same action and same command, just merge them
        if next_seg['action'] == curr['action'] and next_seg.get('robot_command') == curr.get('robot_command'):
            curr['end'] = max(curr['end'], next_seg['end'])
        else:
            if not sparse:
                # If there's a gap between segments, extend the current one to meet the next one
                if next_seg['start'] > curr['end']:
                    curr['end'] = next_seg['start']
                # If they overlap
                elif next_seg['start'] < curr['end']:
                    # If next_seg is completely swallowed by curr, skip it
                    if next_seg['end'] <= curr['end']:
                        continue
                    # Otherwise, adjust next_seg to start where curr ends
                    next_seg['start'] = curr['end']
            
            merged.append(curr)
            curr = next_seg
    
    if not sparse:
        # Ensure last segment ends at the very end of the video
        curr['end'] = duration
    merged.append(curr)
    
    # Final cleanup for TAL mode: enforce float types and strict key order
    if sparse:
        final_tal = []
        for s in merged:
            # Reconstruct to ensure key order: start, end, label, robot_command
            clean_seg = {
                "start": float(s['start']),
                "end": float(s['end']),
                "label": s.get('label', s.get('action', 'Unknown')),
                "robot_command": s.get('robot_command', '')
            }
            final_tal.append(clean_seg)
        return final_tal

    # For dense mode, we still want a clean output format
    clean_final = []
    for s in merged:
        clean_final.append({
            "start": float(s['start']),
            "end": float(s['end']),
            "label": s.get('label', s.get('action', 'Unknown')),
            "robot_command": s.get('robot_command', '')
        })
    return clean_final

def run_labeling(video_path, chunk_size=300, sampling_interval=10, output_file="inference_results.json", workers=4, mode="standard", max_duration=None, format="descriptive"):
    video_basename = os.path.basename(video_path)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    if max_duration:
        duration = min(duration, max_duration)
    cap.release()
    
    # Determine settings based on format
    is_sparse = (format == "tal")
    active_prompt = TAL_SYSTEM_PROMPT if format == "tal" else SYSTEM_PROMPT
    
    # STAGE 1: COARSE LABELING
    sampling_interval = 5 # Use 5s as default for better hooks
    chunk_size = 150      # Smaller chunks (30 frames) to avoid TPM rate limits
    print(f"STAGE 1: Running Coarse Labeling ({sampling_interval}s interval) in {format} mode...")
    coarse_segments = []
    chunks = []
    for start in range(0, int(duration), chunk_size):
        end = min(start + chunk_size, duration)
        chunks.append((start, end))
    
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_chunk = {
            executor.submit(get_labels_for_window, video_path, s, e, sampling_interval, active_prompt): (s, e)
            for s, e in chunks
        }
        for future in tqdm(as_completed(future_to_chunk), total=len(chunks), desc="Coarse Pass"):
            res = future.result()
            if isinstance(res, list): coarse_segments.extend(res)

    all_segments = merge_segments(coarse_segments, duration, sparse=False)

    # STAGE 2: ADAPTIVE REFINEMENT (Optional)
    if mode == "adaptive":
        print(f"STAGE 2: Identifying Active Regions for Refinement...")
        refinement_tasks = []
        regions_to_refine = []
        for seg in all_segments:
            # For trigger detection, handle both keys
            action_text = seg.get('action', seg.get('label', '')).lower()
            is_active = any(kw in action_text for kw in ACTIVE_KEYWORDS)
            is_low_conf = seg.get('confidence', 1.0) < 0.85
            
            if is_active or is_low_conf:
                # Add the segment itself AND a 5s buffer around its boundaries to refinement
                ref_start = max(0, seg['start'] - 5)
                ref_end = min(duration, seg['end'] + 5)
                regions_to_refine.append((ref_start, ref_end))
                
                curr_ref = ref_start
                while curr_ref < ref_end:
                    chunk_end = min(curr_ref + 20, ref_end) # 20s max for high-res chunks
                    refinement_tasks.append((curr_ref, chunk_end))
                    curr_ref = chunk_end
        
        if refinement_tasks:
            print(f"STAGE 3: Running Precision Refinement (1s interval) on {len(refinement_tasks)} regions...")
            precise_segments = []
            with ThreadPoolExecutor(max_workers=workers) as executor:
                future_to_ref = {
                    executor.submit(get_labels_for_window, video_path, s, e, 1, active_prompt): (s, e)
                    for s, e in refinement_tasks
                }
                for future in tqdm(as_completed(future_to_ref), total=len(refinement_tasks), desc="Precision Pass"):
                    res = future.result()
                    if isinstance(res, list): precise_segments.extend(res)
            
            # Combine results: Precise segments overwrite coarse ones
            def is_in_refined_region(s, e):
                for rs, re in regions_to_refine:
                    if s >= rs and e <= re: return True
                return False
            
            non_refined_segments = [s for s in all_segments if not is_in_refined_region(s['start'], s['end'])]
            
            all_segments = non_refined_segments + precise_segments
            all_segments = merge_segments(all_segments, duration, sparse=False)

    with open(output_file, "w") as f:
        json.dump(all_segments, f, indent=4)
    
    update_master_results(video_basename, all_segments)
    print(f"Final labels saved to {output_file}. Detected {len(all_segments)} segments.")
    return all_segments

    with open(output_file, "w") as f:
        json.dump(merged, f, indent=4)
    
    # Update master results
    update_master_results(video_basename, merged)
    
    print(f"Final labels saved to {output_file}. Detected {len(merged)} segments.")
    return merged

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--interval", type=int, default=10)
    parser.add_argument("--chunk", type=int, default=300)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--mode", type=str, choices=["standard", "adaptive"], default="standard")
    parser.add_argument("--format", type=str, choices=["descriptive", "tal"], default="descriptive")
    parser.add_argument("--max_duration", type=int, default=None)
    parser.add_argument("--output", type=str, default="inference_results.json")
    args = parser.parse_args()
    run_labeling(args.video, args.chunk, args.interval, args.output, args.workers, args.mode, args.max_duration, args.format)
