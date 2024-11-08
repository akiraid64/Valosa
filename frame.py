port cv2
import os
from PIL import Image

def extract_frames(video_path, output_folder='frames', frame_interval=30, max_frames=20):
    """
    Extract frames from a video file and save them as images in a folder named 'frames'.

    Args:
        video_path (str): Path to the video file
        output_folder (str): Folder to save extracted frames (default: 'frames')
        frame_interval (int): Extract one frame every N frames
        max_frames (int): Maximum number of frames to extract

    Returns:
        list: Paths of saved frame images
    """
    # Set the default folder to 'frames'
    output_folder = 'frame'
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    print(f"Output folder '{output_folder}' created or already exists.")

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video file")
        return []

    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    print(f"Video properties:")
    print(f"- Total frames: {total_frames}")
    print(f"- FPS: {fps}")
    print(f"- Duration: {total_frames/fps:.2f} seconds")

    frame_count = 0
    saved_frames = 0
    saved_paths = []

    while saved_frames < max_frames:
        ret, frame = cap.read()

        if not ret:
            break

        # Process every Nth frame
        if frame_count % frame_interval == 0:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Save frame using PIL for better quality
            frame_pil = Image.fromarray(frame_rgb)
            frame_path = os.path.join(output_folder, f'frame_{saved_frames:03d}.jpg')
            frame_pil.save(frame_path, quality=95)

            saved_paths.append(frame_path)
            saved_frames += 1

            print(f"Saved frame {saved_frames}/{max_frames} to '{frame_path}'")

        frame_count += 1

    cap.release()
    print(f"\nExtracted {len(saved_paths)} frames to '{output_folder}' folder")
    return saved_paths

# Example usage:
video_path = 'videoplayback.mp4'  # Replace with the actual video path

# Extract frames and save them to 'frames' folder
frames = extract_frames(
    video_path=video_path,
    frame_interval=60,  # Extract one frame every 30 frames
    max_frames=100  # Extract a maximum of 20 frames
)

# Verify the extracted frames
if frames:
    print("\nList of extracted frames:")
    for frame_path in frames:
        print(frame_path)
else:
    print("No frames were extracted.")
