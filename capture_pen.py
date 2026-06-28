"""
Capture 150 images of pen
Captures 1 image every 5 seconds from webcam
"""

import cv2
import os
import time
from datetime import datetime

def capture_pen_images():
    """
    Capture 150 images of pen (1 image every 5 seconds)
    """
    
    # Create output directory
    output_dir = "pen_images"
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*50)
    print("Pen Image Capture Script")
    print("="*50)
    print(f"Output directory: {output_dir}")
    print("Will capture 150 images (1 image every 5 seconds)")
    print("Total time: 750 seconds (12.5 minutes)")
    print("\nINSTRUCTIONS:")
    print("1. Position your pen in front of the webcam")
    print("2. Show different angles and orientations")
    print("3. Press ENTER when ready to start capturing")
    print("4. Move the pen around during capture for variety")
    print("="*50)
    
    input("\nPress ENTER to start capturing...")
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print("\nStarting capture...")
    print("Move your pen around for variety!")
    
    # Capture parameters
    total_images = 150
    interval = 5  # 5 seconds per image
    
    start_time = time.time()
    captured_count = 0
    
    try:
        while captured_count < total_images:
            ret, frame = cap.read()
            
            if not ret:
                print("Error: Could not read frame")
                break
            
            # Display live preview
            cv2.imshow('Pen Capture - Press ESC to stop', frame)
            
            # Capture image at interval
            elapsed_time = time.time() - start_time
            expected_capture_time = captured_count * interval
            
            if elapsed_time >= expected_capture_time:
                # Generate filename with timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                filename = f"{output_dir}/pen_{captured_count + 1:03d}_{timestamp}.jpg"
                
                # Save image
                cv2.imwrite(filename, frame)
                captured_count += 1
                
                # Progress
                progress = (captured_count / total_images) * 100
                print(f"Captured {captured_count}/{total_images} images ({progress:.1f}%)")
            
            # Check for ESC key to stop early
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC key
                print("\nCapture stopped by user")
                break
            
            # Check if time exceeded
            if elapsed_time > duration + 2:  # 2 second buffer
                print("\nTime limit exceeded")
                break
    
    except KeyboardInterrupt:
        print("\nCapture interrupted")
    
    finally:
        # Release resources
        cap.release()
        cv2.destroyAllWindows()
    
    # Summary
    elapsed = time.time() - start_time
    print("\n" + "="*50)
    print("CAPTURE SUMMARY")
    print("="*50)
    print(f"Images captured: {captured_count}/{total_images}")
    print(f"Time elapsed: {elapsed:.2f} seconds")
    print(f"Average rate: {captured_count/elapsed:.2f} images/second")
    print(f"Output directory: {output_dir}/")
    print("="*50)
    
    if captured_count == total_images:
        print("✓ Successfully captured all 150 images!")
    else:
        print(f"⚠ Captured only {captured_count} images")
    
    print("\nNext steps:")
    print("1. Review the captured images in the output folder")
    print("2. Delete any blurry or poor quality images")
    print("3. Use LabelImg to annotate the images")
    print("4. Move annotated files to custom_dataset/images/ and custom_dataset/labels/")

if __name__ == "__main__":
    capture_pen_images()
