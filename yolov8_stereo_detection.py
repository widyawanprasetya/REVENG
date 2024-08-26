import time
from picamera2 import Picamera2
import cv2
import numpy as np
import subprocess
import os
import sys
import select
import threading
from ultralytics import YOLO

class StereoCamera:
    def __init__(self, resolution=(640, 480), framerate=30):
        self.camera1 = None
        self.camera2 = None
        self.resolution = resolution
        self.framerate = framerate
        try:
            self.init_cameras()
        except Exception as e:
            print(f"Error initializing cameras: {e}")
            raise

    def init_cameras(self):
        print("Initializing camera 1...")
        self.camera1 = Picamera2(0)
        config1 = self.camera1.create_preview_configuration(main={"size": self.resolution, "format": "RGB888"})
        self.camera1.configure(config1)
        self.camera1.start()
        print("Camera 1 initialized")
        time.sleep(2)  # Wait before initializing the second camera

        print("Initializing camera 2...")
        self.camera2 = Picamera2(1)
        config2 = self.camera2.create_preview_configuration(main={"size": self.resolution, "format": "RGB888"})
        self.camera2.configure(config2)
        self.camera2.start()
        print("Camera 2 initialized")
        time.sleep(2)  # Allow cameras to warm up

        print("Both cameras initialized")

    def capture_stereo(self):
        if self.camera1 is None or self.camera2 is None:
            raise RuntimeError("Cameras not properly initialized")
        frame1 = self.camera1.capture_array()
        frame2 = self.camera2.capture_array()
        return frame1, frame2

    def close(self):
        if self.camera1:
            self.camera1.stop()
        if self.camera2:
            self.camera2.stop()
        print("Cameras stopped")

def ffplay_display(pipe, width, height):
    cmd = ['ffplay',
           '-f', 'rawvideo',
           '-pixel_format', 'bgr24',
           '-video_size', f'{width}x{height}',
           '-i', 'pipe:0',
           '-window_title', 'Stereo Camera Feed with YOLO Detection']
    process = subprocess.Popen(cmd, stdin=pipe)
    return process

def keyboard_input(stop_event):
    global is_recording, out
    while not stop_event.is_set():
        if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
            key = sys.stdin.read(1).lower()
            if key == 'r':
                if not is_recording:
                    timestamp = time.strftime("%Y%m%d-%H%M%S")
                    output_filename = os.path.join(recordings_dir, f'stereo_yolo_output_{timestamp}.mp4')
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out = cv2.VideoWriter(output_filename, fourcc, framerate, (resolution[0] * 2, resolution[1]))
                    is_recording = True
                    print(f"Started recording: {output_filename}")
                else:
                    is_recording = False
                    out.release()
                    out = None
                    print("Stopped recording")
            elif key == 'q':
                print("Quit command received")
                stop_event.set()
                break

def main():
    global is_recording, out, recordings_dir, resolution, framerate

    resolution = (640, 480)
    framerate = 30
    recordings_dir = 'recordings'
    os.makedirs(recordings_dir, exist_ok=True)

    # Initialize YOLO model
    model = YOLO('yolov8n.pt')  # Use the nano model for better performance on Raspberry Pi

    try:
        stereo_cam = StereoCamera(resolution=resolution, framerate=framerate)
    except Exception as e:
        print(f"Failed to initialize stereo camera: {e}")
        return

    # Create a pipe for FFplay
    pipe_r, pipe_w = os.pipe()
    ffplay_process = ffplay_display(os.fdopen(pipe_r, 'rb'), width=resolution[0]*2, height=resolution[1])

    out = None
    is_recording = False
    print("Press 'r' to start/stop recording, 'q' to quit")

    stop_event = threading.Event()
    input_thread = threading.Thread(target=keyboard_input, args=(stop_event,))
    input_thread.start()

    try:
        while not stop_event.is_set():
            frame1, frame2 = stereo_cam.capture_stereo()
            
            # Run YOLO detection on both frames
            results1 = model(frame1)
            results2 = model(frame2)
            
            # Draw detection results on frames
            annotated_frame1 = results1[0].plot()
            annotated_frame2 = results2[0].plot()
            
            stitched_frame = np.hstack((annotated_frame1, annotated_frame2))

            if is_recording and out is not None:
                out.write(cv2.cvtColor(stitched_frame, cv2.COLOR_RGB2BGR))

            # Add "REC" text if recording
            if is_recording:
                cv2.putText(stitched_frame, "REC", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Write the frame to FFplay
            os.write(pipe_w, cv2.cvtColor(stitched_frame, cv2.COLOR_RGB2BGR).tobytes())

    except Exception as e:
        print(f"Error during capture: {e}")
    finally:
        stop_event.set()
        input_thread.join()
        if out is not None:
            out.release()
        stereo_cam.close()
        ffplay_process.terminate()
        os.close(pipe_w)
        print("Program ended")

if __name__ == "__main__":
    main()