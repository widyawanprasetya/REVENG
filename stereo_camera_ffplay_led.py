import time
from picamera2 import Picamera2
import cv2
import numpy as np
import subprocess
import os
import sys
import select
import threading
import gpiod
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Global variables
gpio_available = False
chip = None
record_button_line = None
quit_button_line = None
led_line = None
is_recording = False
out = None
stop_event = threading.Event()

def setup_gpio():
    global gpio_available, chip, record_button_line, quit_button_line, led_line
    try:
        chip = gpiod.Chip('/dev/gpiochip4')
        record_button_line = chip.get_line(17)  # GPIO 17 for record button
        quit_button_line = chip.get_line(27)    # GPIO 27 for quit button
        led_line = chip.get_line(22)            # GPIO 22 for LED
        record_button_line.request(consumer="record_button", type=gpiod.LINE_REQ_EV_FALLING_EDGE)
        quit_button_line.request(consumer="quit_button", type=gpiod.LINE_REQ_EV_FALLING_EDGE)
        led_line.request(consumer="led", type=gpiod.LINE_REQ_DIR_OUT)
        gpio_available = True
        logging.info("GPIO setup successful")
    except Exception as e:
        logging.error(f"GPIO setup failed: {e}")
        gpio_available = False

class StereoCamera:
    def __init__(self, resolution=(640, 480), framerate=30):
        self.camera1 = None
        self.camera2 = None
        self.resolution = resolution
        self.framerate = framerate
        self.init_cameras()

    def init_cameras(self):
        logging.info("Initializing camera 1...")
        try:
            self.camera1 = Picamera2(0)
            config1 = self.camera1.create_preview_configuration(main={"size": self.resolution, "format": "RGB888"})
            self.camera1.configure(config1)
            self.camera1.start()
            logging.info("Camera 1 initialized")
        except Exception as e:
            logging.error(f"Error initializing camera 1: {e}")
            raise

        time.sleep(2)  # Wait before initializing the second camera

        logging.info("Initializing camera 2...")
        try:
            self.camera2 = Picamera2(1)
            config2 = self.camera2.create_preview_configuration(main={"size": self.resolution, "format": "RGB888"})
            self.camera2.configure(config2)
            self.camera2.start()
            logging.info("Camera 2 initialized")
        except Exception as e:
            logging.error(f"Error initializing camera 2: {e}")
            raise

        time.sleep(2)  # Allow cameras to warm up
        logging.info("Both cameras initialized")

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
        logging.info("Cameras stopped")

def ffplay_display(pipe, width, height):
    cmd = ['ffplay',
           '-f', 'rawvideo',
           '-pixel_format', 'bgr24',
           '-video_size', f'{width}x{height}',
           '-i', 'pipe:0',
           '-window_title', 'Stereo Camera Feed']
    process = subprocess.Popen(cmd, stdin=pipe)
    return process

def toggle_recording():
    global is_recording, out
    if not is_recording:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        output_filename = os.path.join(recordings_dir, f'stereo_output_{timestamp}.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_filename, fourcc, framerate, (resolution[0] * 2, resolution[1]))
        is_recording = True
        if gpio_available:
            led_line.set_value(1)  # Turn on LED
        logging.info(f"Started recording: {output_filename}")
    else:
        is_recording = False
        if out is not None:
            out.release()
            out = None
        if gpio_available:
            led_line.set_value(0)  # Turn off LED
        logging.info("Stopped recording")

def keyboard_input():
    global is_recording
    while not stop_event.is_set():
        if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
            key = sys.stdin.read(1).lower()
            if key == 'r':
                toggle_recording()
            elif key == 'q':
                logging.info("Quit command received from keyboard")
                stop_event.set()
                break

def gpio_input():
    global record_button_line, quit_button_line
    while not stop_event.is_set():
        if record_button_line.event_wait(sec=1):
            event = record_button_line.event_read()
            if event.type == gpiod.LineEvent.FALLING_EDGE:
                toggle_recording()
        
        if quit_button_line.event_wait(sec=1):
            event = quit_button_line.event_read()
            if event.type == gpiod.LineEvent.FALLING_EDGE:
                logging.info("Quit button pressed")
                stop_event.set()
def main():
    global is_recording, out, recordings_dir, resolution, framerate, stop_event

    resolution = (640, 480)
    framerate = 30
    recordings_dir = 'recordings'
    os.makedirs(recordings_dir, exist_ok=True)

    setup_gpio()

    try:
        stereo_cam = StereoCamera(resolution=resolution, framerate=framerate)
    except Exception as e:
        logging.error(f"Failed to initialize stereo camera: {e}")
        return

    # Create a pipe for FFplay
    pipe_r, pipe_w = os.pipe()
    ffplay_process = ffplay_display(os.fdopen(pipe_r, 'rb'), width=resolution[0]*2, height=resolution[1])

    logging.info("Press 'r' or the record button to start/stop recording, 'q' or the quit button to quit")

    input_thread = threading.Thread(target=keyboard_input)
    input_thread.start()

    if gpio_available:
        gpio_thread = threading.Thread(target=gpio_input)
        gpio_thread.start()

    try:
        while not stop_event.is_set():
            frame1, frame2 = stereo_cam.capture_stereo()
            stitched_frame = np.hstack((frame1, frame2))

            if is_recording and out is not None:
                out.write(cv2.cvtColor(stitched_frame, cv2.COLOR_RGB2BGR))

            # Add "REC" text if recording
            if is_recording:
                cv2.putText(stitched_frame, "REC", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Write the frame to FFplay
            os.write(pipe_w, cv2.cvtColor(stitched_frame, cv2.COLOR_RGB2BGR).tobytes())

    except Exception as e:
        logging.error(f"Error during capture: {e}")
    finally:
        stop_event.set()
        input_thread.join()
        if gpio_available and 'gpio_thread' in locals():
            gpio_thread.join()
            record_button_line.release()
            quit_button_line.release()
            led_line.release()
        if out is not None:
            out.release()
        stereo_cam.close()
        ffplay_process.terminate()
        os.close(pipe_w)
        logging.info("Program ended")

if __name__ == "__main__":
    main()