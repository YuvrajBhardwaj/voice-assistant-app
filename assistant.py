import base64
from threading import Lock, Thread
import time
import numpy
import cv2
import requests
import tempfile
from PIL import ImageGrab
from cv2 import imencode
from dotenv import load_dotenv
import os
from speech_recognition import Microphone, Recognizer, UnknownValueError
import pygame
from queue import Queue

# Load environment variables
load_dotenv()

# Initialize pygame mixer once at startup
pygame.mixer.init()

# --- Desktop Screenshot Class ---
class DesktopScreenshot:
    def __init__(self):
        self.screenshot = None
        self.running = False
        self.lock = Lock()

    def start(self):
        if self.running:
            return self
        self.running = True
        self.thread = Thread(target=self.update)
        self.thread.start()
        return self

    def update(self):
        while self.running:
            screenshot = ImageGrab.grab()
            screenshot = cv2.cvtColor(numpy.array(screenshot), cv2.COLOR_RGB2BGR)
            with self.lock:
                self.screenshot = screenshot
            time.sleep(0.1)

    def read(self, encode=False):
        with self.lock:
            screenshot = self.screenshot.copy() if self.screenshot is not None else None
        if encode and screenshot is not None:
            _, buffer = imencode(".jpeg", screenshot)
            return base64.b64encode(buffer)
        return screenshot

    def stop(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join()

# --- Camera Capture Class ---
class CameraCapture:
    def __init__(self, device=0):
        self.cap = cv2.VideoCapture(device)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera device {device}")
        self.frame = None
        self.running = False
        self.lock = Lock()

    def start(self):
        if self.running:
            return self
        self.running = True
        self.thread = Thread(target=self.update)
        self.thread.start()
        return self

    def update(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = frame
            else:
                print("Failed to grab frame from camera.")
            time.sleep(0.01)

    def read(self, encode=False):
        with self.lock:
            frame = self.frame.copy() if self.frame is not None else None
        if encode and frame is not None:
            _, buffer = imencode(".jpeg", frame)
            return base64.b64encode(buffer)
        return frame

    def stop(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join()
        self.cap.release()

# --- ElevenLabs TTS Function ---
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_VOICE_ID = "nPczCjzI2devNBz1zQrb"

def elevenlabs_tts(text):
    if not ELEVENLABS_API_KEY:
        print("ELEVENLABS_API_KEY not set.")
        return

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}" 
    headers = {
        "xi-api-key": ELEVENLABS_API_KEY,
        "Content-Type": "application/json"
    }
    data = {
        "text": text,
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.75
        }
    }

    response = requests.post(url, json=data, headers=headers)
    if response.status_code == 200:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
            f.write(response.content)
            f.flush()
            filename = f.name

        try:
            pygame.mixer.music.load(filename)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
            pygame.mixer.music.unload()
            os.remove(filename)
        except Exception as e:
            print("Error playing audio:", e)
    else:
        print("TTS request failed:", response.status_code, response.text)

# --- Assistant Class ---
class Assistant:
    def __init__(self, tts_queue):
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.model = "meta-llama/llama-3-8b-instruct"
        self.api_url = "https://openrouter.ai/api/v1/chat/completions" 
        self.tts_queue = tts_queue

    def answer(self, prompt, image=None):
        if not prompt or not self.api_key:
            print("Missing prompt or API key.")
            return

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        }

        try:
            response = requests.post(self.api_url, headers=headers, json=payload)
            if response.status_code == 200:
                data = response.json()
                reply = data["choices"][0]["message"]["content"]
                print("Assistant:", reply)
                self.tts_queue.put(reply)
            else:
                print("OpenRouter API error:", response.status_code, response.text)
        except Exception as e:
            print("Exception during OpenRouter API call:", e)

# --- Initialize Components ---
tts_queue = Queue()
desktop_screenshot = DesktopScreenshot().start()
camera_capture = CameraCapture(device=0).start()
assistant = Assistant(tts_queue)

# --- Voice Recognition Setup ---
recognizer = Recognizer()
microphone = Microphone()

with microphone as source:
    recognizer.adjust_for_ambient_noise(source)

def audio_callback(recognizer, audio):
    try:
        prompt = recognizer.recognize_google(audio)
        print("Recognized:", prompt)
        assistant.answer(prompt)
    except UnknownValueError:
        print("Could not understand audio.")

stop_listening = recognizer.listen_in_background(microphone, audio_callback)

# --- Main Loop ---
try:
    print("Listening... Say something!")
    while True:
        desktop_img = desktop_screenshot.read()
        camera_img = camera_capture.read()

        if desktop_img is not None:
            cv2.imshow("Desktop", desktop_img)
        if camera_img is not None:
            cv2.imshow("Camera", camera_img)

        # Process TTS queue
        while not tts_queue.empty():
            text_to_speak = tts_queue.get()
            elevenlabs_tts(text_to_speak)

        if cv2.waitKey(1) in [27, ord("q")]:
            break
finally:
    desktop_screenshot.stop()
    camera_capture.stop()
    cv2.destroyAllWindows()
    stop_listening(wait_for_stop=False)