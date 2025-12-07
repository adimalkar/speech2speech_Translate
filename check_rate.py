# Save as check_rate.py
import pyaudio

p = pyaudio.PyAudio()
# We are checking Device Index 4
device_index = 4 

try:
    info = p.get_device_info_by_index(device_index)
    print(f"Device Name: {info['name']}")
    print(f"Default Sample Rate: {int(info['defaultSampleRate'])} Hz")
except Exception as e:
    print(f"Error: {e}")

p.terminate()