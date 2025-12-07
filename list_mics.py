# Save this as: list_mics.py
import pyaudio

p = pyaudio.PyAudio()

print("\n--- Available Audio Input Devices ---")
for i in range(p.get_device_count()):
    info = p.get_device_info_by_index(i)
    if info['maxInputChannels'] > 0:
        print(f"Index {i}: {info['name']}")

p.terminate()