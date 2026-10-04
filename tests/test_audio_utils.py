import unittest
import numpy as np
from src.utils.audio_utils import (
    calculate_rms_energy,
    normalize_audio,
    is_speech_active,
    remove_dc_offset,
    calculate_zero_crossing_rate,
    detect_audio_clipping,
    apply_pre_emphasis,
    resample_linear,
)


class TestAudioUtils(unittest.TestCase):
    def test_calculate_rms_energy(self):
        silence = np.zeros(1000, dtype=np.int16)
        self.assertEqual(calculate_rms_energy(silence), 0.0)

        sine = (np.sin(np.linspace(0, 2 * np.pi, 1000)) * 10000).astype(np.int16)
        rms = calculate_rms_energy(sine)
        self.assertGreater(rms, 5000)

    def test_detect_audio_clipping(self):
        clean_audio = np.array([100, 500, -200, 1000], dtype=np.int16)
        is_clipped, ratio = detect_audio_clipping(clean_audio)
        self.assertFalse(is_clipped)
        self.assertEqual(ratio, 0.0)

        # Create clipped audio (more than 1% exceeding 32700)
        clipped_audio = np.array([32750] * 50 + [1000] * 50, dtype=np.int16)
        is_clipped, ratio = detect_audio_clipping(clipped_audio)
        self.assertTrue(is_clipped)
        self.assertEqual(ratio, 0.5)

    def test_apply_pre_emphasis(self):
        audio = np.array([1000, 2000, 3000, 4000], dtype=np.int16)
        emphasized = apply_pre_emphasis(audio, coeff=0.97)
        self.assertEqual(len(emphasized), len(audio))
        self.assertEqual(emphasized[0], 1000)
        # y[1] = 2000 - 0.97 * 1000 = 1030
        self.assertEqual(emphasized[1], 1030)

    def test_resample_linear(self):
        # 100 samples at 100Hz -> resample to 200Hz should give 200 samples
        audio = np.linspace(0, 10000, 100).astype(np.int16)
        resampled = resample_linear(audio, orig_sr=100, target_sr=200)
        self.assertEqual(len(resampled), 200)
        self.assertEqual(resampled[0], audio[0])
        self.assertEqual(resampled[-1], audio[-1])

    def test_remove_dc_offset(self):
        # Audio with +500 DC bias
        audio_biased = np.array([600, 400, 600, 400], dtype=np.int16)
        dc_free = remove_dc_offset(audio_biased)
        self.assertAlmostEqual(float(np.mean(dc_free)), 0.0, delta=1.0)


if __name__ == "__main__":
    unittest.main()
