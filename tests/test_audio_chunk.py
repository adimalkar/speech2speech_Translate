import time

import numpy as np
import pytest

from src.audio.chunk import AudioBuffer, AudioChunk


class TestAudioBuffer:
    def test_invalid_max_size_raises(self):
        with pytest.raises(ValueError, match="positive integer"):
            AudioBuffer(max_size=0)
        with pytest.raises(ValueError, match="positive integer"):
            AudioBuffer(max_size=-4)

    def test_initial_empty_state(self):
        buf = AudioBuffer(max_size=5)
        assert len(buf) == 0
        assert buf.is_empty is True
        assert buf.full is False
        assert buf.write_idx == 0
        assert buf.get_chunk().tolist() == []
        assert buf.peek().tolist() == []
        assert buf.peek(3).tolist() == []

    def test_empty_add_is_noop(self):
        buf = AudioBuffer(max_size=4)
        assert buf.add(np.array([], dtype=np.float32)) == 0
        assert buf.is_empty is True
        assert len(buf) == 0
        assert buf.full is False

    def test_non_1d_input_raises(self):
        buf = AudioBuffer(max_size=4)
        with pytest.raises(ValueError, match="1D"):
            buf.add(np.zeros((2, 2), dtype=np.float32))
        with pytest.raises(ValueError, match="1D"):
            buf.add(np.array(1.0, dtype=np.float32))

    def test_partial_and_exact_fill_wrap_write_idx(self):
        buf = AudioBuffer(max_size=5)
        assert buf.add([1.0, 2.0, 3.0]) == 3
        assert len(buf) == 3
        assert buf.is_empty is False
        assert buf.full is False
        assert buf.write_idx == 3
        np.testing.assert_allclose(buf.get_chunk(), [1.0, 2.0, 3.0])

        assert buf.add(np.array([4.0, 5.0], dtype=np.float64)) == 2
        assert buf.full is True
        assert buf.write_idx == 0
        assert len(buf) == 5
        assert buf.get_chunk().dtype == np.float32
        np.testing.assert_allclose(buf.get_chunk(), [1.0, 2.0, 3.0, 4.0, 5.0])

        # Writing after exact fill overwrites oldest samples in chronological order
        buf.add([6.0, 7.0])
        assert buf.full is True
        assert buf.write_idx == 2
        np.testing.assert_allclose(buf.get_chunk(), [3.0, 4.0, 5.0, 6.0, 7.0])

    def test_wrap_around_preserves_chronological_order(self):
        buf = AudioBuffer(max_size=5)
        buf.add([1.0, 2.0, 3.0])
        buf.add([4.0, 5.0, 6.0, 7.0])
        assert buf.full is True
        assert buf.write_idx == 2
        np.testing.assert_allclose(buf.get_chunk(), [3.0, 4.0, 5.0, 6.0, 7.0])

        # Second wrap-around
        buf.add([8.0, 9.0, 10.0, 11.0])
        assert buf.full is True
        assert buf.write_idx == 1
        np.testing.assert_allclose(buf.get_chunk(), [7.0, 8.0, 9.0, 10.0, 11.0])

    def test_oversized_input_keeps_latest_samples(self):
        buf = AudioBuffer(max_size=4)
        buf.add([1.0, 2.0])

        # Adding more than max_size previously raised a broadcast ValueError
        added = buf.add([10.0, 11.0, 12.0, 13.0, 14.0, 15.0])
        assert added == 6
        assert buf.full is True
        assert buf.write_idx == 0
        assert len(buf) == 4
        np.testing.assert_allclose(buf.get_chunk(), [12.0, 13.0, 14.0, 15.0])

        # Exact max_size input when write_idx > 0
        buf.reset()
        buf.add([1.0])
        assert buf.add([2.0, 3.0, 4.0, 5.0]) == 4
        assert buf.full is True
        assert buf.write_idx == 0
        np.testing.assert_allclose(buf.get_chunk(), [2.0, 3.0, 4.0, 5.0])

    def test_peek_and_copy_isolation(self):
        buf = AudioBuffer(max_size=5)
        buf.add([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])  # [2, 3, 4, 5, 6]

        assert buf.peek(0).tolist() == []
        assert buf.peek(-2).tolist() == []
        np.testing.assert_allclose(buf.peek(2), [5.0, 6.0])
        np.testing.assert_allclose(buf.peek(10), [2.0, 3.0, 4.0, 5.0, 6.0])
        np.testing.assert_allclose(buf.peek(), [2.0, 3.0, 4.0, 5.0, 6.0])

        # Mutating returned slices must not alter buffer contents
        chunk = buf.get_chunk()
        chunk[:] = -99.0
        recent = buf.peek(2)
        recent[:] = -88.0
        np.testing.assert_allclose(buf.get_chunk(), [2.0, 3.0, 4.0, 5.0, 6.0])

    def test_reset_clears_buffer_in_place(self):
        buf = AudioBuffer(max_size=4)
        backing = buf.data
        buf.add([1.0, 2.0, 3.0, 4.0])
        buf.reset()

        assert buf.data is backing
        assert buf.is_empty is True
        assert buf.full is False
        assert buf.write_idx == 0
        assert len(buf) == 0
        assert buf.get_chunk().tolist() == []
        np.testing.assert_allclose(buf.data, [0.0, 0.0, 0.0, 0.0])


class TestAudioChunk:
    def test_get_latency_ms(self):
        chunk = AudioChunk(
            chunk_id=1,
            audio_data=np.zeros(160, dtype=np.float32),
            timestamp=time.time(),
            duration_ms=10.0,
            sample_rate=16000,
            pipeline_start_time=time.time() - 0.05,
        )
        latency = chunk.get_latency_ms()
        assert latency >= 40.0
