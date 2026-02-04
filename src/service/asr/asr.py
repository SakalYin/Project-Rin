import torch
import numpy as np
import pyaudio
from .silero_vad_iterator import FixedVADIterator
import time
import threading
import queue
import logging

log = logging.getLogger(__name__)


class ASRProcessor:
    def __init__(self, buffer_span=20, long_pause_thres=4, start_pad_s=1, end_pad_s=1, vad_threshold=0.65):
        self.sample_rate = 16000
        model, _ = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad')
        self.model = model
        self.vad = FixedVADIterator(self.model, threshold=vad_threshold, sampling_rate=self.sample_rate,
                                    min_silence_duration_ms=250, speech_pad_ms=100)
        self.listening = False
        self.speaking = False

        self.buffer_span = buffer_span
        self.audio_buffer = []
        self.silence_start_time = None
        self.chunk_size = 512
        self.long_pause_th = long_pause_thres
        self.start_pad_s = start_pad_s
        self.end_pad_s = end_pad_s

        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=pyaudio.paFloat32,
                                channels=1,
                                rate=self.sample_rate,
                                input=True,
                                frames_per_buffer=self.chunk_size)

        self.listening = True
        self.signal = False
        self.chunks_poses_in_buffer = []
        self.start_pose = 0

        self.speech_segments_queue = queue.Queue()
        self._stop_event = threading.Event()

    def process_audio_stream(self):
        while self.listening and not self._stop_event.is_set():
            try:
                audio_bytes = self.stream.read(self.chunk_size)
                audio_np = np.frombuffer(audio_bytes, dtype=np.float32)
                self.audio_buffer.extend(audio_np)

                if len(self.audio_buffer) > self.sample_rate * self.buffer_span:
                    self.process_and_reset()

                vad_result = self.vad(audio_np)

                if vad_result is None:
                    if not self.vad.triggered:
                        if self.silence_start_time is None:
                            self.silence_start_time = time.time()
                        else:
                            silence_duration = time.time() - self.silence_start_time
                            if silence_duration >= self.long_pause_th:
                                self.process_and_reset()
                    continue

                if 'start' in vad_result:
                    if not self.signal:
                        self.signal = True
                    log.debug('Speech detected')
                    self.speaking = True
                    self.start_pose = max(0, len(self.audio_buffer) - self.sample_rate * self.start_pad_s)
                    self.silence_start_time = None

                elif 'end' in vad_result:
                    log.debug('Speech ended')
                    self.speaking = False
                    self.chunks_poses_in_buffer.append([self.start_pose, len(self.audio_buffer)])

            except IOError as e:
                log.error("PyAudio IOError: %s", e)
                self._stop_event.set()
                break
            except Exception as e:
                log.error("Audio stream error: %s", e)
                self._stop_event.set()
                break

    def process_and_reset(self):
        speech_seg = self._extract_speech_seg()
        if len(speech_seg) > self.sample_rate:
            self._queue_speech_segment(speech_seg)
        self.vad.reset_states()
        self.audio_buffer = self.audio_buffer[-int(self.sample_rate * self.start_pad_s):]
        self.start_pose = max(0, len(self.audio_buffer) - self.sample_rate * self.start_pad_s)
        self.chunks_poses_in_buffer = []
        self.signal = False
        self.silence_start_time = None

    def _extract_speech_seg(self):
        end_sample = len(self.audio_buffer) - max(0, ((self.long_pause_th - self.end_pad_s) * self.sample_rate))
        segment = np.array(self.audio_buffer)[int(0):int(end_sample)]
        return self._trim_silence(segment)

    def _trim_silence(self, audio):
        if len(audio) < self.sample_rate * 0.1:
            return np.array([])

        if len(self.chunks_poses_in_buffer) <= 0:
            return np.array([])

        trimmed_audio = audio

        if len(trimmed_audio) > 0:
            noise_window = audio[:int(self.sample_rate * 0.05)]
            noise_energy = np.sqrt(np.mean(noise_window**2)) + 1e-10
            threshold = noise_energy * 0.5
            non_silent_mask = np.abs(trimmed_audio) > threshold

            if non_silent_mask.any():
                start_idx = np.argmax(non_silent_mask)
                end_idx = len(trimmed_audio) - np.argmax(non_silent_mask[::-1])
                trimmed_audio = trimmed_audio[start_idx:end_idx]

        return trimmed_audio if len(trimmed_audio) > 0 else np.array([])

    def _queue_speech_segment(self, audio_segment: np.ndarray):
        """Put raw float32 audio segment into the queue for transcription."""
        if len(audio_segment) == 0:
            return
        self.speech_segments_queue.put(audio_segment.astype(np.float32))
        log.info("Queued speech segment: %.2fs", len(audio_segment) / self.sample_rate)

    def stop(self):
        self._stop_event.set()
        self.listening = False
        if self.audio_buffer:
            self.process_and_reset()

        try:
            self.stream.stop_stream()
            self.stream.close()
            self.p.terminate()
            log.info("Audio stream stopped and closed")
        except Exception as e:
            log.warning("Error during pyaudio cleanup: %s", e)
        finally:
            self.stream = None
