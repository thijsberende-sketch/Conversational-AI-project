"""
Records one game session to its own folder, for transcribing later:

    recordings/2026-11-24_14-05-12/
        audio.wav       stereo, 16-bit: left = robot microphone (the player), right = what Furhat says
        log.jsonl       one JSON object per line: every robot event and game event, with "t" in seconds
        transcript.txt  readable summary written at the end: [mm:ss.s] Furhat / Player lines

Times in transcript.txt are measured from the start of audio.wav, so they line up with the recording.
In log.jsonl, "t" is seconds since the session started; the "audio_start" entry gives the offset.
log.jsonl is flushed after every line and the WAV header is updated after every chunk, so both stay
usable even if the program crashes. transcript.txt is only written when the session closes normally.
"""

import json
import time
import wave
from datetime import datetime
from pathlib import Path

import numpy as np


class SessionRecorder:
    def __init__(self, base_dir: Path, sample_rate: int = 16000):
        self.folder = Path(base_dir) / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.folder.mkdir(parents=True, exist_ok=True)
        self.sample_rate = sample_rate

        self._start = time.monotonic()
        self._log = open(self.folder / "log.jsonl", "w", encoding="utf-8")
        self._wav: wave.Wave_write | None = None
        self._audio_t0: float | None = None   # session time at which audio.wav starts
        self._last_hear_start: float | None = None
        self._lines: list[tuple[float, str, str]] = []  # (t, speaker, text) for transcript.txt
        self._closed = False

        self.log("session_start", started=datetime.now().isoformat(timespec="seconds"))

    def _now(self) -> float:
        return time.monotonic() - self._start

    def log(self, event: str, **data) -> None:
        """Add one line to log.jsonl."""
        if self._closed:
            return
        entry = {"t": round(self._now(), 3), "event": event}
        entry.update({key: value for key, value in data.items() if key not in entry})
        self._log.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
        self._log.flush()

    # ----- called by the game -----

    def furhat_says(self, text: str) -> None:
        self._lines.append((self._now(), "Furhat", text))
        self.log("furhat_says", text=text)

    def player_says(self, text: str, counted: bool) -> None:
        """What the robot heard. counted=False: it was ignored (not a question)."""
        # The text arrives when the player has finished; use the moment they started speaking if known
        t = self._last_hear_start if self._last_hear_start is not None else self._now()
        self._last_hear_start = None  # don't reuse it for a later sentence
        self._lines.append((t, "Player" if counted else "Player (ignored)", text))
        self.log("player_turn" if counted else "ignored", text=text)

    # ----- called by FurhatRobot -----

    def robot_event(self, event: dict) -> None:
        """Every realtime event from the robot (except raw audio and camera frames)."""
        event_type = event.get("type", "unknown")
        if event_type == "response.hear.start":
            self._last_hear_start = self._now()
        self.log("robot", **{"robot_event": event_type, **{k: v for k, v in event.items() if k != "type"}})

    def write_audio(self, mic: bytes, speaker: bytes) -> None:
        """Append one chunk of 16-bit audio: mic to the left channel, Furhat's speaker to the right."""
        if self._closed:
            return
        if self._wav is None:
            self._wav = wave.open(str(self.folder / "audio.wav"), "wb")
            self._wav.setnchannels(2)
            self._wav.setsampwidth(2)
            self._wav.setframerate(self.sample_rate)
            self._audio_t0 = self._now()
            self.log("audio_start", note="audio.wav starts here; transcript.txt times are relative to this")

        left = np.frombuffer(mic[: len(mic) // 2 * 2], dtype="<i2")
        right = np.frombuffer(speaker[: len(speaker) // 2 * 2], dtype="<i2")
        # If one side is missing or shorter in this chunk, fill it with silence so the channels stay aligned
        frames = np.zeros((max(len(left), len(right)), 2), dtype="<i2")
        frames[: len(left), 0] = left
        frames[: len(right), 1] = right
        self._wav.writeframes(frames.tobytes())

    # ----- end of session -----

    def close(self) -> None:
        if self._closed:
            return
        self.log("session_end")
        self._closed = True
        self._log.close()
        if self._wav is not None:
            self._wav.close()
        self._write_transcript()

    def _write_transcript(self) -> None:
        offset = self._audio_t0 or 0.0
        with open(self.folder / "transcript.txt", "w", encoding="utf-8") as file:
            file.write("Times are relative to the start of audio.wav.\n" if self._audio_t0 is not None
                       else "No audio was recorded; times are from the start of the session.\n")
            file.write("'Player (ignored)' = heard, but not treated as a question.\n\n")
            # Sort by time only (stable), so lines with the same time keep the order they happened in
            for t, speaker, text in sorted(self._lines, key=lambda line: line[0]):
                seconds = max(t - offset, 0.0)
                file.write(f"[{int(seconds // 60):02d}:{seconds % 60:04.1f}] {speaker}: {text}\n")
