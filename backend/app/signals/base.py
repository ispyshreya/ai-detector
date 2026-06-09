"""The interface every signal module implements.

To add a signal: subclass Signal, set `name` and `signal_class`, implement
`available()` and `async analyze()`. Register it in registry.py. The /scan
endpoint runs every available signal in parallel with a timeout and isolates
failures, so a single signal raising never breaks the response.
"""

from __future__ import annotations

import abc
import hashlib
from dataclasses import dataclass

from app.schemas import SignalClass, SignalResult


@dataclass
class ImageInput:
    """The uploaded image, passed to every signal."""

    data: bytes
    filename: str | None
    content_type: str | None

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.data).hexdigest()


class Signal(abc.ABC):
    name: str = "unnamed"
    signal_class: SignalClass = SignalClass.detector

    @abc.abstractmethod
    def available(self) -> bool:
        """True if this signal is configured and can run (e.g. key present)."""

    @abc.abstractmethod
    async def analyze(self, image: ImageInput) -> SignalResult:
        """Analyze the image and return a SignalResult.

        Implementations should NOT raise for expected failures (upstream 4xx/5xx,
        timeouts) — return a SignalResult with status=error and .error set. The
        runner converts uncaught exceptions to status=error as a backstop.
        """
