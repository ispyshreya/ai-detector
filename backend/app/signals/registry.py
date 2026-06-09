"""Collects all signal instances. Agents append their module's instance here.

Kept import-light: each signal is constructed lazily so a module with a missing
optional dependency degrades to unavailable rather than crashing startup.
"""

from __future__ import annotations

from app.signals.base import Signal


def all_signals() -> list[Signal]:
    signals: list[Signal] = []

    # Each block is independent; a failed import disables only that signal.
    try:
        from app.signals.sightengine import SightengineSignal
        signals.append(SightengineSignal())
    except Exception:  # noqa: BLE001 - never let one signal break the registry
        pass

    try:
        from app.signals.exif import ExifSignal
        signals.append(ExifSignal())
    except Exception:  # noqa: BLE001
        pass

    try:
        from app.signals.ela import ElaSignal
        signals.append(ElaSignal())
    except Exception:  # noqa: BLE001
        pass

    try:
        from app.signals.c2pa import C2paSignal
        signals.append(C2paSignal())
    except Exception:  # noqa: BLE001
        pass

    try:
        from app.signals.reverse_search import ReverseSearchSignal
        signals.append(ReverseSearchSignal())
    except Exception:  # noqa: BLE001
        pass

    return signals


def available_signals() -> list[Signal]:
    return [s for s in all_signals() if s.available()]
