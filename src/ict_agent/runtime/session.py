"""Interactive input queue helpers for preemptible sessions."""

from __future__ import annotations

from queue import Empty, Queue
import select
import sys
import threading

from ict_agent.context import ContextManager
from ict_agent.runtime.preemption import is_autonomous_turn, request_preempt


QUEUE_EVENT_INPUT = "input"
QUEUE_EVENT_EOF = "eof"
QUEUE_EVENT_INTERRUPT = "interrupt"


class InputReaderThread(threading.Thread):
    """Read stdin lines into a thread-safe queue."""

    def __init__(self, user_queue: Queue[tuple[str, str]], stop_event: threading.Event):
        super().__init__(daemon=True)
        self._queue = user_queue
        self._stop_event = stop_event

    def run(self) -> None:
        while not self._stop_event.is_set():
            try:
                ready, _, _ = select.select([sys.stdin], [], [], 0.2)
            except (ValueError, OSError):
                return
            if not ready:
                continue
            try:
                line = sys.stdin.readline()
            except KeyboardInterrupt:
                self._queue.put((QUEUE_EVENT_INTERRUPT, ""))
                return
            if line == "":
                self._queue.put((QUEUE_EVENT_EOF, ""))
                return
            text = line.strip()
            if not text:
                continue
            self._queue.put((QUEUE_EVENT_INPUT, text))
            if is_autonomous_turn():
                request_preempt()


def dequeue_user_input_blocking(user_queue: Queue[tuple[str, str]]) -> tuple[str, str]:
    while True:
        event, text = user_queue.get()
        if event != QUEUE_EVENT_INPUT:
            return event, text
        if text.strip():
            return event, text.strip()


def dequeue_user_input_nowait(user_queue: Queue[tuple[str, str]]) -> tuple[str, str] | None:
    while True:
        try:
            event, text = user_queue.get_nowait()
        except Empty:
            return None
        if event != QUEUE_EVENT_INPUT:
            return event, text
        if text.strip():
            return event, text.strip()


def to_pending_input_from_preempt_event(
    preempt_item: tuple[str, str] | None,
    ctx: ContextManager,
) -> str | None:
    if not preempt_item:
        return None
    event, text = preempt_item
    if event in (QUEUE_EVENT_EOF, QUEUE_EVENT_INTERRUPT):
        return "quit"
    if event == QUEUE_EVENT_INPUT:
        ctx.messages.append(
            {
                "role": "system",
                "content": (
                    "## Runtime Preemption\n"
                    "Autonomous execution was preempted by new user input. "
                    "Prioritize the latest user message."
                ),
            }
        )
        return text.strip() if text else None
    return None
