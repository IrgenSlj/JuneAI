"""Tests for the notification bus."""
from __future__ import annotations

from june_brain.notification import Notification, NotificationBus


def test_dispatch_to_log():
    from june_brain.notification import bus as default_bus
    bus = default_bus
    results = bus.dispatch(Notification(title="T", body="B", priority="low", source="test"))
    assert len(results) >= 1
    assert results[0][0] == "log"


def test_fresh_bus_has_log_channel():
    bus = NotificationBus()
    results = bus.dispatch(Notification(title="T", body="B", source="test"))
    assert len(results) == 1
    assert results[0][0] == "log"


def test_custom_channel():
    bus = NotificationBus()
    received = []

    def channel(n: Notification) -> bool:
        received.append(n)
        return True

    bus.register("custom", channel)
    bus.dispatch(Notification(title="T", body="B", source="test"))
    assert len(received) == 1
    assert received[0].title == "T"


def test_channel_hint():
    bus = NotificationBus()
    a_called = False
    b_called = False

    def channel_a(n: Notification) -> bool:
        nonlocal a_called
        a_called = True
        return True

    def channel_b(n: Notification) -> bool:
        nonlocal b_called
        b_called = True
        return True

    bus.register("a", channel_a)
    bus.register("b", channel_b)
    bus.dispatch(Notification(title="T", body="B", channel_hint="a", source="test"))
    assert a_called
    assert not b_called


def test_channel_failure_logged():
    bus = NotificationBus()

    def broken(n: Notification) -> bool:
        raise ValueError("boom")

    bus.register("broken", broken)
    results = bus.dispatch(Notification(title="T", body="B", source="test"))
    assert len(results) == 2  # log + broken
    assert results[0][1] is True  # log succeeds
    assert results[1][1] is False  # broken fails
