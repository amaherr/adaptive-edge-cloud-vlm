import signal

from adaptive_vlm.shutdown import graceful_sigterm


def test_sigterm_sets_stop_flag_and_restores_handler():
    previous = signal.getsignal(signal.SIGTERM)
    with graceful_sigterm() as stop:
        signal.raise_signal(signal.SIGTERM)
        assert stop.is_set()
    assert signal.getsignal(signal.SIGTERM) == previous
