from tools.profiler import tracy_to_perfetto


def test_module_importable():
    assert callable(tracy_to_perfetto.main)
