"""Smoke test for mjlab package."""

import sys


def test_basic_functionality() -> None:
  """Test that AdaPT Tennis tasks register without import errors."""
  import mjlab.tasks  # noqa: F401
  from mjlab.tasks.registry import list_tasks

  ids = list_tasks()
  assert "Mjlab-AdaPT-Tennis-Flat-Unitree-G1" in ids
  assert "Mjlab-Tracking-Flat-Unitree-G1" in ids


if __name__ == "__main__":
  try:
    test_basic_functionality()
    print("✓ Smoke test passed!")
    sys.exit(0)
  except Exception as e:
    print(f"✗ Smoke test failed: {e}")
    sys.exit(1)
