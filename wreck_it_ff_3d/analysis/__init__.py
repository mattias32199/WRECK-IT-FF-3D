"""Looking inside a model while it runs (hooks.py) and switching parts of it off (knockouts.py).

Both are context managers that wrap `runner.model` for one call, so the runner
itself never changes. They currently target VGGT's block structure.
"""
