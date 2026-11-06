"""Compatibility code for retired one-pass and Core-13 SFT protocols.

Current formal SFT uses ``cli/sft/train_axis_b_repeated_stage_sft.py``. These
modules remain importable only because the shared base trainer still exposes
the historical sampling modes.
"""
