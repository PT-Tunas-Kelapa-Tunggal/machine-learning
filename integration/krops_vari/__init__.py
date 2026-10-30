"""Krops VARI backend: one drone photo in, a confidence-weighted Mg/N boost per zone and a spreading plan out.

The logic of integration.ipynb and weather_fertilizer.ipynb as importable functions. Entry points:
`krops_vari.pipeline.run_scan`, the command line (`python -m krops_vari.main`) and the HTTP API (`krops_vari.api`).
"""
__version__ = "0.1.0"
