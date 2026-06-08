"""Classical, GPU-free OpenCV document-cleanup pipeline.

Stage order (PRD §10): ingest -> nonphoto -> gate -> boundary -> perspective ->
illumination -> denoise -> output -> sanity -> pdf. Orchestrated by runner.py.
"""
