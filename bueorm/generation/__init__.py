"""
Bueorm Generation - Image Generation Core
Shared heads and utilities for text->Z->TBV image synthesis.
"""

from bueorm.generation.image_head import TextToLatentHead, LatentToImageDecoder

__all__ = ["TextToLatentHead", "LatentToImageDecoder"]
