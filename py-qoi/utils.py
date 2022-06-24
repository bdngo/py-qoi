"""
Utility functions used by both encoder and decoder.
"""

def px_hash(px: list[int]) -> int:
    """
    Calculates the hash of a pixel for QOI indexing.
    """
    px_r, px_g, px_b = px[:3]
    px_a = px[3] if len(px) == 4 else 0
    return (px_r * 3 + px_g * 5 + px_b * 7 + px_a * 11) % 64
