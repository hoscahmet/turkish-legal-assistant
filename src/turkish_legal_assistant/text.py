def tr_lower(text: str) -> str:
    """Lowercase Turkish text correctly: 'I' -> 'ı' and 'İ' -> 'i'."""
    # str.lower() maps 'I' to 'i' and 'İ' to 'i' + combining dot, both wrong for Turkish.
    return text.replace("I", "ı").replace("İ", "i").lower()
