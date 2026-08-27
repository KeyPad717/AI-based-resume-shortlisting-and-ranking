import hashlib


def content_id(text: str, length: int = 16) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:length]
