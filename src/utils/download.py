import re
import torch
import timm.models.hub as timm_hub
import os
from pathlib import Path


def download_cached_file(url, check_hash=False, progress=True):
    timm_hub.download_cached_file(url, check_hash, progress)
    parts = torch.hub.urlparse(url)
    filename = os.path.basename(parts.path)
    cached_file = os.path.join(timm_hub.get_cache_dir(), filename)

    return cached_file


def is_url(input_url):
    """
    Check if an input string is a url. look for http(s):// and ignoring the case
    """
    is_url = re.match(r"^(?:http)s?://", input_url, re.IGNORECASE) is not None
    return is_url


def load_checkpoint(url_or_path, device):
    if is_url(url_or_path):
        cached_file = download_cached_file(url_or_path)
        checkpoint = torch.load(cached_file, map_location=device)
    else:
        if not isinstance(url_or_path, Path):
            url_or_path = Path(url_or_path)

        if url_or_path.exists():
            checkpoint = torch.load(url_or_path.as_posix(), map_location=device)
        else:
            raise RuntimeError("checkpoint url or path is invalid")
    return checkpoint
