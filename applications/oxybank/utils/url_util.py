from urllib.parse import urlparse, urlunparse


def ensure_url_protocol(url):
    parsed_url = urlparse(url)
    if not parsed_url.scheme:
        # Default to 'http' protocol
        parsed_url = urlparse(f'http://{url}')
    return urlunparse(parsed_url)
