import hashlib

def str_to_md5(input_str: str) -> str:
    """Convert string to MD5 hash value"""
    # Create MD5 hash object
    hash_obj = hashlib.md5()
    # Update hash object, need to encode string to bytes
    hash_obj.update(input_str.encode('utf-8'))
    # Get hexadecimal representation of hash value
    return hash_obj.hexdigest()
