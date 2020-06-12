import os

import requests
from tqdm import tqdm


class Namespace:
    """Simple class meant to make it possible to address dict entries with dot notation
    so dict[key] become dict.key"""
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

def add_element_wise(list1, list2):
    """Adds the two lists interleaved by eachother"""
    return [a + b for a, b in zip(list1, list2)]

# Source: https://gist.github.com/wy193777/0e2a4932e81afc6aa4c8f7a2984f34e2
def download_from_url(url, dst):
    """
    @param: url to download file
    @param: dst place to put the file
    """
    file_size = int(requests.head(url).headers["Content-Length"])
    partial_dst = dst + ".partial"
    if os.path.exists(partial_dst):
        first_byte = os.path.getsize(partial_dst)
    else:
        first_byte = 0
    if first_byte >= file_size:
        return file_size
    header = {"Range": "bytes=%s-%s" % (first_byte, file_size)}
    pbar = tqdm(
        total=file_size, initial=first_byte,
        unit='B', unit_scale=True, desc='Downloading %s' % url.split('/')[-1])
    req = requests.get(url, headers=header, stream=True)
    with(open(partial_dst, 'ab')) as f:
        for chunk in req.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
                pbar.update(1024)
    pbar.close()

    # First download to .partial file, then rename
    os.rename(partial_dst, dst)

    return file_size

def tag_to_readable(tag, conversion):
    """This functions turns a 9 character tag into a human readable morphological
    statement"""
    parts = []
    for index, char in enumerate(tag):
        if char == '-': continue
        parts.append(conversion[index][char] if char in conversion[index] else char)
    return " ".join(parts)

def readable_conversion_file(url):
    """Reads the provided url as a conversion file"""
    conversion_dict = []
    with open(url, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        assert len(lines) == 9, "The conversion file must have exactly 9 lines detailing the 9 conversion categories"

        for line in lines:
            category_dict = {}
            for part in line.split(","):
                key, value = [subpart.strip() for subpart in part.split(':')]
                category_dict[key] = value
            conversion_dict.append(category_dict)
    return conversion_dict
