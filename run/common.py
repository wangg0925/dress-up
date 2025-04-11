import re
import os
import requests
from io import BytesIO
from PIL import Image

def is_url(s):
    url_pattern = re.compile(
        r'^(?:http|ftp)s?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'  # domain...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  #...or ip
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    return re.match(url_pattern, s) is not None

def is_image(url):
    if not is_url(url):
        return False
    valid_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']
    file_extension = os.path.splitext(url)[1].lower()
    return file_extension in valid_extensions

# def is_image_post(url):
#     if not is_url(url):
#         return False
#     try:
#         response = requests.head(url)
#         content_type = response.headers.get('Content-Type')
#         if content_type and content_type.startswith('image/'):
#             return True
#         return False
#     except:
#         return False

def get_image(url, download=False):
    if not is_image(url):
        image = Image.open(url)
        return image.resize((768, 1024))

    try:
        response = requests.get(url)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content))
        if download:
            image.save("downloaded_image.jpg") 
        return image.resize((768, 1024))
    except Exception as e:
        raise RuntimeError(f"处理图片时出错: {e}")