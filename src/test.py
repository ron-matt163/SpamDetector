import re

def shorten_url(text):
    # Define a regex pattern to match URLs
    url_pattern = r'\b(?:https?://)\S+\b'

    # Find all URLs in the text
    urls = re.findall(url_pattern, text)

    if urls:
        for url in urls:
            shortened_url = '/'.join(url.split('/', 3)[:3])
            text = text.replace(url, shortened_url)
    return text

# Example usage
input_text = "Check out https://www.example.com/page1 for more info."
shortened_text = shorten_url(input_text)
print(shortened_text)