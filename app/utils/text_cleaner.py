import re

def clean_text(text):
    """
    Clean and normalize text by removing special characters, punctuation, and extra while space

    Args:
        text (str): text to be cleaned
    
    Returns:
        str: The cleaned and normalized text.
    """

    # Remove non-alphanumeric characters (except spaces, '?', '!', and '.')
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)

    text = re.sub(r'[^a-zA-Z0-9\s?.!]', '', text)

    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)

    # Strip leading and trailing spaces
    text = text.strip()

    return text
