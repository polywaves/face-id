import os
from dotenv import load_dotenv
load_dotenv()


def get(key=None):
    if key:
        return os.getenv(key)
    else:
        return os.environ