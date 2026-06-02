import os
from dotenv import load_dotenv

load_dotenv()

SERVICES = {
    "grit": {
        "url": "https://llm.grit.ucsb.edu/",
        "key": os.getenv("GRIT_KEY")
    },
    "dream-lab": {
        "url": "https://litellm.dreamlab.ucsb.edu/",
        "key": os.getenv("DL_KEY")
    }
}

