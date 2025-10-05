import requests
import json
import os
from dotenv import load_dotenv

_ = load_dotenv()


class Model:
    """Init any OpenRouter model"""

    def __init__(self, model_id="google/gemma-3n-e2b-it:free"):
        self.model_id = model_id
        self.api_key = os.getenv("OPENROUTER")

    def ask(self, prompt):
        """Query"""
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
            },
            data=json.dumps(
                {
                    "model": self.model_id,
                    "messages": [{"role": "user", "content": prompt}],
                }
            ),
        )

        try:
            content = response.json()["choices"][0]["message"]["content"]
            return content
        except Exception:
            print("Error:", response.status_code, response.text)
