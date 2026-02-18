from google import genai
from google.genai import types

# The client gets the API key from the environment variable `GEMINI_API_KEY`.
client = genai.Client()

class LanguageParser:
    def __init__(self, client, model="gemini-3-flash-preview", thinking_level="low"):
        self.client = client
        self.model = model
        self.thinking_level = thinking_level

    def run(self, input_text):

        prompt = (
            "Parse the following natural language text into short, declarative sentences that capture the main entities and their relationships.\n",
            "Here are a few examples:\n",
            "Input text: Alice, who is Bob's sister, loves to play chess with her friend Charlie.\n",
            "Parsed sentences:\n",
            "Alice is Bob's sisteer.\n",
            "Alice loves to play chess with Charlie.\n",
            "Alice is friends with Charlie.\n",
            "Input text: Ideas sleep furiously and a rock plays trumpet.\n",
            "Parsed sentences:\n",
            "Ideas sleep furiously.\n",
            "A rock plays trumpet.\n",
            "Input text: Before the sun rises, the cat jumps over the lazy dog and the bird sings a song.\n",
            "Parsed sentences:\n",
            "The cat jumps over the lazy dog before the sun rises.\n",
            "The bird sings a song before the sun rises.\n",
            "Now, parse the following text. Only generate the list of parsed sentences, split by newlines, without any additional commentary or explanation.\n",
            f"Input text: {input_text}\n",

        )
        response = self.client.models.generate_content(
            model=self.model, contents=prompt,
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_level=self.thinking_level)
            ),
        )
        return response.text
