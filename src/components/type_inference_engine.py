from google import genai
from google.genai import types

# The client gets the API key from the environment variable `GEMINI_API_KEY`.
client = genai.Client()

class TypeInferenceEngine:
    def __init__(self, client, model="gemini-3-flash-preview", thinking_level="low"):
        self.client = client
        self.model = model
        self.thinking_level = thinking_level

    def run(self, input_text, pretyped=False, optional_context=None):

        # Differences from WordNet Noun Hierarchy:
        # Omit noun.relation as this is not relevant for typing entities
        # Enriched group with animate, organization, object. 
        # Enriched object with additional instrument category for objects that can do things (e.g., planes, robots, etc.)
        # Split Quantity and Numeral based on Jenn paper 

        if pretyped:
            input_description = (
                "You will receive a list of sentences that have been annotated with the types of their main entities and the prepositions relating them.\n",
                "However, some of the sentences may contain errors in their type annotations that make them implausible or inconsistent with real-world knowledge.\n",
                "Your task is to identify and correct these errors, based on the context provided below. Please make the minimal number of revisions necessary.\n"
                "Again, please only type entities and prepositions, and use the correct type system for each one.\n",
            )

        else:
            input_description = "Given a list of simple, declarative sentences, infer the types of the main entities and the prepositions relating them.\n"

        if optional_context:
            context = f"Here is some context to help you revise the annotations: {optional_context}\n"
        else:
            context = "Please type each sentence based on the most common, literal interpretation of the words corresponding to entities and prepositions, without making any inferences or assumptions about how these entities and prepositions interact.\n"
        
        prompt = (
            input_description,
            context,
            "For entities, use the following type system (based on the WordNet hierarchy):\n",
            "human (e.g., person, child, teacher, Barack Obama)\n",
            "plant (e.g., tree, flower, grass)\n",
            "action (e.g., swimming, teaching)\n",
            "feeling (e.g., joy, anger)\n",
            "location (e.g., area, country, venue)\n",
            "object (e.g., book, table, bag)\n",
            "instrument (e.g., robot, conveyor belt, plane)\n",
            "body part (e.g., hand, leg, eye)\n",
            "possession (e.g., property, assets)\n",
            "animate group (e.g., family, team, crowd)\n",
            "organization (e.g., company, school, government)\n",
            "object group (e.g., collection, books, chairs)\n",
            "goal (e.g., purpose, objective)\n",
            "event (e.g., party, meeting, concert)\n",
            "animal (e.g., dog, cat, lion)\n",
            "food (e.g., pizza, apple, water)\n",
            "natural process (e.g., weather, growth, life)\n",
            "cognitive (e.g., belief, intention, dream)\n",
            "time (e.g., day, year, moment)\n",
            "natural object (e.g., rock, river, mountain)\n",
            "substance (e.g., water, air, metal)\n",
            "quantity (e.g., gallon, liter, mile)\n",
            "numeral (e.g., one, two, three)\n",
            "natural phenomenon (e.g., earthquake, thunder, rainbow)\n",
            "shape (e.g., circle, square, triangle)\n",
            "communication (e.g., logo, sentence, conversation, the symbol `1`)\n",
            "attribute (e.g., red, tall, beautiful)\n\n",
    
            "For prepositions, use the following type system (based on FrameNet).\n"
            "Only type the prepositions that explicitly indicate a relationship between entities, and fall into one of the following categories:\n",
            "spatial (e.g., in, on, at)\n",
            "directional (e.g., to, from, towards)\n",
            "temporal (e.g., before, after, during)\n",
            "comparative (e.g., than, like)\n",
            "causal (e.g., because of, due to)\n",
            "instrumental (e.g., with, by)\n",

            "Here are a few examples:\n",
            "Input sentences:\n",
            "The boy is playing soccer in the park.\n",
            "Typed Sentences:\n",
            "The boy (human) is playing soccer (action) in (spatial) the park (location).\n",
            "Input sentences:\n",
            "My dog is sleeping on his bed before sunset.\n",
            "My dog has a dream of getting treats.\n",
            "Typed Sentences:\n",
            "My dog (animal) is sleeping on (spatial) his bed (object) before (temporal) sunset (time).\n",
            "My dog (animal) has a dream (cognitive) of getting treats (possession).\n"
            "Input sentences:\n",
            "The teacher writes the equals sign on the board with chalk.\n",
            "The teacher computes the answer to 1 + 1.\n",
            "Typed Sentences:\n",
            "The teacher (human) writes the equals sign (communication) on (spatial) the board (object) with (instrumental) chalk (object).\n",
            "The teacher (human) computes the answer (cognitive) to 1 (numeral) + 1 (numeral).\n\n"
            f"Now, type the following text. Only generate the list of typed sentences without any additional commentary or explanation.\n",
            f"Input text: {input_text}\n",  
        )
            
        response = self.client.models.generate_content(
            model=self.model, contents=prompt,
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_level=self.thinking_level)
            ),
        )
        return response.text
