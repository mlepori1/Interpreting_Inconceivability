from google import genai
from google.genai import types

# The client gets the API key from the environment variable `GEMINI_API_KEY`.
client = genai.Client()

class TypeChecker:
    def __init__(self, client, model="gemini-3-flash-preview", thinking_level="low"):
        self.client = client
        self.model = model
        self.thinking_level = thinking_level

    def run(self, input_text):
        # Differences from WordNet Noun Hierarchy:
        # Omit noun.relation as this is not relevant for typing entities
        # Enriched group with animate, organization, object. 
        # Enriched object with additional instrument category for objects that can do things (e.g., planes, robots, etc.)
        # Split Quantity and Numeral based on Jenn paper 
        rewrite_prompt = (
            "Given a list of typed sentences, rewrite each sentence to replace each entity with its type.\n",

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
    
            "For prepositions, use the following type system (based on FrameNet):\n",
            "spatial (e.g., in, on, at)\n",
            "directional (e.g., to, from, towards)\n",
            "temporal (e.g., before, after, during)\n",
            "comparative (e.g., than, like)\n",
            "causal (e.g., because of, due to)\n",
            "instrumental (e.g., with, by)\n",

            "Here are a few examples:\n",
            "Input sentences:\n",
            "The boy (human) is playing soccer (action) in (spatial) the park (location).\n",
            "Output sentences:\n",
            "The human is playing an action in (spatial) the location.\n",

            "Input sentences:\n",
            "My dog (animal) is sleeping on (spatial) his bed (object) before (temporal) sunset (time).\n",
            "My dog (animal) has a dream (cognitive) of getting treats (possession).\n"
            "My dog (animal) is happy (feeling) because of (causal) the treat (possession).\n"
            "Output sentences:\n",
            "My animal is sleeping on (spatial) his object before (temporal) a time.\n"
            "My animal has a cognitive state of getting a possession.\n"
            "My animal is experiencing a feeling because of a possession.\n"

            "Input sentences:\n",
            "The teacher (human) writes the equals sign (communication) on (spatial) the board (object) with (instrumental) chalk (object).\n",
            "The teacher (human) computes the answer (cognitive) to 1 (numeral) + 1 (numeral).\n"
            "Output sentences:\n",
            "The human writes a communication on (spatial) the object with (instrumental) the object.\n"
            "The human computes a cognitive state to a numeral + a numeral.\n\n"


            "Now, rewrite the following text. Only generate the list of rewritten sentences, split by newlines, without any additional commentary or explanation.\n",
            f"Input text: {input_text}\n",  
        )
            
        rewrites = self.client.models.generate_content(
            model=self.model, contents=rewrite_prompt,
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_level=self.thinking_level)
            ),
        ).text

        # Perform type checking by verifying that the rewritten sentences are consistent.
        type_check_prompt = (
            "You will receive a list of sentences, which have been rewritten to replace their original entities with generalized types. "
            "Additionally, relationships between entities have been annotated with their type (e.g., spatial, temporal, causal).\n",

            "Your task is to check whether each sentence can plausibly describe a situation in the real world, based on your world knowledge and understanding of how different types of entities can interact.\n",
            "If you encounter a sentence that seems implausible or inconsistent with real-world knowledge, return the phrase `TypeError` along with the sentence that caused the error, and a brief description of the cause of the error.",
            "If all sentences seem plausible, return `TypeCheckPassed`.\n",

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
    
            "For prepositions, use the following type system (based on FrameNet):\n",
            "spatial (e.g., in, on, at)\n",
            "directional (e.g., to, from, towards)\n",
            "temporal (e.g., before, after, during)\n",
            "comparative (e.g., than, like)\n",
            "causal (e.g., because of, due to)\n",
            "instrumental (e.g., with, by)\n",     

            "Here are a few examples:\n",
            "Input sentences:\n",
            "The human is playing an action in (spatial) the location.\n",
            "Output:\n",
            "TypeCheckPassed\n",

            "Input sentences:\n",
            "The object is sleeping on (spatial) the body part before (temporal) the natural phenomenon.\n",
            "The object is below (spatial) the object.\n",
            "Output:\n",
            "TypeError\nThe object is sleeping on (spatial) the body part before (temporal) the natural phenomenon.\n"
            "Sleeping is an action that cannot be performed by an object.\n",

            "Input sentences:\n",
            "The human writes a communication on (spatial) the object with (instrumental) the attribute.\n",
            "Output:\n",
            "TypeError\nThe human writes a communication on (spatial) the object with (instrumental) the attribute.\n"
            "An attribute (e.g., red, tall, beautiful) cannot be used as an instrument to perform an action.\n\n"

            "Now, type check the following text. Format your response precisely like the examples, omitting `Output:\n`.\n",
            f"Input text: {rewrites}\n",  
            )

        type_check = self.client.models.generate_content(
                model=self.model, contents=type_check_prompt,
                config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_level=self.thinking_level)
            )
            ).text

        return type_check, rewrites