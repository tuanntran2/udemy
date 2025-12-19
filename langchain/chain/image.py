from langchain_community.utilities.dalle_image_generator import DallEAPIWrapper

from dotenv import load_dotenv


# load environment variables
load_dotenv()

dalle = DallEAPIWrapper()

image_url = dalle.run(
    """Generate a high resolution image of a cute dog""")
print(image_url)
