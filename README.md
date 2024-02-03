## Free GenAI APIs You Can Use in 2024

Many small-scale companies are offering powerful APIs at no cost or providing a free trial that may extend up to a year based on your usage. We will look into some of those APIs and explore their benefit and usage.

## 1. Voyage AI

Voyage is a team of leading AI researchers and engineers, building embedding models for better retrieval and RAG.

* As good as OpenAI Embedding Models

* Price: Currently Free (Feb 2024)

* Documentation: [https://docs.voyageai.com/](https://docs.voyageai.com/)

* Get Started: [https://docs.voyageai.com/](https://docs.voyageai.com/)

Supported Embedding models, and more to come.

 <iframe src="https://medium.com/media/f8464a95617451325678308e64d14308" frameborder=0></iframe>

To install voyage library:
```python
# Use pip to insatll the 'voyageai' Python package to the latest version.
pip install voyageai
```

let’s use one of the embedding model voyage-2 and see its output:
```python
# Import the 'voyageai' module
import voyageai

# Create a 'Client' object from the 'voyageai' module and initialize it with your API key
vo = voyageai.Client(api_key="<your secret voyage api key>")

# user query
user_query = "when apple is releasing their new Iphone?"


# The 'model' parameter is set to "voyage-2", and the 'input_type' parameter is set to "document"
documents_embeddings = vo.embed(
    [user_query], model="voyage-2", input_type="document"
).embeddings

# printing the embedding
print(documents_embeddings)

########### OUTPUT ###########
[ 0.12, 0.412, 0.573, ... 0.861 ] # dimension is 1024
########### OUTPUT ###########
```

## 2. AnyScale AI

Anyscale, the company behind Ray, releases APIs for LLM developers to run and fine-tune open-source LLMs quickly, cost-efficiently, and at scale.

* Running/Fine-Tuning Powerful Open-Source LLM at a very low or no cost

* Price (no credit card): Free tier $10, where $0.15 per Million/tokens

* Documentation: [https://docs.endpoints.anyscale.com/](https://docs.endpoints.anyscale.com/)

* Get Started: [https://app.endpoints.anyscale.com/welcome](https://app.endpoints.anyscale.com/welcome)

Supported LLM and Embedding models

 <iframe src="https://medium.com/media/d063ecf567aa49f3bab642c0704e6d6e" frameborder=0></iframe>

Anyscale endpoints works with OpenAI library:
```python
# Use pip to insatll the 'openai' Python package to the latest version.
pip install openai
```

let’s use one of the Text generation LLM and see its output:
```python
# Import necessary modules
import openai

# Define the Anyscale endpoint token
ANYSCALE_ENDPOINT_TOKEN = "<your secret anyscale api key>"

# Create an OpenAI client with the Anyscale base URL and API key
oai_client = openai.OpenAI(
    base_url="https://api.endpoints.anyscale.com/v1",
    api_key=anyscale_key,
)

# Define the OpenAI model to be used for chat completions
model = "mistralai/Mistral-7B-Instruct-v0.1"

# Define a prompt for the chat completion
prompt = '''hello, how are you?
'''

# Use the AnyScale model for chat completions
# Send a user message using the defined prompt
response = oai_client.chat.completions.create(
    model=model,
    messages=[
        {"role": "user", "content": prompt}
    ],
)

# printing the response
print(response.choices[0].message.content)


########### OUTPUT ###########
Hello! I am just a computer program, so I dont have 
feelings or emotions like a human does ...
########### OUTPUT ###########
```

## 3. Gemini Multi Model

This one you may already know, but it’s worth mentioning, Google released their Gemini Multi-Model last year, and its free tier API usage is what makes it more interesting.

* Chat with text and images (Similar to GPT-4) and Embedding Models

* Price: Free Version (60 Query per minute)

* Documentation: [https://ai.google.dev/docs](https://ai.google.dev/docs)

* Get Started: [https://makersuite.google.com/app/apikey](https://makersuite.google.com/app/apikey)

Supported Models

 <iframe src="https://medium.com/media/b1f73ec8466b9931984f97394495355c" frameborder=0></iframe>

To install required libraries
```python
# Install necessary libraries
pip install google-generativeai grpcio grpcio-tools
```
To use text model gemini-pro  
```python
# importing google.generativeai as genai
import google.generativeai as genai

# setting the api key
genai.configure(api_key="<your secret gemini api key>")

# setting the text model
model = genai.GenerativeModel('gemini-pro')

# generating response
response = model.generate_content("What is the meaning of life?")

# printing the response
print(response.text)

########### OUTPUT ###########
he query of life purpose has perplexed people 
across centuries ... 
########### OUTPUT ###########
```

To use image model gemini-pro-vision 
```python
# importing google.generativeai as genai
import google.generativeai as genai

# setting the api key
genai.configure(api_key="<your secret gemini api key>")

# setting the text model
model = genai.GenerativeModel('gemini-pro-vision')

# loading Image
import PIL.Image
img = PIL.Image.open('cat_wearing_hat.jpg')

# chating with image
response =  model.generate_content([img, "Is there a cat in this image?"])

# printing the response
print(response.text)

########### OUTPUT ###########
Yes there is a cat in this image
########### OUTPUT ###########
```

## 4. Depth Anything AI

Image depth estimation is about figuring out how far away objects in an image are. It’s an important problem in computer vision because it help in tasks such as self-driving cars. A Hugging Face space from [Lihe Young](https://huggingface.co/LiheYoung) offers an API through which you can find image depth.

* find image depth in seconds without storing or loading the model

* Price: Free (Required HuggingFace Token)

* Get HuggingFace Token: [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

* Web Demo: [https://huggingface.co/spaces/LiheYoung/Depth-Anything](https://huggingface.co/spaces/LiheYoung/Depth-Anything)

Supported Models:

* [https://huggingface.co/spaces/LiheYoung/Depth-Anything/tree/main/checkpoints](https://huggingface.co/spaces/LiheYoung/Depth-Anything/tree/main/checkpoints)

To install required libraries
```python
# Install necessary libraries
pip install  gradio_client

Finding image depth using depth-anything model.

from gradio_client import Client

# Your Hugging Face API token
huggingface_token = "YOUR_HUGGINGFACE_TOKEN"

# Create a Client instance with the URL of the Hugging Face model deployment
client = Client("https://liheyoung-depth-anything.hf.space/--replicas/odat1/")

# Set the headers parameter with your Hugging Face API token
headers = {"Authorization": f"Bearer {huggingface_token}"}

# image link or path
my_image = "house.jpg"

# Use the Client to make a prediction, passing the headers parameter
result = client.predict(
    my_image,
    api_name="/on_submit",
    headers=headers  # Pass the headers with the Hugging Face API token
)

# loading the result
from IPython.display import Image
image_path = result[0][1]
Image(filename=image_path)
```

![Output of Depth estimation](https://cdn-images-1.medium.com/max/9104/1*jRaq7jiSFE1HivnUiIJxzQ.png)

## 5. Screenshot to HTML/CSS

You can create a webpage template using an API provided by [HuggingFace M4](https://huggingface.co/HuggingFaceM4).

* Just take a screenshot of webpage and pass it in API.

* Price: Free (Required HuggingFace Token)

* Get HuggingFace Token: [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

* Web Demo: [https://huggingface … screenshot2html](https://huggingface.co/spaces/HuggingFaceM4/screenshot2html)

To install required libraries

```python
# Install necessary libraries
pip install  gradio_client
```

Converting website screenshot to code using screenshot-to-code model.

```python
# Installing required library
from gradio_client import Client

# Your Hugging Face API token
huggingface_token = "YOUR_HUGGINGFACE_TOKEN"

# Create a Client instance with the URL of the Hugging Face model deployment
client = Client("https://huggingfacem4-screenshot2html.hf.space/--replicas/cpol9/")

# Set the headers parameter with your Hugging Face API token
headers = {"Authorization": f"Bearer {huggingface_token}"}

# website image link or path
my_image = "mywebpage_screenshot.jpg"

# Use the Client to generate code, passing the headers parameter
result = client.predict(
    my_image,
    api_name="/model_inference",
    headers=headers  # Pass the headers with the Hugging Face API token
)

# printing the output
printing(result)


########### OUTPUT ###########
<html>
<style>
body {
...
</body>
</html>
########### OUTPUT ###########
```
![generated output comparison with actual image](https://cdn-images-1.medium.com/max/10472/1*TandXjyDj9dQrnH3D58MQA.png)

## 6. Whisper (Audio to Text)

Convert audio to text using Whisper API.

* Just convert audio to text using API, without loading whisper model.

* Price: Free (Required HuggingFace Token)

* Get HuggingFace Token: [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

* Web Demo: [https://hugging … whisper](https://huggingface.co/spaces/openai/whisper)

To install required libraries
```python
# Install necessary libraries
pip install  gradio_client
```
Converting audio to text using Whisper model.
```python
# Installing required library
from gradio_client import Client

# Your Hugging Face API token
huggingface_token = "YOUR_HUGGINGFACE_TOKEN"

# Create a Client instance with the URL of the Hugging Face model deployment
client = Client("https://huggingfacem4-screenshot2html.hf.space/--replicas/cpol9/")

# Set the headers parameter with your Hugging Face API token
headers = {"Authorization": f"Bearer {huggingface_token}"}

# audio link or path
my_image = "myaudio.mp4"

# Use the Client to generate a response, passing the headers parameter
result = client.predict(
    my_audio,
    "transcribe", # str in 'Task' Radio component
    api_name="/predict"
    headers=headers  # Pass the headers with the Hugging Face API token
)

# printing the output
printing(result)

########### OUTPUT ###########
Hi, how are you?
########### OUTPUT ###########
```
## What’s Next

There are many more APIs you can explore through [Hugging Face Spaces](https://huggingface.co/spaces). Many SME companies provide powerful generative AI tools at a very low cost, such as OpenAI embeddings, which cost $0.00013 for 1K/Tokens. Make sure to check their licenses, as many free APIs in the free tier either limit per day requests or are for non-commercial use.
