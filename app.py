import openai
import os
import requests
import base64
import streamlit as st
from io import BytesIO
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed

@st.cache(show_spinner=False)
def load_gif():
    gif_path = "loading.gif"
    return st.image(gif_path, use_column_width=True)

css = """
<style>


    .body {
                
                font-family: Roboto, sans-serif;
    }
            
    .grid-container {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
        grid-gap: 10px;
        padding: 10px;
    }
    .grid-item {
        border: 1px solid #ccc;
        padding: 20px;
        background-color: #ffffff;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    .grid-text {
        font-size: 1.1em;
    }
    .grid-image {
        width: 100%;
        height: auto;
        border-radius: 10px;
    }
    img {
        width: 100%;
        height: auto;
        border-radius: 10px;
    }
     p {
        font-size: 22px;
    }
    @media (max-width: 480px) {
        .grid-container {
            grid-template-columns: 1fr;
        }
    }
</style>
"""

st.markdown(css, unsafe_allow_html=True)

st.write("<p style='font-size: 1.2em;'>Innovative educational tool that uses SDXL and GPT-3 to create personalized and engaging stories, fostering education and personal growth in children in a fun and accessible way.</p>", unsafe_allow_html=True)

openai.api_key = os.getenv("OPENAI_API_KEY")


def generate_images(prompts, seed_values):
    engine_id = "stable-diffusion-xl-beta-v2-2-2"
    api_host = os.getenv('API_HOST', 'https://api.stability.ai')
    api_key = os.getenv("STABILITY_API_KEY")

    def generate_image(prompt, seed_value):
        with ThreadPoolExecutor(max_workers=4) as executor:
            future = executor.submit(
                requests.post,
                f"{api_host}/v1/generation/{engine_id}/text-to-image",
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "Authorization": f"Bearer {api_key}"
                },
                json={
                    "text_prompts": [
                        {
                            "text": prompt
                        }
                    ],
                    "cfg_scale": 7,
                    "clip_guidance_preset": "NONE",
                    "style_preset": "digital-art",
                    "height": 512,
                    "width": 512,
                    "samples": 1,
                    "steps": 40,
                    "seed": seed_value,
                },
            )
            response = future.result()
            if response.status_code != 200:
                st.error("Respuesta distinta de 200: " + str(response.text))
            else:
                data = response.json()
                img_data = base64.b64decode(data["artifacts"][0]["base64"])
                img = Image.open(BytesIO(img_data))
                img = img.convert("RGBA")
                png_bytes = BytesIO()
                img.save(png_bytes, "PNG")
                png_bytes.seek(0)
                return png_bytes.getvalue()

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(generate_image, prompt, seed_value) for prompt, seed_value in zip(prompts, seed_values)]
        return [future.result() for future in futures]

nombre_personaje = st.text_input("Ingrese el nombre del personaje:")

animal_personaje = ["Dog", "Cat", "Rabbit", "Elephant", "Lion", "Tiger", "Monkey", "Bear", "Fox", "Giraffe", "Zebra", "Hippopotamus", "Squirrel", "Bird", "Kangaroo", "Koala", "Panda", "Penguin", "Wolf", "Snake", "Turtle", "Crocodile", "Rhinoceros"]
temas = ["fairy tale", "science fiction", "adventure", "mystery"]
valores_morales = ["friendship", "honesty", "responsibility", "respect"]

tema = st.radio("Please select the genre of the story.:", temas)
valor_moral = st.radio("Select the moral value of the story:", valores_morales)
animal_personaje = st.selectbox("Select the main character's animal:", animal_personaje)

parrafos = []

st.write("<p style='font-size: 1.2em;'>story creation can take 120 seconds.</p>", unsafe_allow_html=True)

if st.button("Generate storie"):
    generating = True
    loading_gif_path = "loading.gif"
    loading_placeholder = st.empty()
    loading_placeholder.image(loading_gif_path, use_column_width=True)

    entrada_historia = (
    f"Write a children's story that teaches the importance of {valor_moral} and has a moral lesson."
    f"The narrative style of the story belongs to the world of {tema}. You should include a main character named {nombre_personaje} who is a {animal_personaje}. "
    f"The story must be interesting and entertaining enough to capture everyone's attention."
    
    )

    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=entrada_historia,
        max_tokens=700,
        n=1,
        stop=None,
        temperature=0.7,
    )

    historia = response.choices[0].text.strip()
    parrafos_sin_filtrar = historia.split('\n')

    parrafos = [parrafo for parrafo in parrafos_sin_filtrar if parrafo.strip()]

    parrafos = parrafos[:7]

    generating = False 

def obtener_contexto(historia, parrafo_actual, ventana=2):
    indices_parrafo_actual = historia.index(parrafo_actual)
    inicio = max(0, indices_parrafo_actual - ventana)
    fin = min(len(historia), indices_parrafo_actual + ventana + 1)
    contexto = historia[inicio:fin]

    return ' '.join(contexto)

def generate_image_prompt(contexto, parrafo, animal_personaje):
    entrada_generar_prompt = (
        f"Remove the character's name from the following paragraph and describe a scene with the {animal_personaje}: \"{parrafo}\."
        f"taking into account the following context: '{contexto}', create a more detailed description to generate an image based on the scene without the character's name"
    )

    response = openai

    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=entrada_generar_prompt,
        max_tokens=150,
        n=1,
        stop=None,
        temperature=0.4,
    )

    prompt_result = response.choices[0].text.strip()

    return prompt_result

image_prompts = []

for i, parrafo in enumerate(parrafos):
    contexto = obtener_contexto(parrafos, parrafo)
    prompt = generate_image_prompt(contexto, parrafo, animal_personaje)
    entrada_imagen = f"ilustration \"{prompt}\" beautiful color, digital art"
    image_prompts.append(entrada_imagen)

seed_values = [0] * len(parrafos)  # This line is modified
grid_imagenes = []
images = generate_images(image_prompts, seed_values)
for i, parrafo in enumerate(parrafos):
    grid_imagenes.append((parrafo, images[i]))

st.write("<div class='grid-container'>", unsafe_allow_html=True)
for i, (texto, imagen) in enumerate(grid_imagenes):
    cols = st.columns(2)
    cols[0].write(
        f"<div class='grid-item'><p class='grid-text'>{texto}</p></div>", unsafe_allow_html=True)
    if imagen is not None:
        cols[1].write(
            f"<div class='grid-item'><img class='grid-image' src='data:image/png;base64,{base64.b64encode(imagen).decode('utf-8')}' /></div>", unsafe_allow_html=True)
    else:
        cols[1].write(
            "<div class='grid-item'><p class='grid-text'>Imagen no disponible.</p></div>", unsafe_allow_html=True)
        
    loading_placeholder.empty()

    
st.write("</div>", unsafe_allow_html=True)

