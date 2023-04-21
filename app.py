import openai
import os
import requests
import base64
import streamlit as st
from io import BytesIO
from PIL import Image


@st.cache(show_spinner=False)
def load_gif():
    # Asegúrate de que el archivo GIF esté en el mismo directorio que tu script de Streamlit
    gif_path = "loading.gif"
    return st.image(gif_path, use_column_width=True)



css = """
<style>
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
        border-radius: 10px;  /* Agrega bordes redondeados a las imágenes */
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

openai.api_key = os.getenv("OPENAI_API_KEY")


def generate_images(prompt, seed_value):
    if not prompt.strip():
        st.error(
            "La descripción de la escena está vacía. No se puede generar una imagen.")
        return None

    # Limit the prompt length to avoid exceeding the API's token limit
    prompt = prompt[:1000]

    engine_id = "stable-diffusion-xl-beta-v2-2-2"
    api_host = os.getenv('API_HOST', 'https://api.stability.ai')
    api_key = os.getenv("STABILITY_API_KEY")

    response = requests.post(
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
            "steps": 70,
            "seed": seed_value,
        },
    )

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


nombre_personaje = st.text_input("Enter character name:")
animal_personaje = st.selectbox(
    "Select the main character's animal:",
    [
        "Dog", "Cat", "Rabbit", "Elephant", "Lion", "Tiger", "Monkey",
        "Bear", "Fox", "Giraffe", "Zebra", "Hippopotamus", "Squirrel",
        "Bird", "Kangaroo", "Koala", "Panda", "Penguin", "Wolf",
        "Snake", "Turtle", "Crocodile", "Rhinoceros"
    ]
)

themes = ["fairy tale", "science fiction", "adventure", "mystery"]
moral_values = ["friendship", "honesty", "responsibility", "respect"]

# Create selection boxes for theme and moral value
theme = st.selectbox("Select the theme of the story:", themes)
moral_value = st.selectbox("Select the moral value of the story:", moral_values)


parrafos = []  # Initialize the 'parrafos' variable before the if block



# Generar historia con el tema seleccionado
if st.button("Generar historia"):

    # Muestra el GIF de carga
    loading_gif_path = "loading.gif"
    loading_placeholder = st.empty()
    loading_placeholder.image(loading_gif_path, use_column_width=True)


    entrada_historia = (
    f"Write a children's story that teaches the importance of {moral_value} and has a moral lesson. "
    f"The narrative style of the story belongs to the world of {theme}. You should include a main character named {nombre_personaje} who is a {animal_personaje}. "
    f"The story must be interesting and entertaining enough to capture everyone's attention."
    )
  
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=entrada_historia,
        max_tokens=900,
        n=1,
        stop=None,
        temperature=0.7,
    )

    historia = response.choices[0].text.strip()
    parrafos_sin_filtrar = historia.split('\n')

    # Filtrar los párrafos vacíos antes de agregarlos a la lista 'parrafos'
    parrafos = [parrafo for parrafo in parrafos_sin_filtrar if parrafo.strip()]

    # Limit the number of paragraphs to 6
    parrafos = parrafos[:7]


def obtener_contexto(historia, parrafo_actual, ventana=2):
    indices_parrafo_actual = historia.index(parrafo_actual)
    inicio = max(0, indices_parrafo_actual - ventana)
    fin = min(len(historia), indices_parrafo_actual + ventana + 1)
    contexto = historia[inicio:fin]

    return ' '.join(contexto)

def generate_image_prompt_without_name(parrafo, animal_personaje):
    entrada_generar_prompt = f"Remove the character's name from the following paragraph and describe a scene with the {animal_personaje}: \"{parrafo}\""
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=entrada_generar_prompt,
        max_tokens=50,
        n=1,
        stop=None,
        temperature=0.4,
    )

    prompt_sin_nombre = response.choices[0].text.strip()

    return prompt_sin_nombre    


def generate_image_prompt(contexto, parrafo_actual, animal_personaje):
    entrada_generar_prompt = (
        f"Taking into account the following context: '{contexto}', create a more detailed description to generate an image based on the following paragraph: '{parrafo_actual}'. "
        f"The story is set in the world of {theme} and has a main character who is a {animal_personaje}. The story should teach the importance of {moral_value}."
    )

    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=entrada_generar_prompt,
        max_tokens=100,
        n=1,
        stop=None,
        temperature=0.3,
    )

    prompt = response.choices[0].text.strip()
    
    # Reemplazar el nombre del personaje con el animal seleccionado por el usuario
    prompt_sin_nombre = generate_image_prompt_without_name(prompt, animal_personaje)
    
    return prompt_sin_nombre


image_prompts = []

for i, parrafo in enumerate(parrafos):
    contexto = obtener_contexto(parrafos, parrafo)
    prompt = generate_image_prompt(contexto, parrafo, animal_personaje)
    entrada_imagen = f"ilustration \"{prompt}\" beautiful color, artistic style "
    image_prompts.append(entrada_imagen)




# Establece el límite máximo de caracteres que deseas para cada prompt de imagen
max_chars = 100


# Generate images based on the image_prompts
seed_values = [3511231505] + [0] * (len(parrafos) - 1)
grid_imagenes = []
for i, parrafo in enumerate(parrafos):
    # Utiliza la función generate_images con el valor de la semilla correspondiente
    imagen = generate_images(image_prompts[i], seed_values[i])
    grid_imagenes.append((parrafo, imagen))

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
        
        # Borra el GIF de carga
    loading_placeholder.empty()
    
st.write("</div>", unsafe_allow_html=True)

