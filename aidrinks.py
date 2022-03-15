import streamlit as st
import json
import logging
from itertools import chain
from random import choice
import requests, json
import logging
from streamlit.report_thread import get_report_ctx

NAMES_SERVICE_URL = "http://127.0.0.1:8081"
NAMES_TEMP = 0.7
NAMES_N_SAMPLES = 10

RECIPES_SERVICE_URL = "http://127.0.0.1:8082"
RECIPES_TEMP = 0.5
RECIPES_N_SAMPLES = 10



MIX_COCK_PATH = 'data/mixology_cocktails.json'
MIX_ING_PATH = 'data/mixology_ingredients.json'
COCKS_FLAVORS_PATH = 'data/cocktails_by_flavors.json'
INGS_SYN_PATH = 'data/cleaned_ingredients_synonims.json'

with open(MIX_COCK_PATH) as json_file:
    mixology_cocktails = json.load(json_file)

with open(MIX_ING_PATH) as json_file:
    mixology_ingredients = json.load(json_file)

with open(COCKS_FLAVORS_PATH) as json_file:
    cocktails_by_flavors = json.load(json_file)

with open(INGS_SYN_PATH) as json_file:
    ingredients_synonims = json.load(json_file)

def get_ingredients(flavors):
    assert any(flavors.values())
    feasible_cocktail = list(set(chain.from_iterable([cocktails_by_flavors[flavor] for flavor in flavors if flavors[flavor]])))
    seed_cocktail_id = choice(feasible_cocktail)
    ingredients = dict()
    for ingredient in mixology_cocktails[seed_cocktail_id]["ingredients"]:
        ing_id = ingredient["id"]
        vol = ingredient["ml"]
        sub_ids = []
        if "substitute" in mixology_ingredients[ing_id]:
            links = mixology_ingredients[ing_id]["substitute"]["links"]
            sub_ids = [link.split('/')[-1] for link in links]
            sub_ids = [id for id in sub_ids if id in ingredients_synonims]
        sub_ids.append(ing_id)
        ing_id = choice(sub_ids)
        synonims = list(ingredients_synonims[ing_id].keys())
        text = choice(synonims)
        ingredients[ing_id] = {"text": text, "ml": int(vol)}
    return ingredients


def gen_title(ingredients):
    data = dict()
    data["temp"] = NAMES_TEMP
    data["n"] = NAMES_N_SAMPLES
    data["ingredients"] = ingredients
    response_decoded_json = requests.post(NAMES_SERVICE_URL, data=json.dumps(data))
    response_json = response_decoded_json.json()
    name = choice(response_json["names"])
    return name.strip().capitalize()

def gen_recipe(ingredients):
    data = dict()
    data["temp"] = RECIPES_TEMP
    data["n"] = RECIPES_N_SAMPLES
    data["ingredients"] = ingredients
    response_decoded_json = requests.post(RECIPES_SERVICE_URL, data=json.dumps(data))
    response_json = response_decoded_json.json()
    recipe = choice(response_json["recipes"])
    return recipe.strip()

@st.cache(suppress_st_warning=True)
def get_logger(session_id):
        # Create a custom logger
        logger = logging.getLogger(session_id)

        # Create handlers
        c_handler = logging.StreamHandler()
        f_handler = logging.FileHandler('app.log')
        c_handler.setLevel(logging.INFO)
        f_handler.setLevel(logging.INFO)

        # Create formatters and add it to handlers
        format = '%(asctime)s\t%(levelname)s: %(message)s'
        c_format = logging.Formatter(format)
        f_format = logging.Formatter(format)
        c_handler.setFormatter(c_format)
        f_handler.setFormatter(f_format)

        logger.addHandler(f_handler)
        return logger

ctx = get_report_ctx()
session_id = ctx.session_id
logger = get_logger(session_id)


st.title('AI drinks recipe generator')
st.markdown('Enter some information about yourself and your preferences and our neural network will generate a unique cocktail recipe for you. You can try it out in one of our partner bars!')
st.markdown('<div style="text-align: center;font-weight: bold;color: #ffffff; background-color: #800080"> Pay for your first cocktail only if you like it! </div>', unsafe_allow_html=True)
st.subheader('Tell a little about yourself')
col = st.beta_columns(2)

sex = col[0].selectbox('Sex', ['Male', 'Female'])
age = col[1].slider('Age', min_value=18, max_value=80, step=1)

newdrinks = st.selectbox('Do you like trying new drinks?', ['Sometimes', 'Yes', 'No'])
howoften = st.select_slider('How often do you drink cocktails?', options=['Very rare', 'About once a month', 'Two or three times a month', 'Once a week', 'Often'])

col = st.beta_columns(2)
moodcolor = col[0].multiselect('What color(s) is your mood associated with?', ['Black', 'White', 'Red', 'Green', 'Yellow', 'Blue', 'Pink', 'Gray', 'Brown', 'Orange', 'Purple'], [])
dog = col[1].selectbox('Do you have dog?', ['No', 'Yes'])

col = st.beta_columns(2)
geners = col[0].multiselect('What genres in music do you prefer?', ['Pop', 'Electronic', 'Funk', 'Hip-hop', 'Jazz', 'Rock', 'Metal', 'Punk', 'Soul', 'Reggae'],[])
games = col[1].selectbox('Do you play computer games?', ['No', 'Yes'])

st.subheader('Choose flavor preferences')
flavors = dict()
n, m = 4, 2
grid = iter(chain.from_iterable([st.beta_columns(m) for _ in range(n)]))
for flavor in cocktails_by_flavors.keys():
    cell = next(grid)
    flavors[flavor] = cell.checkbox(flavor.capitalize())

button_cell = st.beta_columns(3)[1]
if button_cell.button('Generate recipe'):
    if not any(flavors.values()):
        st.write('Please indecate flavor preferences')
    else:
        selected_flavors = [flavor for flavor in flavors if flavors[flavor]]
        log_data = str(session_id), str(sex), str(age), str(newdrinks), str(howoften), str(moodcolor), str(geners), str(selected_flavors)
        log_template = "id:{0}|sex:{1}|age:{2}|newdrinks:{3}|howoften:{4}|moodcolors:{5}|geners:{6}|flavors:{7}"
        logger.info(log_template.format(*log_data))
        ingredients = get_ingredients(flavors)
        title = gen_title(ingredients)
        recipe = gen_recipe(ingredients)
        st.write('## {0}'.format(title))
        for ingredient in ingredients.values():
            st.write('- {0} - {1} ml'.format(ingredient["text"], ingredient["ml"]))
        for instruction in recipe.split('\n'):
            st.write("{0}".format(instruction))
        st.write()
        st.markdown('<div style="text-align: center;font-weight: bold;color: #ffffff; background-color: #800080"> You can try it out in one of our partner bars!</div>', unsafe_allow_html=True)
        st.markdown("""
<details>
  <summary style="font-weight: bold">Our partner bars in Moscow</summary>

    - Mirofbar (Мясницкая 32, стр.2)

    - коt шрёdiнгера (Б. Дмитровка, 32)

    - Black Hat Bar (Садовая-Каретная, 20, стр. 1)

    - Bumbule Bat ( Солянка, 1/2)

    - Noor Bar Electro (Тверская, 23/12с1)

    - Папа Вейдер (Большой Златоуст-ий пер, 3/5)
</details>
""", unsafe_allow_html=True)
