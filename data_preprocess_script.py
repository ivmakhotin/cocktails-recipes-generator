import json
from collections import Counter
from itertools import chain
from collections import defaultdict

MIX_COCK_PATH = 'data/mixology_cocktails.json'
MIX_ING_PATH = 'data/mixology_ingredients.json'
COCKS_FLAVORS_PATH = 'data/cocktails_by_flavors.json'
INGS_SYN_PATH = 'data/ingredients_synonims.json'

with open(MIX_COCK_PATH) as json_file:
    mixology_cocktails = json.load(json_file)

with open(MIX_ING_PATH) as json_file:
    mixology_ingredients = json.load(json_file)


def get_flavors(ingredients):
    flavors = list()
    for ing in ingredients:
        if "flavor" in mixology_ingredients[ing["id"]]:
            flavors = flavors + mixology_ingredients[ing["id"]]["flavor"]["link_names"]
    return flavors

all_flavors_occ = chain.from_iterable([get_flavors(cock["ingredients"]) for cock in mixology_cocktails.values()])


flavors_dict = {
    'sweet': 'sweet',
    'herbal': 'herbal',
    'herbs': 'herbal',
    'juniper': 'herbal',
    'floral': 'herbal',
    'bitter': 'bitter',
    'orange': 'citrus',
    'lemon': 'citrus',
    'lime': 'citrus',
    'pineapple': 'citrus',
    'lime citrus': 'citrus',
    'citrus': 'citrus',
    'grapefruit': 'citrus',
    'apple': 'fruit',
    'apricot': 'fruit',
    'cherry': 'fruit',
    'peach': 'fruit',
    'peach pits': 'fruit',
    'passion fruit': 'fruit',
    'banana': 'fruit',
    'spice': 'spicy',
    'spicy': 'spicy',
    'spices': 'spicy',
    'allspice': 'spicy',
    'vanilla': 'coffee/choco/vanila',
    'almond': 'coffee/choco/vanila',
    'coffee': 'coffee/choco/vanila',
    'cinnamon': 'coffee/choco/vanila',
    'chocolate': 'coffee/choco/vanila',
    'mint': 'mint',
    'peppermint': 'mint',
    'spearmint': 'mint'
}

cocktails_by_flavors = defaultdict(set)

for cock in mixology_cocktails.values():
    ingredients = cock["ingredients"]
    for ing in ingredients:
        if "flavor" in mixology_ingredients[ing["id"]]:
            for flavor in mixology_ingredients[ing["id"]]["flavor"]["link_names"]:
                if flavor in flavors_dict:
                    cocktails_by_flavors[flavors_dict[flavor]].add(cock["id"])

cocktails_by_flavors = {k:list(v) for k,v in cocktails_by_flavors.items()}
with open(COCKS_FLAVORS_PATH, 'w') as fp:
    json.dump(cocktails_by_flavors, fp, indent=4)

sysnonims = defaultdict(dict)
for cocktail in mixology_cocktails.values():
    for ing in cocktail["ingredients"]:
        if not ing["text"]:
            continue
        if ing["text"] not in sysnonims[ing["id"]]:
            sysnonims[ing["id"]][ing["text"]] = 1
        else:
            sysnonims[ing["id"]][ing["text"]] += 1

with open(INGS_SYN_PATH, 'w') as fp:
    json.dump(sysnonims, fp, indent=4)
