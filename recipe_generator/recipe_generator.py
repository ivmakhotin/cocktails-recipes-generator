import flask
from flask import jsonify
import torch, torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from collections import Counter, defaultdict

MIX_COCK_PATH = '../data/mixology_cocktails.json'
MIX_ING_PATH = '../data/mixology_ingredients.json'

MAX_LENGTH = 200
MODEL_PATH = "recipe_generator_model.torch"
NUM_TOKENS = 54
EMBEDDING_SIZE = 32
RNN_NUM_UNITS = 512

with open(MIX_COCK_PATH) as json_file:
    mixology_cocktails = json.load(json_file)

with open(MIX_ING_PATH) as json_file:
    mixology_ingredients = json.load(json_file)


class DoubleLSTMCell(nn.Module):
    """
    Two layer LSTMCell
    """
    def __init__(self, num_tokens, embedding_size=16, rnn_num_units=64):
        super(self.__class__,self).__init__()
        self.num_units = rnn_num_units
        self.emb = nn.Embedding(num_tokens, embedding_size)
        self.lstm1 = nn.LSTMCell(embedding_size, rnn_num_units)
        self.lstm2 = nn.LSTMCell(rnn_num_units, rnn_num_units)
        self.rnn_to_logits = nn.Linear(rnn_num_units, num_tokens)

    def forward(self, x, prev_state):
        (prev_h1, prev_c1, prev_h2, prev_c2) = prev_state
        (next_h1, next_c1) = self.lstm1(self.emb(x), (prev_h1, prev_c1))
        (next_h2, next_c2) = self.lstm2(next_h1, (prev_h2, prev_c2))
        logits = self.rnn_to_logits(next_h2)

        return (next_h1, next_c1, next_h2, next_c2), F.log_softmax(logits, -1)

    def initial_state(self, batch_size):
        """ LSTM has two state variables, cell and hid for each layer"""
        return torch.zeros(batch_size, self.num_units).cuda(), torch.zeros(batch_size, self.num_units).cuda(), torch.zeros(batch_size, self.num_units).cuda(), torch.zeros(batch_size, self.num_units).cuda()

def generate_sample(char_rnn, init_state, seed_phrase=' ', max_length=MAX_LENGTH, temperature=1.0):
    '''
    The function generates text given a phrase of length at least SEQ_LENGTH.
    :param seed_phrase: prefix characters. The RNN is asked to continue the phrase
    :param max_length: maximum output length, including seed_phrase
    :param temperature: coefficient for sampling.  higher temperature produces more chaotic outputs,
                        smaller temperature converges to the single most likely output
    '''

    x_sequence = [token_to_id[token] for token in seed_phrase]
    x_sequence = torch.tensor([x_sequence], dtype=torch.int64)
    hid_state = torch.tensor(init_state), torch.zeros(1, char_rnn.num_units), torch.tensor(init_state), torch.zeros(1, char_rnn.num_units)

    #feed the seed phrase, if any
    for i in range(len(seed_phrase) - 1):
        hid_state, _ = char_rnn(x_sequence[:, i], hid_state)

    #start generating
    for _ in range(max_length - len(seed_phrase)):
        hid_state, logp_next = char_rnn(x_sequence[:, -1], hid_state)
        p_next = F.softmax(logp_next / temperature, dim=-1).data.numpy()[0]

        # sample next token and push it back into x_sequence
        next_ix = np.random.choice(num_tokens,p=p_next)
        next_ix = torch.tensor([[next_ix]], dtype=torch.int64)
        x_sequence = torch.cat([x_sequence, next_ix], dim=1)

    return ''.join([tokens[ix] for ix in x_sequence.data.numpy()[0]])


model = DoubleLSTMCell(num_tokens=NUM_TOKENS, embedding_size=EMBEDDING_SIZE, rnn_num_units=RNN_NUM_UNITS)
device = "cpu"
model.to(device)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

synonims = defaultdict(dict)
for cocktail in mixology_cocktails.values():
    for ing in cocktail["ingredients"]:
        if ing["text"] not in synonims[ing["id"]]:
            synonims[ing["id"]][ing["text"]] = 1
        else:
            synonims[ing["id"]][ing["text"]] += 1

ings_cnt = defaultdict(int)
for cocktail in mixology_cocktails.values():
    for ing in cocktail["ingredients"]:
        most_freq_name = max(list(synonims[ing["id"]].items()), key = lambda x: x[1])
        ings_cnt[(most_freq_name[0], ing["id"])] += 1

n_most_common = 512
most_common = sorted(ings_cnt.items(), key = lambda x: -x[1])[-n_most_common:]
most_common_ids = [int(e[0][1]) for e in most_common]
most_common_ids.sort()
id_to_idx = dict()
for idx, id in enumerate(most_common_ids):
    id_to_idx[id] = idx

init_states = np.zeros((len(mixology_cocktails),n_most_common), dtype=np.float32)
lines = list()
cocktail_id_to_idx = dict()
for idx, (id, cocktail_data) in enumerate(mixology_cocktails.items()):
    lines.append(" " + cocktail_data["notes"].lower())
    cocktail_id_to_idx[id] = idx
    for ing in cocktail_data["ingredients"]:
        init_states[idx, id_to_idx[int(ing["id"])]] = np.float64(ing["ml"])

tokens = set(''.join(lines))
tokens = sorted(list(tokens))
token_to_id = dict(zip(tokens, range(len(tokens))))
num_tokens = len(tokens)

# cocktail_name = 'mary-pickford-3822'
# init_state = init_states[[cocktail_id_to_idx[cocktail_name]]]

def get_init_state(ingredients):
    init_state = np.zeros(RNN_NUM_UNITS, dtype=np.float32)
    for ing_id, data in ingredients.items():
        init_state[id_to_idx[int(ing_id)]] = np.float32(data["ml"])
    return init_state[np.newaxis, :]

app = flask.Flask(__name__)

@app.route("/", methods=['POST'])
def gen_recipes():
    json_data = flask.request.get_json(force=True)
    n = int(json_data["n"])
    temp = float(json_data["temp"])
    init_state = get_init_state(json_data["ingredients"])
    response = dict()
    response["recipes"] = list()
    for _ in range(n):
        response["recipes"].append(generate_sample(model, init_state, temperature = temp))
    return jsonify(response)

if __name__ == '__main__':
    from waitress import serve
    serve(app, host="0.0.0.0", port=8082)
    #app.run(host="0.0.0.0", port=8082)
