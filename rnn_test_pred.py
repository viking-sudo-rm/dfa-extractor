import torch
import pickle
import os
import argparse

from utils import Tokenizer
from models import Tagger

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("sent", type=str)
    parser.add_argument("--epoch", type=int, default=12)
    parser.add_argument("--lang", default="Tom7")
    return parser.parse_args()

args = parse_args()

tfilename = os.path.join("models", args.lang, "tokenizer.pkl")
with open(tfilename, "rb") as fh:
    tokenizer = pickle.load(fh)
model = Tagger(tokenizer.n_tokens, 10, 100)
filename = f"./models/{args.lang}/epoch{args.epoch}.th"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.load_state_dict(torch.load(filename, map_location=device))

token_ids = torch.tensor(tokenizer.tokenize(args.sent, add=False)).unsqueeze(dim=0)
print(token_ids)
labels = torch.zeros_like(token_ids)
mask = torch.zeros_like(token_ids)
results = model(token_ids, labels, mask)
predictions = results["predictions"].squeeze().tolist()

print("Tokenizer:", tokenizer.token_to_index)
print("Sentence:", args.sent)
print("Predictions:", predictions)
breakpoint()