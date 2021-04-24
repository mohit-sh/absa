import os
import copy
import string
import json
import argparse

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from sklearn.metrics import accuracy_score

parser = argparse.ArgumentParser()

parser.add_argument("--data-dir", help="Directory with json files for train, dev, and test split.")
parser.add_argument("--dataset-name", help="Dataset Name. Used for logging.")
parser.add_argument("--log-dir", default="runs", help="Tensorboard logging dir.")
parser.add_argument("--glove-embed-file", help="Glove 6B embedding file with 300d vectors.")
parser.add_argument("--hidden-dim", type=int, default=300, help="Dimension of LSTM Hidden layers")
parser.add_argument("--batch-size", type=int, default=25, help="Batch Size")
parser.add_argument("--num-epochs", type=int, default=300)
parser.add_argument("--learning-rate", type=float, default=5e-4)
parser.add_argument("--l2-penalty", type=float, default=5e-3)

parser.add_argument("--val-steps", type=int, default=100)

args = parser.parse_args()
EMBEDDING_DIM = 300
PAD_TOKEN = "<pad>"
EOS_TOKEN = "<eos>"
BATCH_SIZE = 25
HIDDEN_SIZE = 300
NUM_EPOCHS = 500
LEARNING_RATE = 5e-4 # In the paper, this is initial learning rate, not sure what learning schedule is
L2_PENALTY = 1e-3
EPSILON = 0.01 # Initialize OOV embeddings with U(-\epsilon, \epsilon)

VAL_STEPS = 100

POLARITY_MAPPING = {
        "positive" : 2,
        "neutral" : 1,
        "negative" : 0
}

DEVICE = torch.device('cuda')


log_file_path = os.path.join(
        args.log_dir,
        f"dataset-{args.dataset_name}::lr-{args.learning_rate}::bs-{args.batch_size}::nepochs-{args.num_epochs}::hd-{args.hidden_dim}::l2pen-{args.l2_penalty}"
)
writer = SummaryWriter(log_file_path)

def tokenize(text):
    """
    Split text into constituent words, splitting on following characters:
    1. All whitespaces
    2. All punctuations
    """
    pass

def read_dataset(data_file):
    with open(data_file) as f:
        data = json.load(f)
        sentences, terms = zip(*[(v['sentence'], v['term']) for k, v in data.items()])
        sentences, terms = sentences, terms
    all_text = list(sentences + terms)

    return all_text

def build_vocab(text_seq):
    word_to_idx = {}
    all_words = set()

    for text in text_seq:
        words = text.split()
        all_words.update([ w.strip(string.punctuation).lower() for w in words])

    all_words_list = [PAD_TOKEN, EOS_TOKEN] + list(all_words)

    for idx, word in enumerate(all_words_list):
        word_to_idx[word] = idx

    return word_to_idx, list(all_words)

def build_embedding_matrix(word2idx, glove_embed_path):
    
    V = len(word2idx)
    D = EMBEDDING_DIM
    # [NOTE][DONE]: Might need to add embeddings for <pad> and <eos> token
    embedding_matrix = torch.ones((V, D), dtype=torch.float)
    low, high = -1 * EPSILON, 1 * EPSILON
    #embedding_matrix.uniform_(from=low, to=high)
    embedding_matrix[word2idx[PAD_TOKEN]] = 0
    embedding_matrix[word2idx[EOS_TOKEN]] = 0

    # Parse glove embeddings
    with open(glove_embed_path) as f:
        for line in f:
            line = line.split()
            word, embed = line[0], line[1:]

            if word in word2idx:
                embed = [float(v) for v in embed]
                embedding_matrix[word2idx[word]] = torch.FloatTensor(embed)

    return embedding_matrix


class AspectTermDataset(Dataset):

    def __init__(self, data_file):
        
        with open(data_file) as f:
            self.data = [v for k, v in json.load(f).items()]

    def __getitem__(self, index):

        return self.data[index]["sentence"].lower(), self.data[index]["term"].lower(), self.data[index]["polarity"]

    def __len__(self):
        
        return len(self.data)

class AspectTermCollator:
    def __init__(self, pad_token, eos_token, vocab, polarity2idx):
        self.pad_token = pad_token
        self.eos_token = eos_token
        self.vocab = vocab
        self.polarity2idx = polarity2idx

    def collate_fn(self, batch_data):

        text_batch = []
        term_batch = []
        polarity_batch = []
        for text, term, polarity in batch_data:

            text_tokens, term_tokens, polarity = text.split(), term.split(), self.polarity2idx[polarity]
            
            # [NOTE] This is a good lesson. You should have a single function which does this splitting so that every place that uses it has the latest logic
            #try:
            #    text_token_ids = list(map(lambda token: self.vocab[token.strip(string.punctuation).lower()], text_tokens))
            #    term_token_ids = list(map(lambda token: self.vocab[token.strip(string.punctuation).lower()], term_tokens))
            #except :
            #    print(f"TEXT: \t {text}")
            #    print(f"TERM: \t {term}")

            text_token_ids, term_token_ids = [], []

            try:
                for token in text_tokens:
                    text_token_ids.append(self.vocab[token.strip(string.punctuation)])
                
                for token in term_tokens:
                    term_token_ids.append(self.vocab[token.strip(string.punctuation)])
            except:
                print(f"TEXT: \t {text}")
                print(f"TERM: \t {term}")

            text_batch.append(text_token_ids)
            term_batch.append(term_token_ids)
            polarity_batch.append(polarity)
        
        # Apply padding
        text_token_ids_max_len = max([len(text_token_ids) for text_token_ids in text_batch])
        term_token_ids_max_len = max([len(term_token_ids) for term_token_ids in term_batch])
        
        processed_text_batch = [] 
        processed_term_batch = []
        for text_token_ids, term_token_ids in zip(text_batch, term_batch):
            
            text_to_pad = text_token_ids_max_len - len(text_token_ids)
            term_to_pad = term_token_ids_max_len - len(term_token_ids)
            padded_text_token_ids = text_token_ids + [self.vocab[self.eos_token]] +  [self.vocab[self.pad_token]] * text_to_pad 
            padded_term_token_ids = term_token_ids + [self.vocab[self.eos_token]] +  [self.vocab[self.pad_token]] * term_to_pad

            processed_text_batch.append(padded_text_token_ids)
            processed_term_batch.append(padded_term_token_ids)

        # Convert to tensors

        text_batch, term_batch, polarity_batch = torch.LongTensor(processed_text_batch), torch.LongTensor(processed_term_batch), torch.LongTensor(polarity_batch)

        return text_batch, term_batch, polarity_batch

class LSTMModel(nn.Module):
    def __init__(
            self,
            eos_index,
            embedding_matrix, 
            hidden_size,
            num_layers=1,
            num_classes=3 # Pos/Neu/Negative
            ):

        super().__init__()

        self.eos_index = eos_index
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.hidden_size = hidden_size
        self.input_size = embedding_matrix.shape[-1]
        self.num_layers = num_layers

        self.embedding_layer = nn.Embedding.from_pretrained(embedding_matrix, freeze=False)

        # Setup LSTM Network
        self.sentence_encoder = nn.LSTM(
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                batch_first=True)

        self.prediction_network = nn.Sequential(
                nn.Linear(
                    in_features=self.hidden_size,
                    out_features=self.num_classes
                ),
                nn.LogSoftmax(dim=-1)
            )

    def forward(self, text_token_ids):
        """
        x : T x B
        """
        #B, T = x.shape
        #H = self.hidden_size

        #h_0, c_0 = torch.ones((1, B, H))  for now, assume h_0, c_0 are zero

        text_token_embeddings = self.embedding_layer(text_token_ids)
        output, _ = self.sentence_encoder(text_token_embeddings)
         
        # Now, get the hidden_states for the last step for each batch element
        last_index = (text_token_ids == self.eos_index).nonzero(as_tuple=True)
        last_but_one_index = (last_index[0], last_index[1] - 1)

        sentence_embeddings = output[last_but_one_index]
        class_probs = self.prediction_network(sentence_embeddings)

        return class_probs

if __name__ == "__main__":
    
    train_file = os.path.join(args.data_dir, "train.json")
    dev_file = os.path.join(args.data_dir, "dev.json")
    test_file = os.path.join(args.data_dir, "test.json")

    train_texts = read_dataset(train_file)
    dev_texts = read_dataset(dev_file)
    test_texts = read_dataset(test_file)

    all_texts = train_texts + dev_texts + test_texts

    word2idx, idx2word = build_vocab(all_texts)

    embeddings = build_embedding_matrix(word2idx, args.glove_embed_file)
    
    train_dataset = AspectTermDataset(train_file)
    dev_dataset = AspectTermDataset(dev_file)
    test_dataset = AspectTermDataset(test_file)

    aspect_term_collator = AspectTermCollator(PAD_TOKEN, EOS_TOKEN, word2idx, POLARITY_MAPPING)

    train_dataloader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            collate_fn=aspect_term_collator.collate_fn,
            shuffle=True)

    dev_dataloader = DataLoader(
            dev_dataset,
            batch_size=args.batch_size,
            collate_fn=aspect_term_collator.collate_fn,
            shuffle=True)

    test_dataloader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            collate_fn=aspect_term_collator.collate_fn,
            shuffle=True)

    model = LSTMModel(
            eos_index = word2idx[EOS_TOKEN],
            embedding_matrix=embeddings,
            hidden_size=args.hidden_dim)
    
    #[TODO] See if a context manager can help here!! I don't want to manually move everything to the right device. Set DEVICE as the default device 
    model.to(DEVICE)

    loss_fn = nn.NLLLoss(reduction="mean")
    optimizer = optim.Adagrad(model.parameters(), lr=args.learning_rate, weight_decay=args.l2_penalty)

    step = 0
    model.train()

    best_val_loss = float('inf')
    for epoch in range(NUM_EPOCHS):
        for batch_idx, (text, _, labels) in enumerate(train_dataloader):

            step += 1    
            text, labels = text.to(DEVICE), labels.to(DEVICE)

            class_lprobs = model(text)

            loss = loss_fn(class_lprobs, labels)

            optimizer.zero_grad()
            loss.backward()
            writer.add_scalar("Loss/Train", loss.item(), step) 
            optimizer.step()

            if step % VAL_STEPS == 0:
                # Validate
                preds = []
                gts = []
                model.eval()
                val_loss = 0
                for text, _, labels in dev_dataloader:
                    B = labels.shape[0]
                    text, labels = text.to(DEVICE), labels.to(DEVICE)

                    gts += labels.tolist()
                    
                    with torch.no_grad():
                        class_lprobs = model(text)
                    val_loss += loss_fn(class_lprobs, labels) * B

                    preds += torch.argmax(class_lprobs, dim=-1).tolist()
                
                val_loss = val_loss.item() / len(dev_dataset)
                writer.add_scalar("Loss/Val", val_loss, step)

                if loss < best_val_loss:
                    best_val_loss = loss
                    best_model = copy.deepcopy(model)
                acc = accuracy_score(gts, preds)

                writer.add_scalar("Acc/Val", acc, step)
                model.train()
    # Test the best model
    test_preds = []
    test_gts = []
    best_model.eval()
    test_loss = 0
    for text, _, labels in test_dataloader:
        B = labels.shape[0]
        text, labels = text.to(DEVICE), labels.to(DEVICE)

        test_gts += labels.tolist()
        
        with torch.no_grad():
            class_lprobs = model(text)
        test_loss += loss_fn(class_lprobs, labels) * B

        test_preds += torch.argmax(class_lprobs, dim=-1).tolist()
    
    test_loss = test_loss.item() / len(test_dataset)
    writer.add_scalar("Loss/Test", test_loss, 0)

    if loss < best_val_loss:
        best_val_loss = loss
        best_model = copy.deepcopy(model)
    test_acc = accuracy_score(test_gts, test_preds)
    writer.add_scalar("Acc/Test", test_acc, 0)
    writer.close()
