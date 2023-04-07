import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
import matplotlib.pyplot as plt


# Define a custom dataset for BERT input
class QueryDataset(Dataset):
    def __init__(self, queries, targets, tokenizer, max_len):
        self.queries = queries
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, item):
        query = str(self.queries[item])
        target = self.targets[item]

        encoding = self.tokenizer.encode_plus(
            query,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )

        return {
            'query_text': query,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'targets': torch.tensor(target, dtype=torch.long)
        }


# Define the BERT fine-tuning function
def fine_tune_BERT(data, tokenizer, max_len, batch_size, epochs):
    # Load the dataset
    train_data = QueryDataset(data["query"], data["isChild"], tokenizer, max_len)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    # Load the pre-trained BERT model and add a classification layer
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

    # Define the optimizer and the learning rate scheduler
    optimizer = AdamW(model.parameters(), lr=5e-5, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=len(train_loader) * epochs
    )

    # Fine-tune the BERT model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    train_losses = []
    train_accs = []

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        epoch_loss = 0
        epoch_acc = 0

        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            targets = batch['targets'].to(device)

            optimizer.zero_grad()
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=targets
            )
            loss = outputs.loss
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            epoch_acc += torch.sum(preds == targets)

        epoch_loss /= len(train_loader)
        epoch_acc /= len(train_data)

        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)

        print(f"Train loss: {epoch_loss:.2f} | Train accuracy: {epoch_acc:.2f}")
    return model, (train_losses, train_accs)


def test_BERT(model, data, tokenizer, max_len, batch_size):
    # Evaluate the model on the test set
    test_data = QueryDataset(data["query"], data["isChild"], tokenizer, max_len)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    test_loss = 0
    test_acc = 0

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            targets = batch['targets'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=targets
            )
            loss = outputs.loss
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)

            test_loss += loss.item()
            test_acc += torch.sum(preds == targets)

    test_loss /= len(test_loader)
    test_acc /= len(test_data)

    print(f"Test loss: {test_loss:.2f} | Test accuracy: {test_acc:.2f}")

    return model, (test_loss, test_acc)


def plot_results(results, labels=[]):
    plt.plot(results, label=labels)
    plt.show()


if __name__ == '__main__':
    # TODO: tweak parameters batch_size and epochs
    # TODO: find max_len from longest query in train_data
    # Variables (can be tweaked)
    max_len = 81  # Int describing the maximum query length, probably used for efficiency
    batch_size = 5  # Int describing the amount of samples per batch (used by the Dataloader)
    epochs = 5  # Int describing the amount of training iterations for BERT model

    # Load the dataset
    df = pd.read_csv('ChildrenQueries.csv', delimiter='\t', header=None, names=(['query', 'isChild']))

    train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    model, train_results = fine_tune_BERT(data=train_data, tokenizer=tokenizer, max_len=max_len, batch_size=batch_size,
                                          epochs=epochs)

    plot_results(train_results, ["train_losses", "train_accuracies"])
    model, test_results = test_BERT(model=model, data=test_data, tokenizer=tokenizer, max_len=max_len,
                                    batch_size=batch_size)

    plot_results(train_results, ["test_losses", "test_accuracies"])
