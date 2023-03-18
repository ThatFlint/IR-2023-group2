import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup

# Load the dataset
df = pd.read_csv("lib/ChildrenQueries.csv")
train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

train_data.head()
exit(0)

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
    train_data = QueryDataset(data["query"], data["target"], tokenizer, max_len)
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

        print(f"Train loss: {epoch_loss:.2f} | Train accuracy: {epoch_acc:.2f}")
