import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import re
from deap import base, creator, tools, algorithms
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
import random
from sklearn.metrics import roc_auc_score
import pickle
import time
import gc
import math


start_time = time.time()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Set random seeds for reproducibility
random_seed = 42
torch.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)  # If using GPU
np.random.seed(random_seed)
random.seed(random_seed)

batch_size_ = 128
pre_batch_size_ = 8000
initial_epoch = 100
self_training_epoch = 10
self_iteration = 60
# unknown_data_number = 60000
test_number = 'res-RNN-CNN-Incremental-Constraint-Corrected'
# Number of samples to select as pseudo-negatives based on the top 1% of lowest confidence predictions
top_percent = 0.01  # 1% of the total unlabeled data


# Custom Dataset Class
class ProteinDataset(Dataset):
    def __init__(self, sequences, features, labels, max_length=512):
        self.sequences = sequences
        self.features = features
        self.labels = labels
        self.max_length = max_length

        # Define a simple mapping for amino acids to unique integers
        self.amino_acid_map = {aa: idx + 1 for idx, aa in enumerate("ACDEFGHIKLMNPQRSTVWY")}  # 20 standard amino acids
        self.amino_acid_map["X"] = 0  # Use 0 for unknown or masked amino acids

    def tokenize_sequence(self, sequence):
        # Convert the sequence to a list of integers based on the mapping
        tokenized = [self.amino_acid_map.get(aa, 0) for aa in sequence]  # Default to 0 if amino acid is not in the map
        return tokenized

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        feature = self.features[idx]
        label = self.labels[idx]

        # Tokenize the sequence and pad/truncate it to max_length
        tokenized_sequence = self.tokenize_sequence(sequence)
        if len(tokenized_sequence) > self.max_length:
            tokenized_sequence = tokenized_sequence[:self.max_length]
        else:
            tokenized_sequence += [0] * (self.max_length - len(tokenized_sequence))  # Pad with 0

        input_ids = torch.tensor(tokenized_sequence, dtype=torch.long)
        return input_ids, torch.tensor(feature, dtype=torch.float), torch.tensor(label, dtype=torch.float)


# Custom Model with Dual Pathway
class ProteinClassificationModel(nn.Module):
    def __init__(self, cnn_feature_dim, rnn_hidden_dim=128, num_rnn_layers=2, num_labels=1, max_length=512):
        super(ProteinClassificationModel, self).__init__()

        # CNN expects input as (batch_size, in_channels, input_length)
        # num_features = 625, so treat this as the input length (not channels)
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2)  # Downsample by a factor of 2
        self.relu = nn.ReLU()

        # Calculate the size after CNN and pooling layers
        # After two pooling layers, the length will be reduced by 4
        self.cnn_flattened_size = (cnn_feature_dim // 4) * 128  # After two max-pooling layers

        # RNN pathway for sequence features
        self.embedding = nn.Embedding(21, rnn_hidden_dim)  # 21 amino acids including unknowns
        self.rnn = nn.LSTM(input_size=rnn_hidden_dim, hidden_size=rnn_hidden_dim, num_layers=num_rnn_layers,
                           batch_first=True)

        # Fully connected layers after feature concatenation (CNN + RNN)
        self.fc1 = nn.Linear(self.cnn_flattened_size + (rnn_hidden_dim * max_length), 256)
        self.fc2 = nn.Linear(256, num_labels)

        # Output activation
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, features):
        # CNN pathway for physicochemical features
        x_cnn = features.unsqueeze(1)  # Add a channel dimension (batch_size, 1, num_features)
        x_cnn = self.relu(self.conv1(x_cnn))
        x_cnn = self.pool(x_cnn)
        x_cnn = self.relu(self.conv2(x_cnn))
        x_cnn = self.pool(x_cnn)
        x_cnn = x_cnn.view(x_cnn.size(0), -1)  # Flatten the CNN output

        # RNN pathway for sequence features
        x_rnn = self.embedding(input_ids)
        rnn_out, _ = self.rnn(x_rnn)  # LSTM layers
        x_rnn = rnn_out.reshape(rnn_out.size(0), -1)  # Flatten the RNN output

        # Concatenate CNN and RNN features
        x_concat = torch.cat((x_cnn, x_rnn), dim=1)

        # Fully connected layers for classification
        x = self.relu(self.fc1(x_concat))
        logits = self.fc2(x)
        return logits


# Objective Function for DE (Silhouette Score for Feature Selection)
def objective_function(individual, features):
    selected_features = [int(i) for i in individual]
    if len(selected_features) == 0:
        return -float('inf'),  # Prevent empty feature selection
    reduced_features = features[:, selected_features]
    kmeans = KMeans(n_clusters=2, random_state=42).fit(reduced_features)
    score = silhouette_score(reduced_features, kmeans.labels_)
    return score,


# Apply DE to reduce the dimension of physicochemical features
# def reduce_features_with_de(features, pop_size=1, max_iter=1):
def reduce_features_with_de(features, pop_size=100, max_iter=50):
    num_features = features.shape[1]
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_bool", random.randint, 0, 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=num_features)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", objective_function, features=features)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)

    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)

    algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.2, ngen=max_iter, halloffame=hof, verbose=True)

    best_individual = hof[0]
    selected_features = [int(i) for i in best_individual]
    return selected_features


# Function to classify a list of protein sequences
def classify_proteins(sequences, features, model, _batch_size):
    dataset = ProteinDataset(sequences, features, [0] * len(sequences))  # Dummy labels
    dataloader = DataLoader(dataset, batch_size=_batch_size)
    predictions = []
    model.to(device)
    with torch.no_grad():
        for input_ids, features, _ in dataloader:
            input_ids, features = input_ids.to(device), features.to(device)
            logits = model(input_ids=input_ids, features=features)
            probs = torch.sigmoid(logits).squeeze().tolist()
            predictions.extend(probs)
    return predictions


def initial_training(model, train_data):
    print(f"Initialize Model")

    model.to(device)
    model.train()

    # Create initial dataset with positive samples and pseudo-labeled samples
    train_sequences, train_features, train_labels = train_data
    dataset = ProteinDataset(train_sequences, train_features, train_labels)
    dataloader = DataLoader(dataset, batch_size=batch_size_, shuffle=True)

    # Calculate class weights
    positive_count = train_labels.count(1)
    negative_count = train_labels.count(0)
    total_count = positive_count + negative_count

    # Class weights: inverse of the class frequency
    pos_weight = negative_count / total_count
    neg_weight = positive_count / total_count

    # Create tensor for pos_weight to use in BCEWithLogitsLoss
    pos_weight_tensor = torch.tensor([pos_weight], device=device)

    # Define the optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=2e-5)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

    # Training loop
    for epoch in range(initial_epoch):
        total_loss = 0
        for input_ids, features, labels in dataloader:
            input_ids, features, labels = input_ids.to(device), features.to(
                device), labels.to(device)

            optimizer.zero_grad()
            logits = model(input_ids=input_ids, features=features)
            loss = criterion(logits, labels.unsqueeze(1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # Clear GPU memory
            del input_ids, features, labels, logits, loss
            torch.cuda.empty_cache()
            gc.collect()

        avg_loss = total_loss / len(dataloader)
        print(f"  Epoch {epoch + 1}, Loss: {avg_loss:.4f}")

    return model


def compute_positive_proportion(predictions, threshold=0.5):
    """Compute the proportion of predictions that are classified as positive."""
    positive_count = (predictions >= threshold).sum().item()
    total_count = len(predictions)
    return positive_count / total_count


# Self-training framework with ensemble and pseudo-labeling (focused on pseudo-negatives)
def self_training(models, train_data, unlabeled_data, num_iterations=self_iteration):
    pseudo_label_distributions = []  # To store the distribution of pseudo-labels across iterations
    loss_history = []  # To store the loss values across iterations
    last_sequence = None
    last_features = None

    initial_penalty_lambda = 1.0  # Start with a strong penalty
    penalty_decay_rate = 0.95  # Decay the penalty each iteration
    expected_positive_proportion_range = (0.30, 0.40)  # Based on the 30-40% range
    penalty_scaling_factor = 20.0  # New scaling factor for the penalty term

    for iteration in range(num_iterations):
        print(f"Iteration {iteration + 1}/{num_iterations}")

        # Unpack positive data
        train_sequences, train_features, train_labels = train_data
        unlab_sequences, unlab_features = unlabeled_data

        # Initialize the positive dataset (no need to add pseudo-positives)
        all_sequences = train_sequences[:]
        all_features = np.array(train_features)
        all_labels = train_labels[:]

        # Generate predictions for the entire unlabeled dataset in batches
        all_predictions = []
        for model in models:
            model.eval()
            model.to(device)
            batch_predictions = []
            for i in range(0, len(unlab_sequences), pre_batch_size_):
                batch_sequences = unlab_sequences[i:i + pre_batch_size_]
                batch_features = unlab_features[i:i + pre_batch_size_]
                pseudo_labels = classify_proteins(batch_sequences, batch_features, model, pre_batch_size_)
                batch_predictions.extend(pseudo_labels)
            all_predictions.append(batch_predictions)

        # Adjust lambda over iterations
        penalty_lambda = initial_penalty_lambda * (penalty_decay_rate ** iteration)
        # Average the predictions from the ensemble
        mean_predictions = np.mean(all_predictions, axis=0)
        # Compute the predicted proportion of positives
        predicted_proportion = compute_positive_proportion(mean_predictions, threshold=0.5)
        # Calculate the regularization penalty based on the deviation from the expected range
        lower_bound, upper_bound = expected_positive_proportion_range
        if predicted_proportion < lower_bound:
            regularization_penalty = penalty_lambda * (lower_bound - predicted_proportion)
        elif predicted_proportion > upper_bound:
            regularization_penalty = penalty_lambda * (predicted_proportion - upper_bound)
        else:
            regularization_penalty = 0

        # Calculate the number of samples to select
        num_to_select = max(1, int(top_percent * len(unlab_sequences)))
        # Sort the predictions in ascending order and select the indices of the top 1% lowest probabilities
        neg_threshold_index = np.argsort(mean_predictions)[:num_to_select]  # Select the lowest top 1% predictions
        # Sort the predictions in descending order for pseudo-positives
        pos_threshold_index = np.argsort(mean_predictions)[-num_to_select:]  # Select the highest top 1% predictions

        # Collect pseudo-negatives
        pseudo_negatives = [unlab_sequences[idx] for idx in neg_threshold_index]
        pseudo_neg_features = [unlab_features[idx] for idx in neg_threshold_index]

        # Collect pseudo-positives
        pseudo_positives = [unlab_sequences[idx] for idx in pos_threshold_index]
        pseudo_pos_features = [unlab_features[idx] for idx in pos_threshold_index]

        # Combine the true positives with the newly generated pseudo-negatives
        all_sequences.extend(pseudo_negatives + pseudo_positives)
        all_features = np.vstack([all_features, np.array(pseudo_neg_features), np.array(pseudo_pos_features)])
        all_labels.extend([0] * len(pseudo_negatives) + [1] * len(pseudo_positives))  # Label pseudo-negatives as 0

        # Create the training dataset and dataloader
        dataset = ProteinDataset(all_sequences, all_features, all_labels)
        dataloader = DataLoader(dataset, batch_size=batch_size_, shuffle=True)

        # Calculate class weights
        positive_count = all_labels.count(1)
        negative_count = all_labels.count(0)
        total_count = positive_count + negative_count

        # Class weights: inverse of the class frequency
        pos_weight = negative_count / total_count
        neg_weight = positive_count / total_count

        # Create tensor for pos_weight to use in BCEWithLogitsLoss
        pos_weight_tensor = torch.tensor([pos_weight], device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

        # Save the distribution of pseudo-labels
        pseudo_label_distribution = {
            'iteration': iteration + 1,
            'num_samples': len(all_labels),
            'mean_prediction_score': np.mean(mean_predictions),
            'std_prediction_score': np.std(mean_predictions)
        }
        pseudo_label_distributions.append(pseudo_label_distribution)

        # Training loop for each model
        for i in range(len(models)):  # Iterate over models with index
            model = models[i]  # Access the model directly by index
            model.train()
            model.to(device)
            optimizer = optim.Adam(model.parameters(), lr=2e-5)
            for epoch in range(self_training_epoch):
                total_loss = 0
                for input_ids, features, labels in dataloader:
                    optimizer.zero_grad()
                    input_ids, features, labels = input_ids.to(device), features.to(device), labels.to(device)
                    logits = model(input_ids=input_ids, features=features)
                    loss = criterion(logits, labels.unsqueeze(1)) + penalty_scaling_factor * regularization_penalty
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()

                    del input_ids, features, labels, logits, loss
                    torch.cuda.empty_cache()
                    gc.collect()

                avg_loss = total_loss / len(dataloader)
                print(f"  Model {models.index(model) + 1}, Epoch {epoch + 1}, Loss: {avg_loss:.4f}")

                # Save loss history
                loss_history.append({
                    'iteration': iteration + 1,
                    'epoch': epoch + 1,
                    'model_index': i + 1,
                    'loss': avg_loss
                })
        # Store the last iteration's pseudo-negatives and pseudo-neg-features
        last_sequence = all_sequences
        last_features = all_features

    # Save pseudo-label distributions and loss history to CSV
    pd.DataFrame(pseudo_label_distributions).to_csv(f'pseudo_label_distributions_{test_number}.csv',
                                                    index=False)
    pd.DataFrame(loss_history).to_csv(f'loss_history_{test_number}.csv', index=False)

    # Save the last iteration's pseudo-negatives and features to CSV
    pd.DataFrame(last_sequence).to_csv(f'last_sequences_{test_number}.csv', index=False)
    pd.DataFrame(last_features).to_csv(f'last_features_{test_number}.csv', index=False)

    return models


# Prepare the dataset
positive_data_path = '../prepare-final-dataset/positive_data.csv'
negative_ID_path = './getNegative/data/meeted-uniprotID-thershold-1.1.csv'
unknown_data_path = '../prepare-final-dataset/unknown_data.csv'
positive_data = pd.read_csv(positive_data_path)
negative_ID = list(set(pd.read_csv(negative_ID_path)['Entry'].tolist()))
positive_data = positive_data.drop(columns=['UniprotID'])
unknown_data = pd.read_csv(unknown_data_path)
unknown_data = unknown_data.drop(columns=['UniprotID'])
# print(unknown_data)
negative_data = unknown_data[unknown_data['ID'].isin(negative_ID)]  # intersection
negative_data = negative_data.reset_index(drop=True)
# print(negative_data)
# 从 DataFrame 中删除 ID 列包含在 id_list 中的所有行
unknown_data = unknown_data[~unknown_data['ID'].isin(negative_ID)]
unknown_data = unknown_data.reset_index(drop=True)
# print(unknown_data)

positive_data['Sequence'] = positive_data['Sequence'].apply(lambda x: re.sub(r"[UZOB]", "X", x))
negative_data['Sequence'] = negative_data['Sequence'].apply(lambda x: re.sub(r"[UZOB]", "X", x))
unknown_data['Sequence'] = unknown_data['Sequence'].apply(lambda x: re.sub(r"[UZOB]", "X", x))
# positive_data = positive_data.iloc[:10, :]
# negative_data = negative_data.iloc[:10, :]
unknown_data = unknown_data.sample(n=len(unknown_data), random_state=random_seed)
# unknown_data = unknown_data.sample(n=unknown_data_number, random_state=random_seed)

# positive_sequences = positive_data['Sequence'].tolist()
# negative_sequences = negative_data['Sequence'].tolist()
# unlabeled_sequences = unknown_data['Sequence'].tolist()
All_data = pd.concat([positive_data, negative_data], ignore_index=True)
All_data = pd.concat([All_data, unknown_data], ignore_index=True)
# Drop columns with any NaN values and get the retained columns
retained_columns = All_data.dropna(axis=1, how='any').columns.tolist()
# If needed, you can also filter All_data to keep only the retained columns
All_data = All_data[retained_columns]


# Load the reduced feature indices
reduced_features_index_path = f'reduced_features_index_{test_number}.pkl'
with open(reduced_features_index_path, 'rb') as f:
    reduced_features_index = pickle.load(f)
# Convert the list to a boolean mask
reduced_features_index = np.array(reduced_features_index, dtype=bool)

# print(reduced_features_index)
total_train_data = All_data.iloc[:positive_data.shape[0]+negative_data.shape[0], :]
total_train_data['label'] = [1] * positive_data.shape[0] + [0] * negative_data.shape[0]
train_df, val_df = train_test_split(total_train_data, test_size=0.2, random_state=42)
# Split positive samples into training and validation sets
train_sequences, val_sequences = train_df['Sequence'].tolist(), val_df['Sequence'].tolist()
train_features, val_features = train_df.iloc[:, 4:-1].values[:, reduced_features_index], \
    val_df.iloc[:, 4:-1].values[:, reduced_features_index]
train_labels, val_labels = train_df['label'].tolist(), val_df['label'].tolist()


train_data = (train_sequences, train_features, train_labels)
unlabeled_data = (unknown_data['Sequence'].tolist(), \
                  All_data.iloc[len(positive_data)+len(negative_data):, 4:].values[:, reduced_features_index])

# Initialize the ensemble of models
num_models = 3
models = [ProteinClassificationModel(cnn_feature_dim=train_features.shape[1]) for _ in range(num_models)]
# Initialize the parameters of each model and update the list
for i in range(len(models)):
    models[i] = initial_training(models[i], train_data)

# Perform self-training with pseudo-labeling
fine_tuned_models = \
    self_training(models, train_data, unlabeled_data)

# Save the fine-tuned models
model_save_path = "./fine_tuned_prot_bert_bfd_localization"
for i in range(len(fine_tuned_models)):
    torch.save(fine_tuned_models[i].state_dict(), f"{model_save_path}/pytorch_model_{test_number}_{i}.bin")


# Load the fine-tuned model for inference
fine_tuned_models = [ProteinClassificationModel(cnn_feature_dim=train_features.shape[1]) for _ in range(num_models)]
for i in range(len(fine_tuned_models)):
    fine_tuned_models[i].load_state_dict(torch.load(f"{model_save_path}/pytorch_model_{test_number}_{i}.bin"))
    fine_tuned_models[i].eval()


# Evaluate the model
def evaluate_model(models, val_sequences, val_features, val_labels):
    val_predictions = [classify_proteins(val_sequences, val_features, model, len(val_sequences)) for model in models]
    val_predictions = np.mean(val_predictions, axis=0)
    # Convert validation labels to numpy array for calculation consistency
    val_labels = np.array(val_labels)
    # Calculate accuracy
    val_predictions_binary = (val_predictions >= 0.5).astype(int)
    accuracy = np.mean(val_predictions_binary == val_labels)
    # Calculate AUC using the validation set only
    auc_score = roc_auc_score(val_labels, val_predictions)

    return accuracy, auc_score


# Perform evaluation
accuracy, auc_score = evaluate_model(fine_tuned_models, val_sequences, val_features, val_labels)
print(f"Validation Accuracy: {accuracy:.4f}")
print(f"AUC: {auc_score:.4f}")

# with open('reduced_features_index_' + test_number + '.pkl', 'rb') as f:
#     reduced_features_index = pickle.load(f)

end_time = time.time()
total_time = end_time - start_time
print(f"The total running time of the program : {total_time:.2f} s")


