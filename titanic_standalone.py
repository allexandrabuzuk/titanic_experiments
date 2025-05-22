import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from catboost import CatBoostClassifier
from sklearn import neighbors, svm
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.tree import DecisionTreeClassifier
from torch.utils.data import Dataset, DataLoader

import wandb


# --------- Preprocessing ---------
def create_preprocessor():
    def add_name_length(X):
        X = X.copy()
        X['Name_length'] = X['Name'].apply(lambda x: len(x) if isinstance(x, str) else 0)
        return X

    add_name_length_transformer = FunctionTransformer(add_name_length)
    drop_cols = ['Cabin', 'Ticket', 'PassengerId']

    def drop_columns(X):
        X = X.copy()
        return X.drop([c for c in drop_cols if c in X.columns], axis=1)

    numerical_cols = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Name_length']
    categorical_cols = ['Sex', 'Embarked']

    numerical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numerical_pipeline, numerical_cols),
        ('cat', categorical_pipeline, categorical_cols)
    ])

    return Pipeline(steps=[
        ('drop_cols', FunctionTransformer(drop_columns)),
        ('add_name_length', add_name_length_transformer),
        ('preprocessor', preprocessor)
    ])


def transformer_preprocess(data_train):
    data_train = data_train.copy()
    data_train = data_train.drop(columns=['Cabin', 'Ticket', 'PassengerId'], errors='ignore')
    data_train['Name_length'] = data_train['Name'].apply(lambda x: len(x) if isinstance(x, str) else 0)
    data_train['Age'] = data_train['Age'].fillna(data_train['Age'].median())
    data_train['Fare'] = data_train['Fare'].fillna(data_train['Fare'].median())
    data_train['Sex'] = data_train['Sex'].map({'male': 0, 'female': 1})
    data_train['Embarked'] = data_train['Embarked'].fillna(data_train['Embarked'].mode()[0])
    data_train['Embarked'] = data_train['Embarked'].map({v: k for k, v in enumerate(data_train['Embarked'].unique())})
    return data_train[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Name_length', 'Sex', 'Embarked']]


# --------- PyTorch Components ---------
class TitanicDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X.values, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class TabTransformer(nn.Module):
    def __init__(self, num_features, d_model=64, nhead=8, num_layers=3, dropout=0.2):
        super().__init__()
        self.embedding = nn.Linear(1, d_model)
        self.pos_encoder = nn.Parameter(torch.randn(1, num_features, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 2)
        )

    def forward(self, x):
        x = x.unsqueeze(-1)  # (batch, features, 1)
        x = self.embedding(x) + self.pos_encoder
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.classifier(x)


class TabRNN(nn.Module):
    def __init__(self, num_features, hidden_size=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.gru = nn.GRU(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 2)
        )

    def forward(self, x):
        x = x.unsqueeze(-1)  # (batch, features, 1)
        output, _ = self.gru(x)
        return self.classifier(output[:, -1, :])


def train_pytorch_model(model, X_train, y_train, X_test, y_test, config, model_name):
    train_dataset = TitanicDataset(X_train, y_train)
    test_dataset = TitanicDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        device = torch.device("mps" if torch.mps.is_available() else device)
    except:
        pass
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(config.epochs):
        model.train()
        total_loss = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        wandb.log({f"{model_name}/train_loss": total_loss / len(train_loader)})

    model.eval()
    correct = 0
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            correct += (outputs.argmax(1) == y).sum().item()

    acc = correct / len(test_dataset)
    wandb.log({f"{model_name}/test_acc": acc})
    return acc


# --------- Main Experiment ---------
def model_comparison():
    wandb.init(project="Titanic_Models_2", config={
        "batch_size": 64,
        "lr": 3e-4,
        "transformer_epochs": 200,
        "rnn_epochs": 200,
        "transformer_d_model": 64,
        "rnn_hidden_size": 128
    })
    config = wandb.config

    # Load and prepare data
    data_train = pd.read_csv("train.csv")
    X = data_train.drop("Survived", axis=1)
    y = data_train["Survived"]

    # Models
    models = {"LogisticRegression": LogisticRegression(max_iter=500),
              "DecisionTree": DecisionTreeClassifier(),
              "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=20, criterion="entropy"),
              "GradientBoosting": GradientBoostingClassifier(learning_rate=0.01, max_depth=10,
                                                             min_samples_split=15, n_estimators=250,
                                                             subsample=0.5),
              "MLP": MLPClassifier(max_iter=350, activation="tanh",
                                   learning_rate="constant",
                                   alpha=0.0001, hidden_layer_sizes=(50, 50, 50),
                                   solver="sgd"
                                   ),
              "Bayes": GaussianNB(),
              "BayesBernoulli": BernoulliNB(),
              "CatBoost": CatBoostClassifier(iterations=700, learning_rate=0.05, depth=3,
                                             l2_leaf_reg=5),
              "KNN": neighbors.KNeighborsClassifier(n_neighbors=3),
              "Svm": svm.SVC(probability=True),
              "VotingEnsemble": VotingClassifier(estimators=[
                  ('mlp', MLPClassifier(max_iter=350, activation="tanh",
                                        learning_rate="constant",
                                        alpha=0.0001, hidden_layer_sizes=(50, 50, 50),
                                        solver="sgd"
                                        )),
                  ('rf', LogisticRegression(max_iter=700)),
                  ('gb', GradientBoostingClassifier(learning_rate=0.2, max_depth=3,
                                                    min_samples_split=5, n_estimators=100,
                                                    subsample=0.8))
              ], voting='soft'),
              "VotingEnsembleRGC": VotingClassifier(estimators=[
                  ('rf', RandomForestClassifier(n_estimators=100, max_depth=20, criterion="entropy")),
                  ('gb', GradientBoostingClassifier(learning_rate=0.2, max_depth=3,
                                                    min_samples_split=5, n_estimators=100,
                                                    subsample=0.8)),
                  ('cb', CatBoostClassifier(iterations=700, learning_rate=0.05, depth=3,
                                            l2_leaf_reg=5),)
              ], voting='soft')
              }

    preprocessor = create_preprocessor()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    for name, model in models.items():
        pipe = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
        pipe.fit(X_train, y_train)
        acc = accuracy_score(y_test, pipe.predict(X_test))
        wandb.log({f"{name}/acc": acc})

    # Neural models
    df_processed = transformer_preprocess(data_train)
    X_train_nn, X_test_nn, y_train_nn, y_test_nn = train_test_split(
        df_processed, y, test_size=0.2)

    # Transformer
    transformer = TabTransformer(
        num_features=X_train_nn.shape[1],
        d_model=config.transformer_d_model
    )
    transformer_acc = train_pytorch_model(
        transformer, X_train_nn, y_train_nn, X_test_nn, y_test_nn,
        config=type('', (), {"epochs": config.transformer_epochs, "batch_size": config.batch_size, "lr": config.lr}),
        model_name="Transformer"
    )

    # RNN
    rnn = TabRNN(
        num_features=X_train_nn.shape[1],
        hidden_size=config.rnn_hidden_size
    )

    rnn_acc = train_pytorch_model(
        rnn, X_train_nn, y_train_nn, X_test_nn, y_test_nn,
        config=type('', (), {"epochs": config.rnn_epochs, "batch_size": config.batch_size, "lr": config.lr}),
        model_name="RNN"
    )


    # Final report
    results = wandb.Table(columns=["Model", "Accuracy"])
    for name in models:
        results.add_data(name, wandb.run.summary[f"{name}/acc"])
    results.add_data("Transformer", transformer_acc)
    results.add_data("RNN", rnn_acc)
    wandb.log({"Model Comparison": results})

    wandb.finish()

# SWEEP SETTINGS
def parameter_sweep():
    wandb.init()
    config = wandb.config

    # -----RandomForest-----
    model = RandomForestClassifier(n_estimators=config.n_estimators,
                                   max_depth=config.max_depth,
                                   criterion=config.criterion,
                                   min_samples_split=config.min_samples_split
                                   )
    # # -----MLP-----
    # model = MLPClassifier(max_iter=config.max_iter,
    #                       activation=config.activation,
    #                       hidden_layer_sizes=config.hidden_layer_sizes,
    #                       solver=config.solver,
    #                       alpha=config.alpha,
    #                       learning_rate=config.learning_rate
    #                       )

    # -----GradientBosting-----
    # model = GradientBoostingClassifier(
    #     n_estimators=config.n_estimators,
    #     learning_rate=config.learning_rate,
    #     max_depth=config.max_depth,
    #     subsample=config.subsample,
    #     min_samples_split=config.min_samples_split)
    # ---CatBoost---
    # model = CatBoostClassifier(
    #     iterations=config.iterations,
    #     learning_rate=config.learning_rate,
    #     depth=config.depth,
    #     l2_leaf_reg=config.l2_leaf_reg,
    #     verbose=0 # Suppress verbose output
    # )

    df = pd.read_csv("train.csv")
    X = df.drop("Survived", axis=1)
    y = df["Survived"]
    preprocessor = create_preprocessor()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    pipe = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    precision = report["macro avg"]["precision"]
    recall = report["macro avg"]["recall"]

    # Log the results
    wandb.log(
        {"accuracy": accuracy, "precision": precision, "recall": recall})

    print(f"Run with {config.items()} --> Accuracy: {accuracy}")


def do_sweep():

    # ----  LIST OF CONFIGURATION ----

    # ---RandomForest---
    config = {
        "method": "grid",  # Can be "grid" or "bayes" as well
        "metric": {"name": "accuracy", "goal": "maximize"},
        "parameters": {
            'n_estimators': {
                'values': [50, 100, 120, 150, 170, 190, 210]
            },
            'max_depth': {
                'values': [None, 10, 15, 20, 25, 30, 35]  # None means no maximum depth
            },
            'criterion': {
                'values': ['gini', 'entropy']
            },
            'min_samples_split': {
                'values': [2, 3]
            },
        },
    }

    # ---MPL---
    # config = {
    #     "method": "grid",  # Can be "grid" or "bayes" as well
    #     "metric": {"name": "accuracy", "goal": "maximize"},
    #     "parameters": {
    #         'max_iter': {
    #             'values': [200, 350, 500, 700, 900]
    #         },
    #         'activation': {
    #             'values': ['tanh', 'relu']
    #         },
    #         'hidden_layer_sizes': {
    #             'values': [(50, 50, 50), (50, 100, 50), (100,)]
    #         },
    #         'solver': {
    #             'values': ['sgd', 'adam']
    #         },
    #         'alpha': {
    #             'values': [0.0001, 0.05]
    #         },
    #         'learning_rate': {
    #             'values': ['constant', 'adaptive']
    #         },
    #     },
    # }
    # ---GradientBoosting---
    # config = {
    #     'method': 'bayes',  # optimization method: random, grid, or bayes
    #     'metric': {
    #         'name': 'accuracy',
    #         'goal': 'maximize'
    #     },
    #     'parameters': {
    #         'n_estimators': {
    #             'values': [50, 100, 150, 200, 250]
    #         },
    #         'learning_rate': {
    #             'values': [0.01, 0.1, 0.2, 0.3]
    #         },
    #         'max_depth': {
    #             'values': [3, 5, 7, 9, 10]
    #         },
    #         'subsample': {
    #             'values': [0.3, 0.5, 0.8, 1.0]
    #         },
    #         'min_samples_split': {
    #             'values': [2, 5, 10, 15]
    #         }
    #     }
    # }
    # config = {
    #     "method": "random",  # Can also be "random" or "bayes"
    #     "metric": {"name": "accuracy", "goal": "maximize"},
    #     "parameters": {
    #         "iterations": {
    #             "values": [500, 700, 900, 1000, 1100]
    #         },
    #         "learning_rate": {
    #             "values": [0.02, 0.05, 0.1, 0.2]
    #         },
    #         "depth": {
    #             "values": [3, 4, 5]
    #         },
    #         "l2_leaf_reg": {
    #             "values": [1, 3, 5, 7, 8]
    #         }
    #     },
    # }

    sweep_id = wandb.sweep(config, project="Titanic_Models_2")
    wandb.agent(sweep_id, parameter_sweep)

if __name__ == "__main__":
    #----- Runs the main model comparison and logs to W&B-----
    model_comparison()
    #----- Runs sweep on selected model with selected configuration, logs to W&B -----
    # do_sweep()
