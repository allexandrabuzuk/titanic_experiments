import pandas as pd
from catboost import CatBoostClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer


#
# SEED = random.randint(0, 10000)
# np.random.seed(SEED)
# random.seed(SEED)
# torch.manual_seed(SEED)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed_all(SEED)

# print(SEED)


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


if __name__ == '__main__':
    train = pd.read_csv("train.csv")
    X = train.drop("Survived", axis=1)
    y = train["Survived"]

#------ A list of models with optimized parameters -----
    # model = GradientBoostingClassifier(learning_rate=0.01, max_depth=10,
    #                                    min_samples_split=15, n_estimators=250,
    #                                    subsample=0.5)

    # model = CatBoostClassifier(iterations=500, learning_rate=0.05, depth=3,
    #                    l2_leaf_reg=5)

    # model = VotingClassifier(estimators=[
    #     ('mlp', MLPClassifier(max_iter=350, activation="tanh",
    #                           learning_rate="constant",
    #                           alpha=0.0001, hidden_layer_sizes=(50, 50, 50),
    #                           solver="sgd"
    #                           )),
    #     ('rf', LogisticRegression(max_iter=700)),
    #     ('gb', GradientBoostingClassifier(learning_rate=0.01, max_depth=10,
    #                                       min_samples_split=15, n_estimators=250,
    #                                       subsample=0.5))])

    # model = MLPClassifier(max_iter=350, activation="tanh",
    #                                     learning_rate="constant",
    #                                     alpha=0.0001, hidden_layer_sizes=(50, 50, 50),
    #                                     solver="sgd"
    #                                     )

    # model = RandomForestClassifier(n_estimators=190, max_depth=20, criterion="entropy", min_samples_split=7)

    # model = VotingClassifier(estimators=[
    #               ('rf', RandomForestClassifier(n_estimators=190, max_depth=20, criterion="entropy", min_samples_split=5)),
    #               ('gb', GradientBoostingClassifier(learning_rate=0.2, max_depth=5,
    #                                                 min_samples_split=10, n_estimators=100,
    #                                                 subsample=0.8)),
    #               ('cb', CatBoostClassifier(iterations=900, learning_rate=0.2, depth=3,
    #                                         l2_leaf_reg=8),)
    #           ], voting='soft')

    preprocessor = create_preprocessor()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    pipe = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
    pipe.fit(X_train, y_train)
    # pipe.fit(X, y)

    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Train accuracy {acc:.4f}\n")

    test = pd.read_csv("test.csv")
    prediction = pipe.predict(test)
    test.insert(0, "Survived", prediction)
    test = test.drop(columns=["Pclass", "Name", "Sex", "Age", "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked"])
    print(prediction)
    test.to_csv("test_predicted.csv")
