# %%
import numpy as np
np.random.seed(42)
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

# %%
dataset = load_wine()
x, y = dataset.data, dataset.target

# %%
df = pd.DataFrame(x, columns = dataset.feature_names)
df["y"] = y
df.head()

# %% [markdown]
# ## Cart Classifier

# %%
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size = 0.3, random_state=42)


# %%
parameters = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 2, 4, 8, 10],
    'min_samples_split': [1, 2, 4], # Wie viele Samples müssen noch in diesen Split gehen, um diesen zu erstellen?
    'min_samples_leaf': [1, 2],
    'max_features': ['auto', 'sqrt', 'log2']
}

clf = DecisionTreeClassifier()
grid_cv = GridSearchCV(clf, parameters, cv = 10, n_jobs = -1)
grid_cv.fit(x_train, y_train)

# %%
print(f"Parameters of best model: {grid_cv.best_params_}")
print(f"Score of best model: {grid_cv.best_score_}")

# %%
# Train best model
clf = DecisionTreeClassifier(
    criterion='gini', 
    max_depth=4, 
    max_features='sqrt', 
    min_samples_leaf=1, 
    min_samples_split=2)
clf.fit(x_train, y_train)
score = clf.score(x_test, y_test)
print(f"Accuracy: {score}")

# %%
dataset.feature_names, type(dataset.feature_names)

# %%
from sklearn import tree
import graphviz
# DOT data
dot_data = tree.export_graphviz(clf, out_file=None, 
                                feature_names=dataset.feature_names,  
                                class_names=dataset.target_names,
                                filled=True)

# Draw graph
graph = graphviz.Source(dot_data, format="png") 
graph

# %%
graph.render("decision_tree_graphivz")


