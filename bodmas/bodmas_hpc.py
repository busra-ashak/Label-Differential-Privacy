# %%
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import autokeras as ak

# %%
filename = 'bodmas.npz'
data = np.load('./' + filename)
X = data['X']  # all the feature vectors
y = data['y']  # labels, 0 as benign, 1 as malicious

print(X.shape, y.shape)

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y.astype(int), test_size = 0.3, random_state = 0)

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

print(X_train.shape, X_test.shape)


# %% [markdown]
# # Train-Test

# %%
# Initialize the structured data classifier.
clf = ak.StructuredDataClassifier(
    overwrite=True, max_trials=10
)  # It tries 3 different models.
# Feed the structured data classifier with training data.
clf.fit(
    x=X_train,
    y=y_train,
    epochs=15,
)

# Evaluate the best model with testing data.
print(clf.evaluate(x=X_test, y=y_test))

# %%
model = clf.export_model()
model.summary()


