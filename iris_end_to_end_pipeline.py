# %% [markdown]
# # Iris end to ent pipeline

# load libraries

# %%
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import seaborn as sns

# %%
iris_df = pd.read_csv("https://repo.hops.works/master/hopsworks-tutorials/data/iris.csv")
iris_df.sample(10)


# %%
sns.set(style='white', color_codes = True)
sns.boxplot(x ='variety', y = "sepal_length", data=iris_df)


# %%
sns.set(style='white', color_codes = True)
sns.boxplot(x ='variety', y = "sepal_width", data=iris_df)

# %%
sns.set(style='white', color_codes = True)
sns.boxplot(x ='variety', y = "petal_width", data=iris_df)


# %%
sns.set(style='white', color_codes = True)
sns.boxplot(x ='variety', y = "petal_length", data=iris_df)


# %%
features = iris_df[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
labels = iris_df[["variety"]]
features

# %%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size = 0.2)
y_train

# %%
model = KNeighborsClassifier(n_neighbors=2)
model.fit(X_train, y_train.values.ravel())


# %%
y_pred = model.predict(X_test)
y_pred

# %%
from sklearn.metrics import classification_report

metrics = classification_report(y_test, y_pred, output_dict = True)
print(metrics)

# %%
from sklearn.metrics import confusion_matrix

results = confusion_matrix(y_test, y_pred)
print(results)

# %%
from matplotlib import pyplot

df_cm = pd.DataFrame(results, ['True Setosa', 'True Versicolor', 'True Virginica'],
                     ['Pred Setosa', 'Pred Versicolor', 'Pred Virginica'])

sns.heatmap(df_cm, annot=True)

# %%
import gradio as gr
import numpy as np
from PIL import Image
import requests


def iris(sepal_length, sepal_width, petal_length, petal_width):
    input_list = []
    input_list.append(sepal_length)
    input_list.append(sepal_width)
    input_list.append(petal_length)
    input_list.append(petal_width)
    # 'res' is a list of predictions returned as the label.
    res = model.predict(np.asarray(input_list).reshape(1, -1))
    # We add '[0]' to the result of the transformed 'res', because 'res' is a list, and we only want
    # the first element.
#     flower_url = "https://repo.hops.works/master/hopsworks-tutorials/data/" + res[0] + ".png"
    flower_url = "https://raw.githubusercontent.com/featurestoreorg/serverless-ml-course/main/src/01-module/assets/" + res[0] + ".png"
    img = Image.open(requests.get(flower_url, stream=True).raw)
    return img

demo = gr.Interface(
    fn=iris,
    title="Iris Flower Predictive Analytics",
    description="Experiment with sepal/petal lengths/widths to predict which flower it is.",
    allow_flagging="never",
    inputs=[
        gr.inputs.Number(default=1.0, label="sepal length (cm)"),
        gr.inputs.Number(default=1.0, label="sepal width (cm)"),
        gr.inputs.Number(default=1.0, label="petal length (cm)"),
        gr.inputs.Number(default=1.0, label="petal width (cm)"),
        ],
    outputs=gr.Image(type="pil"))

demo.launch(share=True)