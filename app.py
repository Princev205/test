```python
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# --------------------------------------------------
# PAGE
# --------------------------------------------------

st.set_page_config(
    page_title="Tweet Engagement Predictor",
    page_icon="📊",
    layout="centered"
)

st.title("📊 Tweet Engagement Predictor")
st.write(
    "Predict whether a tweet is likely to have **High** or **Low** engagement."
)


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

@st.cache_data
def load_data():
    df = pd.read_csv("tweets.csv")

    # Clean Potential Impressions
    df["Potential Impressions"] = (
        df["Potential Impressions"]
        .astype(str)
        .str.replace(",", "", regex=False)
        .astype(float)
    )

    # Feature engineering
    df["TextLength"] = df["Text"].astype(str).str.len()

    df["Hashtags"] = (
        df["Text"]
        .astype(str)
        .str.count("#")
    )

    df["HasLink"] = (
        df["Text"]
        .astype(str)
        .str.contains("http", case=False, na=False)
        .astype(int)
    )

    df["PostingHour"] = (
        pd.to_datetime(df["Time"], errors="coerce")
        .dt.hour
    )

    # Remove rows with missing required values
    df = df.dropna(
        subset=[
            "Text",
            "TextLength",
            "Hashtags",
            "HasLink",
            "PostingHour",
            "Potential Impressions"
        ]
    )

    # Median split
    median = df["Potential Impressions"].median()

    df["Engagement"] = np.where(
        df["Potential Impressions"] >= median,
        "High",
        "Low"
    )

    return df


df = load_data()


# --------------------------------------------------
# FEATURES AND TARGET
# --------------------------------------------------

X = df[
    [
        "Text",
        "TextLength",
        "Hashtags",
        "HasLink",
        "PostingHour"
    ]
]

y = df["Engagement"]


# --------------------------------------------------
# TRAIN / TEST SPLIT
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# --------------------------------------------------
# PREPROCESSING
# --------------------------------------------------

preprocessor = ColumnTransformer([
    (
        "text",
        TfidfVectorizer(
            stop_words="english"
        ),
        "Text"
    ),

    (
        "num",
        "passthrough",
        [
            "TextLength",
            "Hashtags",
            "HasLink",
            "PostingHour"
        ]
    )
])


# --------------------------------------------------
# MODEL
# --------------------------------------------------

model = Pipeline([
    (
        "preprocessor",
        preprocessor
    ),

    (
        "model",
        RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
    )
])


# --------------------------------------------------
# TRAIN
# --------------------------------------------------

@st.cache_resource
def train_model(X_train, y_train):
    model.fit(X_train, y_train)
    return model


model = train_model(X_train, y_train)


# --------------------------------------------------
# EVALUATION
# --------------------------------------------------

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(
    y_test,
    y_pred,
    pos_label="High"
)
recall = recall_score(
    y_test,
    y_pred,
    pos_label="High"
)
f1 = f1_score(
    y_test,
    y_pred,
    pos_label="High"
)


# --------------------------------------------------
# MODEL PERFORMANCE
# --------------------------------------------------

st.subheader("📈 Model Performance")

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Accuracy",
        f"{accuracy * 100:.2f}%"
    )

    st.metric(
        "Precision",
        f"{precision * 100:.2f}%"
    )

with col2:
    st.metric(
        "Recall",
        f"{recall * 100:.2f}%"
    )

    st.metric(
        "F1 Score",
        f"{f1 * 100:.2f}%"
    )


# --------------------------------------------------
# DATASET INFO
# --------------------------------------------------

st.subheader("📊 Dataset")

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Total Tweets",
        len(df)
    )

with col2:
    st.metric(
        "Median Impressions",
        f"{df['Potential Impressions'].median():,.0f}"
    )


# --------------------------------------------------
# USER INPUT
# --------------------------------------------------

st.subheader("🔮 Predict Tweet Engagement")

tweet = st.text_area(
    "Enter Tweet Text",
    placeholder="Enter your tweet here..."
)

text_length = len(tweet)

hashtags = tweet.count("#")

has_link = int(
    "http" in tweet.lower()
)

posting_hour = st.slider(
    "Posting Hour",
    min_value=0,
    max_value=23,
    value=12
)


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

if st.button("Predict Engagement"):

    if tweet.strip() == "":
        st.warning("Please enter a tweet.")

    else:

        new_tweet = pd.DataFrame({
            "Text": [tweet],
            "TextLength": [text_length],
            "Hashtags": [hashtags],
            "HasLink": [has_link],
            "PostingHour": [posting_hour]
        })

        prediction = model.predict(new_tweet)[0]

        probabilities = model.predict_proba(new_tweet)[0]

        classes = model.classes_

        high_index = list(classes).index("High")

        high_probability = probabilities[high_index]

        st.subheader("Prediction")

        if prediction == "High":
            st.success("🔥 HIGH ENGAGEMENT")

        else:
            st.info("📉 LOW ENGAGEMENT")

        st.write(
            f"Probability of High Engagement: "
            f"**{high_probability * 100:.2f}%**"
        )

        st.progress(
            float(high_probability)
        )


# --------------------------------------------------
# FEATURE IMPORTANCE
# --------------------------------------------------

st.subheader("🔍 Important Model Features")

feature_names = model[
    "preprocessor"
].get_feature_names_out()

importances = model[
    "model"
].feature_importances_

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
}).sort_values(
    "Importance",
    ascending=False
)

st.dataframe(
    importance_df.head(20),
    use_container_width=True
)
```
