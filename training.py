import string
import streamlit as st
import joblib
import nltk
from nltk import pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

# --- page config ---
st.set_page_config(page_title="Spam Detector", page_icon="📧", layout="centered")

st.title("📧 Email/SMS Spam Detector")
st.caption("Demo: ML model trained on SMS Spam dataset (your custom preprocessing + RandomForest).")

# load artifacts
@st.cache_resource
def load_artifacts():
    model = joblib.load("model.joblib")
    vector = joblib.load("vectorizer.joblib")
    stop_words = joblib.load("stopwords.joblib")
    return model, vector, stop_words

model, vector, stop_words = load_artifacts()

# Ensure required NLTK resources in runtime (first deploy run)
for pkg in ["averaged_perceptron_tagger", "wordnet"]:
    try:
        nltk.data.find(f"taggers/{pkg}") if "tagger" in pkg else nltk.data.find(f"corpora/{pkg}")
    except LookupError:
        nltk.download(pkg)

lemmatizer = WordNetLemmatizer()

def get_wordnet_pos(word):
    tag = pos_tag([word])[0][1][0].upper()
    return {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}.get(tag, wordnet.NOUN)

def preprocess_text(text: str) -> str:
    tokens = text.lower().translate(str.maketrans('', '', string.punctuation)).split()
    lemmas = []
    for w in tokens:
        if w not in stop_words:
            pos = get_wordnet_pos(w)
            lemmas.append(lemmatizer.lemmatize(w, pos))
    return " ".join(lemmas)

# --- UI ---
with st.sidebar:
    st.header("🔧 Options")
    show_probs = st.checkbox("Show probabilities", value=True)
    st.markdown("**Try examples:**")
    examples = {
        "Free prize": "Congratulations! You have WON a free prize. Click this link to claim now!!!",
        "Bank alert": "Your account is compromised. Verify your password here immediately.",
        "Normal chat": "Hey, are we still meeting for lunch at 1 pm tomorrow?",
    }
    choice = st.selectbox("Examples", ["(none)"] + list(examples.keys()))
    if st.button("Use example") and choice in examples:
        st.session_state["msg"] = examples[choice]

msg = st.text_area("Paste message/email content here:", key="msg", height=180, placeholder="Type or paste the message to classify...")

if st.button("Classify"):
    if not msg.strip():
        st.warning("Please enter a message.")
    else:
        cleaned = preprocess_text(msg)
        X_new = vector.transform([cleaned])
        pred = model.predict(X_new)[0]

        label = "SPAM" if pred == 1 else "HAM"
        color = "#e74c3c" if pred == 1 else "#2ecc71"

        st.markdown(f"**Prediction:** <span style='color:{color};font-size:22px'>{label}</span>", unsafe_allow_html=True)
        if show_probs and hasattr(model, "predict_proba"):
            prob_spam = model.predict_proba(X_new)[0][1]
            st.progress(float(prob_spam))
            st.write(f"Spam probability: **{prob_spam:.2%}**")

        with st.expander("Show cleaned text (model input)"):
            st.code(cleaned)

st.markdown("---")
st.caption("Built with Streamlit • RandomForest • CountVectorizer • NLTK Lemmatizer + POS")
