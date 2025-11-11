import string
import streamlit as st
import joblib
import nltk
from nltk import pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

# --- Page setup ---
st.set_page_config(page_title="📧 AI Spam Detector", page_icon="📨", layout="centered")

st.title("📧 Spam Detector")
st.caption("Classify emails or SMS messages as Spam or Ham using NLP and Machine Learning.")

# --- Load trained artifacts ---
@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load("model.joblib")
        vectorizer = joblib.load("vectorizer.joblib")
        stop_words = joblib.load("stopwords.joblib")
        return model, vectorizer, stop_words
    except FileNotFoundError:
        st.error("❌ Model files not found! Please run **training.py** first to create model.joblib, vectorizer.joblib, and stopwords.joblib.")
        st.stop()

model, vectorizer, stop_words = load_artifacts()

# --- Ensure required NLTK resources (first run only) ---
for pkg in ["averaged_perceptron_tagger", "wordnet"]:
    try:
        nltk.data.find(f"taggers/{pkg}") if "tagger" in pkg else nltk.data.find(f"corpora/{pkg}")
    except LookupError:
        nltk.download(pkg)

# --- Text cleaning helpers ---
lemmatizer = WordNetLemmatizer()

def get_wordnet_pos(word):
    tag = pos_tag([word])[0][1][0].upper()
    return {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}.get(tag, wordnet.NOUN)

def preprocess_text(text: str, stop_words=set()) -> str:
    tokens = text.lower().translate(str.maketrans('', '', string.punctuation)).split()
    lemmas = []
    for w in tokens:
        if w not in stop_words:
            pos = get_wordnet_pos(w)
            lemmas.append(lemmatizer.lemmatize(w, pos))
    return " ".join(lemmas)

# --- Sidebar ---
with st.sidebar:
    st.header("🔧 Options")
    show_probs = st.checkbox("Show Spam Probability", value=True)
    st.markdown("### Example messages:")
    examples = {
        "Free Prize": "Congratulations! You have WON a free prize. Click this link to claim now!!!",
        "Bank Alert": "Your account has been suspended. Verify your password immediately.",
        "Normal Chat": "Hey, are we still meeting for lunch tomorrow at 1pm?",
    }
    choice = st.selectbox("Choose an example", ["(none)"] + list(examples.keys()))
    if st.button("Use Example"):
        if choice in examples:
            st.session_state["msg"] = examples[choice]

# --- Main Input ---
msg = st.text_area(
    "✉️ Enter your message below:",
    key="msg",
    height=180,
    placeholder="Type or paste a message to classify as Spam or Ham...",
)

# --- Classify button ---
if st.button("Classify Message"):
    if not msg.strip():
        st.warning("Please enter a message to classify.")
    else:
        cleaned = preprocess_text(msg, stop_words)
        X_new = vectorizer.transform([cleaned])
        pred = model.predict(X_new)[0]

        label = "SPAM" if pred == 1 else "HAM"
        color = "#e74c3c" if pred == 1 else "#2ecc71"

        st.markdown(f"### Result: <span style='color:{color}'>{label}</span>", unsafe_allow_html=True)

        if show_probs and hasattr(model, "predict_proba"):
            prob_spam = model.predict_proba(X_new)[0][1]
            st.progress(float(prob_spam))
            st.write(f"Spam probability: **{prob_spam:.2%}**")

        with st.expander("Show cleaned text (model input)"):
            st.code(cleaned)

st.markdown("---")
st.caption("💡 Built with Streamlit, scikit-learn, and NLTK • UI assisted by AI (ChatGPT).")
