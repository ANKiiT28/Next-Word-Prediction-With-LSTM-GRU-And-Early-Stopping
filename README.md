# Next Word Prediction using LSTM / GRU with Early Stopping

A simple deep learning project for predicting the next word in a sequence using LSTM or GRU architectures. It includes training experiments, model saving, and a Streamlit app to interactively test predictions.

---

## 🧠 Features

- Train an LSTM (or GRU) based next-word prediction model  
- Use **early stopping** to prevent overfitting  
- Save and reload tokenizer and model artifacts  
- Streamlit web app to input text and get predicted next words

---

## 📂 Repository Structure

.
├── app.py
├── experiemnts.ipynb
├── hamlet.txt
├── next_word_lstm.h5
├── tokenizer.pickle
└── README.md

markdown
Copy code

| File / Folder        | Description |
|----------------------|-------------|
| `app.py`             | Streamlit app for serving the model and making predictions |
| `experiemnts.ipynb`  | Jupyter notebook with data processing, model training & evaluation |
| `hamlet.txt`         | Sample text used for training or demonstration |
| `next_word_lstm.h5`  | Trained model weights in HDF5 format |
| `tokenizer.pickle`   | Tokenizer object used for mapping words ↔ tokens |
| `README.md`          | Project overview and instructions (you fill this) |

---

## 🛠 Getting Started

### Prerequisites

- Python 3.8+ (or compatible)  
- `venv` or conda for environment isolation  
- Basic libraries: `tensorflow`, `keras`, `numpy`, `streamlit`, `pickle`

### Setup Steps

1. Clone this repository:

   ```bash
   git clone https://github.com/ANKiiT28/Next-Word-Prediction-With-LSTM-GRU-And-Early-Stopping.git
   cd Next-Word-Prediction-With-LSTM-GRU-And-Early-Stopping
Create a virtual environment and activate it:

bash
Copy code
python -m venv venv
# On Windows PowerShell:
.\venv\Scripts\Activate.ps1
# Or on Linux/macOS:
source venv/bin/activate
Install dependencies:

bash
Copy code
pip install -r requirements.txt
If you don’t have a requirements.txt, install manually:

bash
Copy code
pip install tensorflow streamlit numpy
(Optional) Retrain the model (if you change or add data):

Open experiemnts.ipynb

Process the text data

Train with LSTM / GRU architectures

Save model & tokenizer

Run the Streamlit app:

bash
Copy code
streamlit run app.py
Visit the local URL (e.g. http://localhost:8501) in your browser to input text and see predictions.

🧩 How It Works
Data Preprocessing

Read raw text (e.g. hamlet.txt)

Tokenize the text, create n-gram sequences

Pad sequences to equal lengths

One-hot encode next-word targets

Model Architecture

Input layer

Embedding layer

LSTM (or GRU) layer(s)

Dense + softmax output over vocabulary

Training

Use EarlyStopping callback (monitor validation loss)

Save the best model (lowest validation loss)

Save the tokenizer for prediction

Prediction via Streamlit app

Load the saved model and tokenizer

Accept user text input

Tokenize, pad, feed into model → predict next word(s)

Output the top prediction(s)

🧪 Example
Input: "to be or not"

Prediction: "to"
(or whichever next word model has learned given context)

You can test many phrases in the web UI once the app is running.

⚠️ Known Issues / Tips
Loading .h5 models into newer versions of tensorflow/keras may throw errors (e.g. name scope or graph issues).

Use compile=False when loading:

python
Copy code
model = load_model("next_word_lstm.h5", compile=False)
Or convert and save in the newer native TensorFlow format (.keras) after training.

Make sure tokenizer.pickle matches exactly the tokenizer used during training; mismatches will lead to index errors or unexpected predictions.

The vocabulary is limited by your training data; unseen words in the app’s input will be treated as “out of vocabulary” and may break predictions.

🔄 Extensions & Ideas to Try
Use GRU instead of LSTM and compare performance

Stack multiple LSTM / GRU layers

Try Bidirectional RNNs

Use attention layers

Train on larger datasets or from multiple sources

Output top-k predictions instead of just one

Create a more sophisticated UI (e.g. autocomplete suggestion box)

📜 License & Credits
Feel free to use, modify, or extend this project.
If you reuse parts, a credit would be appreciated.

🙏 Acknowledgments
Thanks to the TensorFlow / Keras communities, Streamlit team, and all open source contributors.
