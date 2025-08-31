# EEG-Chatbot-Generative-AI


### A Voice-Enabled Healthcare Chatbot for Neurodegenerative Disorder Identification
This chatbot, either with a female or male voice option, talks with the user in natural language and helps the user identify what kind of neurodegenerative disorder one might have based on the symptoms the user provides.
---

## **Overview**
This project is a voice-enabled healthcare chatbot that utilizes deep learning to assist users in identifying potential neurodegenerative disorders based on their symptoms. The chatbot supports both **female and male voice options** and interacts in natural language conversations to provide insights.

**Key Features:**
- Natural language processing (NLP) for symptom analysis.
- Voice interaction (male/female options).
- Deep learning model trained to recognize patterns in neurodegenerative disorders.
---

## 🛠 **Technologies & Libraries**

| Library/Module          | Purpose                                                                                     |
|-------------------------|---------------------------------------------------------------------------------------------|
| **NLTK**                | Natural Language Toolkit for text processing (tokenization, stemming, lemmatization).      |
| **WordNetLemmatizer**   | Groups inflected word forms (e.g., "running" → "run") for consistent analysis.             |
| **Keras (Sequential)**  | Builds a stack of neural network layers for the chatbot’s deep learning model.              |
| **Dense Layers**        | Fully connected layers to learn complex patterns from symptom data.                        |
| **Dropout**             | Prevents overfitting by randomly ignoring neurons during training.                          |
| **Activation Functions**| Determines which neurons fire (e.g., ReLU, sigmoid) to refine predictions.                  |
| **SGD Optimizer**       | Stochastic Gradient Descent for efficiently training the neural network.                  |
---

## 🚀 **Getting Started**
### Prerequisites
- Python 3.8+
- Pip (for dependency installation)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/denizyennerr/EEG-Chatbot-Generative-AI.git
   cd EEG-Chatbot-Generative-AI
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the chatbot:
   ```bash
   python main.py
   ```
---

## **How It Works**
1. **User Input**: The chatbot asks about symptoms in natural language (e.g., "I’ve been forgetting things lately").
2. **NLP Processing**: Symptoms are lemmatized and analyzed using NLTK.
3. **Deep Learning Model**: The Sequential model predicts potential disorders based on trained patterns.
4. **Voice Response**: The chatbot responds with a voice (male/female) and suggests next steps (e.g., "Consider consulting a neurologist").

---
## 📂 **Project Structure**
```
EEG-Chatbot-Generative-AI/
│
├── venv/                  # Virtual environment (excluded from version control)
│
├── .gitattributes         # Git configuration for file attributes
├── .gitignore              # Specifies files/directories ignored by Git
│
├── chatbot                 # Main chatbot application logic
│
├── chatbot_model.keras     # Pre-trained Keras model for symptom analysis
│
├── classes.pkl             # Serialized class labels (e.g., disorder categories)
├── classes1                # Additional class-related data (if applicable)
│
├── intents                 # Training data: user intents and symptom patterns
│
├── LICENSE                 # Project license (e.g., MIT)
├── README.md               # Project documentation (this file)
│
├── train_chatbot           # Script to train the chatbot model
├── verify_env              # Environment verification script (e.g., checks dependencies)
│
└── words.pkl               # Serialized vocabulary (e.g., lemmatized words)
```
---

## 🤝 **Contributing**
Contributions are welcome! Here’s how you can help:
- **Report bugs**: Open an issue with details.
- **Suggest features**: Share ideas for improvements.
- **Submit pull requests**: Fix bugs or add features.

**Guidelines**:
- Follow PEP 8 for Python code.
- Document new features in the `README.md`.
- Test changes before submitting.
---

## 📜 **License**
This project is licensed under the **MIT License** – see [LICENSE](LICENSE) for details.
---

## 📬 **Contact**
For questions or collaborations, reach out:
- **Email**: deniz.yener@proton.me
- **GitHub**: [@denizyennerr](https://github.com/denizyennerr)

---
**Note**: This chatbot is for **educational and preliminary use only**. Always consult a healthcare professional for medical advice.
