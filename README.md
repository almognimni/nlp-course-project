# **Language Modeling and Classification**

This project explores two fundamental tasks in Natural Language Processing (NLP) using the PyTorch framework: Language Modeling and Text Classification. The work is based on the "Language Modeling and Classification with PyTorch" assignment from The Academic College of Tel-Aviv, Yaffo.

## **Project Overview**

The project is divided into two main tasks:

1. **Task 1: Language Modeling**: An LSTM-based Recurrent Neural Network (RNN) is built and trained from scratch on the IMDB movie review dataset to predict the next word in a sequence.  
2. **Task 2: Text Classification**: The goal is to classify movie reviews as either positive or negative. This is accomplished through two distinct experiments:  
   * **Experiment A**: Uses the pre-trained language model from Task 1 as a frozen "backbone" to extract features for a simple downstream classifier. This demonstrates the power of transfer learning with limited data.  
   * **Experiment B**: A new classifier is built from scratch, using pre-trained Word2Vec embeddings and an LSTM layer that are trained on the full dataset.

The results from both classification experiments are then compared to draw conclusions about the effectiveness of each approach.

## **Project Structure**

The project is organized into a modular structure for clarity and reusability:

.  
├── models/                 \# Stores saved model weights (.pth) and vocabularies  
├── notebooks/              \# Contains the original exploratory Jupyter notebooks  
├── src/                    \# All source code for the project  
│   ├── \_\_init\_\_.py  
│   ├── config.py           \# Central configuration for all hyperparameters  
│   ├── data\_loader.py      \# Pytorch Dataset, DataLoader, and Vocabulary classes  
│   ├── model.py            \# All nn.Module model definitions  
│   ├── train.py            \# Training loop functions  
│   └── evaluate.py         \# Evaluation, plotting, and error analysis functions  
├── main.py                 \# Main script to run experiments from the command line  
└── requirements.txt        \# Project dependencies

## **Setup**

To get started, clone the repository and install the required dependencies.

1. **Clone the repository:**  
   git clone \<your-repository-url\>  
   cd \<your-repository-name\>

2. **Create a virtual environment (recommended):**  
   python \-m venv venv  
   source venv/bin/activate  \# On Windows, use \`venv\\Scripts\\activate\`

3. **Install dependencies:**  
   pip install \-r requirements.txt

## **Usage**

The main.py script is the primary entry point for running the experiments. It accepts a command-line argument to specify which task to execute.

### **Task 1: Train the Language Model**

This command will load the IMDB dataset, build and save the vocabulary, train the language model, and save the final model weights to the models/ directory.

python main.py task1

### **Task 2: Run Classification Experiments**

Make sure you have run task1 at least once before running task2a, as it depends on the saved language model and vocabulary.

**Experiment A (LM Backbone):**

python main.py task2a

Experiment B (Word2Vec):  
This will download the word2vec-google-news-300 model from gensim, which may take a few minutes on the first run.  
python main.py task2b

### **Run All Tasks**

To run the entire pipeline from start to finish, use the all command:

python main.py all

## **Expected Outputs**

After running the scripts, the following outputs will be generated:

* **Saved Models**: models/language\_model.pth and models/vocab.pth.  
* **Training Graphs**: Matplotlib graphs showing training and validation loss/accuracy will be displayed and can be saved.  
* **Confusion Matrices**: For classification tasks, a confusion matrix for the test set will be plotted.  
* **Error Analysis Reports**: error\_analysis\_A.txt and error\_analysis\_B.txt will be created, containing examples the models misclassified.