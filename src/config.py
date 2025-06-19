import torch

# -- General Settings --
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42
MIN_VOCAB_FREQ = 5

# -- Task 1: Language Model Config --
LM_BATCH_SIZE = 32
LM_EMBEDDING_DIM = 128
LM_HIDDEN_DIM = 256
LM_NUM_LAYERS = 2
LM_LEARNING_RATE = 0.001
LM_NUM_EPOCHS = 10
LM_MODEL_SAVE_PATH = 'models/language_model.pth'
VOCAB_SAVE_PATH = 'models/vocab.pth'


# -- Task 2, Experiment A: LM Backbone Classifier --
CLF_A_BATCH_SIZE = 32
CLF_A_SUBSET_SIZE = 5000  # 20% of 25k
CLF_A_OUTPUT_DIM = 1
CLF_A_LAYERS = 2
CLF_A_DROPOUT = 0.5
CLF_A_LEARNING_RATE = 0.001
CLF_A_NUM_EPOCHS = 10
CLF_A_ERROR_ANALYSIS_FILE = 'error_analysis_A.txt'


# -- Task 2, Experiment B: Word2Vec Classifier --
CLF_B_BATCH_SIZE = 64
CLF_B_HIDDEN_DIM = 256
CLF_B_OUTPUT_DIM = 1
CLF_B_LAYERS = 2
CLF_B_DROPOUT = 0.5
CLF_B_LEARNING_RATE = 0.001
CLF_B_NUM_EPOCHS = 10
W2V_MODEL_NAME = 'word2vec-google-news-300'
CLF_B_ERROR_ANALYSIS_FILE = 'error_analysis_B.txt'