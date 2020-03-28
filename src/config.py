import transformers

MAX_LEN = 280
TRAIN_BATCH_SIZE = 16
VALID_BATCH_SIZE = 16
EPOCHS = 5
BERT_PATH = '../input/bert_base_uncased/'
MODEL_PATH = 'model.bin'
TRAIN_FILE = '../input/imdb.csv'
TOKENIZER = transformers.BertTokenizer.from_pretrained(BERT_PATH, do_lower_case=True)
