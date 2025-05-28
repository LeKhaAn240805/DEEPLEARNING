import numpy as np
import tensorflow as tf
import os
import pandas as pd
import matplotlib.pyplot as plt
from statistics import mean, stdev
from nltk.translate.bleu_score import sentence_bleu

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, TimeDistributed, Dropout, Embedding, Bidirectional
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.losses import sparse_categorical_crossentropy
from sentence_transformers import SentenceTransformer

# --- Configurations ---
MAX_NUM_WORDS = 10000
MAX_SEQUENCE_LENGTH = 100
EMBEDDING_DIM = 300
GRU_UNITS = 256
DROPOUT_RATE = 0.5
EPOCHS = 20
embedding_model = SentenceTransformer('dangvantuan/vietnamese-document-embedding', trust_remote_code=True)

# --- Suppress TensorFlow warnings and disable GPU ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# --- Paths ---
csv_path = 'D:/PhoMT.csv'
log_dir = 'D:/'
os.makedirs(log_dir, exist_ok=True)

# --- Load Data ---
df = pd.read_csv(csv_path)
df.dropna(subset=['EnglishSentences', 'VietnameseSentences'], inplace=True)

english_sentences = df['EnglishSentences'].str.strip().tolist()
vietnamese_sentences = ['<start> ' + sent.strip() + ' <end>' for sent in df['VietnameseSentences']]
split_index = int(len(english_sentences) * 0.8)
# --- Tokenization ---
def tokenize(sentences, num_words):
    tokenizer = Tokenizer(num_words=num_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(sentences)
    sequences = tokenizer.texts_to_sequences(sentences)
    return sequences, tokenizer

def preprocess_with_embedding(x, y, tokenizer, max_sequence_length):
    x_encoded = np.array([embedding_model.encode(sentence) for sentence in x])
    y_sequences = tokenizer.texts_to_sequences(y)
    y_padded = pad_sequences(y_sequences, maxlen=max_sequence_length, padding='post')
    return x_encoded, y_padded


# Tiền xử lý dữ liệu
x_train_encoded, y_train_encoded = preprocess_with_embedding(
    english_sentences[:split_index], vietnamese_sentences[:split_index]
)
x_val_encoded, y_val_encoded = preprocess_with_embedding(
    english_sentences[split_index:], vietnamese_sentences[split_index:]
)


# --- BLEU Evaluation ---
def decode_predictions(preds, tokenizer):
    index_to_word = {idx: word for word, idx in tokenizer.word_index.items()}
    index_to_word[0] = ''  # Padding
    results = []
    for pred_seq in preds:
        pred_ids = np.argmax(pred_seq, axis=-1)  # Lấy chỉ số lớn nhất
        words = [index_to_word.get(idx, '') for idx in pred_ids]
        sentence = ' '.join([w for w in words if w])
        results.append(sentence)
    return results



def evaluate_bleu(model, x_val, y_val, tokenizer, batch_size):
    preds = model.predict(x_val, batch_size=batch_size, verbose=0)
    pred_sentences = decode_predictions(preds, tokenizer)
    true_sentences = decode_predictions(y_val, tokenizer)
    bleu_scores = [
        sentence_bleu([ref.split()], hyp.split(), weights=(0.5, 0.5))
        for ref, hyp in zip(true_sentences, pred_sentences)
    ]
    return np.mean(bleu_scores)


# --- Plotting ---
def plot_training_history(history, config_index, save_dir):
    plt.figure(figsize=(12, 5))
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title(f'Loss - Config {config_index}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title(f'Accuracy - Config {config_index}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plot_path = os.path.join(save_dir, f'training_plot_config_{config_index}.png')
    plt.savefig(plot_path)
    plt.close()

# --- Build Model ---
input_shape = (x_train_encoded.shape[1],)  
output_dim = len(vietnamese_tokenizer.word_index) + 1

def build_model_with_pretrained_embedding(input_shape, output_dim):
    model = Sequential([
        GRU(256, return_sequences=True, input_shape=input_shape),
        Dropout(0.5),
        GRU(256, return_sequences=True),
        Dropout(0.5),
        TimeDistributed(Dense(output_dim, activation='softmax'))  # Sử dụng TimeDistributed
    ])
    return model




def get_optimizer(name, lr):
    if name == 'adam':
        return Adam(learning_rate=lr, clipnorm=1.0)  # <-- thêm dòng này
    elif name == 'rmsprop':
        return RMSprop(learning_rate=lr, clipnorm=1.0)  # <-- thêm dòng này
    else:
        raise ValueError(f"Unsupported optimizer: {name}")


# --- Data Prep ---
preproc_english, preproc_vietnamese, english_tokenizer, vietnamese_tokenizer = preprocess(
    english_sentences, vietnamese_sentences, MAX_SEQUENCE_LENGTH
)

tmp_x = preproc_english.reshape((-1, MAX_SEQUENCE_LENGTH))
src_vocab_size = min(MAX_NUM_WORDS, len(english_tokenizer.word_index) + 1)
tgt_vocab_size = min(MAX_NUM_WORDS, len(vietnamese_tokenizer.word_index) + 1)

split_index = int(len(tmp_x) * 0.8)
x_train = tmp_x[:split_index]
y_train = preproc_vietnamese[:split_index]
x_val = tmp_x[split_index:]
y_val = preproc_vietnamese[split_index:]

# --- Hyperparameter Grid ---
param_grid = [
    {'learning_rate': 0.001, 'batch_size': 64, 'optimizer': 'adam'},
    {'learning_rate': 0.005, 'batch_size': 32, 'optimizer': 'adam'},
    {'learning_rate': 0.001, 'batch_size': 64, 'optimizer': 'rmsprop'},
    {'learning_rate': 0.001, 'batch_size': 32, 'optimizer': 'adam'},
    {'learning_rate': 0.005, 'batch_size': 64, 'optimizer': 'rmsprop'}
]

results = []

# --- Training Loop ---
gpus = tf.config.list_physical_devices('GPU')  # Kiểm tra GPU
if gpus:
    print(f"Using GPU: {gpus[0].name}")
else:
    print("No GPU found, using CPU.")

for i, params in enumerate(param_grid):
    if len(gpus) > 0 and params['batch_size'] > 256:
        print(f"Skipping config {i+1} due to high batch size on limited resources.")
        continue

    print(f"\n🔁 Training model config {i+1}: {params}")


    model = build_model_with_pretrained_embedding(input_shape, output_dim)
    model.compile(
        optimizer=Adam(learning_rate=0.001),  # Adam optimizer với learning rate = 0.001
        loss='mse',  # Mean Squared Error để so sánh embedding vector
        metrics=['accuracy']  # Đánh giá bằng độ chính xác
        )
    history = model.fit(
        x_train,
        y_train,
        batch_size=params['batch_size'],
        epochs=EPOCHS,
        validation_data=(x_val, y_val),
        verbose=1
    )

    val_loss = history.history['val_loss'][-1]
    val_acc = history.history['val_accuracy'][-1]
    avg_bleu = evaluate_bleu(model, x_val, y_val, vietnamese_tokenizer, batch_size=params['batch_size'])

    results.append({
        'config': params,
        'val_loss': val_loss,
        'val_accuracy': val_acc,
        'bleu': avg_bleu
    })

    with open(os.path.join(log_dir, f'summary_config_{i+1}.txt'), 'w', encoding='utf-8') as f:
        f.write(f"Config: {params}\n")
        f.write(f"Validation Loss: {val_loss:.4f}\n")
        f.write(f"Validation Accuracy: {val_acc:.4f}\n")
        f.write(f"BLEU Score: {avg_bleu:.4f}\n")
        for key, values in history.history.items():
            f.write(f"{key}: {values}\n")

    model.save(os.path.join(log_dir, f'model_config_{i+1}.h5'))
    plot_training_history(history, i + 1, log_dir)

# --- Final Summary ---
val_losses = [r['val_loss'] for r in results]
val_accuracies = [r['val_accuracy'] for r in results]
bleu_scores = [r['bleu'] for r in results]

print("\n=== Tổng hợp kết quả các cấu hình ===")
print(f"Loss - Mean: {mean(val_losses):.4f}, Std: {stdev(val_losses):.4f}")
print(f"Accuracy - Mean: {mean(val_accuracies):.4f}, Std: {stdev(val_accuracies):.4f}")
print(f"BLEU - Mean: {mean(bleu_scores):.4f}, Std: {stdev(bleu_scores):.4f}")
