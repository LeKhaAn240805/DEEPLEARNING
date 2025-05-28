import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import underthesea
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import TensorBoard, LearningRateScheduler
import datetime
import numpy as np

def get_article_links(homepage_url, limit=100):
    resp = requests.get(homepage_url)
    soup = BeautifulSoup(resp.text, 'html.parser')
    links = {
        a['href'] for a in soup.find_all('a', href=True)
        if a['href'].startswith('https://vnexpress.net/') and len(a['href'].split('/')) > 3
    }
    return list(links)[:limit]

def crawl_vnexpress_article(url):
    try:
        resp = requests.get(url, timeout=5)
        soup = BeautifulSoup(resp.text, 'html.parser')
        
        title_tag = soup.find('h1', class_='title-detail')
        content_tags = soup.find_all('p', class_='Normal')
        category_tag = soup.find('ul', class_='breadcrumb')
        
        if title_tag and content_tags and category_tag:
            title = title_tag.text.strip()
            content = ' '.join(p.text.strip() for p in content_tags)
            categories = category_tag.find_all('li')
            category = categories[1].text.strip() if len(categories) > 1 else "Unknown"
            return title, content, category
    except:
        return None, None, None

def clean_text(text):
    text = text.lower().replace('\n', ' ').replace('\r', ' ')
    return underthesea.word_tokenize(text, format="text")

def build_model(num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(20000, 128, input_length=500),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Learning rate scheduler example (optional)
def lr_schedule(epoch, lr):
    if epoch > 5:
        return lr * 0.5
    return lr

def main():
    homepage = "https://vnexpress.net"
    links = get_article_links(homepage, limit=100)

    data = []
    for link in links:
        title, content, category = crawl_vnexpress_article(link)
        if title and content:
            data.append({'title': title, 'content': content, 'category': category})
        time.sleep(0.5)

    df = pd.DataFrame(data)
    df['title_clean'] = df['title'].apply(clean_text)
    df['content_clean'] = df['content'].apply(clean_text)

    texts = (df['title_clean'] + " " + df['content_clean']).tolist()
    labels = df['category'].tolist()

    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels)

    tokenizer = Tokenizer(num_words=20000, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(sequences, maxlen=500, padding='post')

    X_train, X_test, y_train, y_test = train_test_split(padded, labels_encoded, test_size=0.2, random_state=42)

    # Các cấu hình siêu tham số (bạn có thể thêm hoặc sửa)
    configs = [
        {'epochs': 10, 'batch_size': 32},
        {'epochs': 15, 'batch_size': 32},
        {'epochs': 10, 'batch_size': 64},
    ]

    results = []

    for idx, cfg in enumerate(configs):
        print(f"Training config {idx+1}/{len(configs)}: epochs={cfg['epochs']}, batch_size={cfg['batch_size']}")

        model = build_model(len(le.classes_))

        log_dir = f"logs/fit/config_{idx+1}_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_cb = TensorBoard(log_dir=log_dir, histogram_freq=1)
        lr_scheduler = LearningRateScheduler(lr_schedule)

        history = model.fit(
            X_train, y_train,
            epochs=cfg['epochs'],
            validation_data=(X_test, y_test),
            batch_size=cfg['batch_size'],
            callbacks=[tensorboard_cb, lr_scheduler],
            verbose=2
        )

        # Lấy loss và accuracy cuối cùng của training và validation
        train_loss = history.history['loss'][-1]
        val_loss = history.history['val_loss'][-1]
        train_acc = history.history['accuracy'][-1]
        val_acc = history.history['val_accuracy'][-1]

        results.append({
            'config': cfg,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_acc': train_acc,
            'val_acc': val_acc
        })

    # Tính trung bình và độ lệch chuẩn các metrics trên các cấu hình
    def mean_std(metric):
        vals = [r[metric] for r in results]
        return np.mean(vals), np.std(vals)

    print("\n=== Kết quả tổng hợp ===")
    print(f"Train loss mean/std: {mean_std('train_loss')}")
    print(f"Validation loss mean/std: {mean_std('val_loss')}")
    print(f"Train accuracy mean/std: {mean_std('train_acc')}")
    print(f"Validation accuracy mean/std: {mean_std('val_acc')}")

if __name__ == '__main__':
    main()
