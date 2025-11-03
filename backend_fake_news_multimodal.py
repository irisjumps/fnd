"""
Fake News Detection — Multimodal (text + image + metadata)
Single-file Python toolkit that:
 - Loads a multimodal CSV dataset with columns: id, text, image_path, label, (optional) metadata...
 - Extracts TF-IDF features for text
 - Extracts image embeddings using Keras MobileNetV2 (pretrained, include_top=False, pooling='avg')
 - Trains and evaluates multiple models with 5-fold cross-validation:
     * Logistic Regression (on combined text+image features)
     * Linear SVM (on combined features)
     * Random Forest (on combined features)
     * Multinomial Naive Bayes (text only)
     * Transfer Learning (Keras MLP combining image embeddings + dense text vector)
 - Reports accuracy, precision, recall, f1 (macro)
 - Saves final trained sklearn models and a small Flask API to serve predictions

Requirements (add to requirements.txt):
  numpy,pandas,scikit-learn,scipy,joblib,flask,opencv-python, tensorflow, pillow, tqdm

NOTE: This script assumes you have a local dataset and (if images are large)
resizing and caching will be helpful. See `README` section at the bottom for usage
and deployment instructions (how to run locally and deploy to a hosting provider).

"""

import os
import argparse
import json
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.base import BaseEstimator, ClassifierMixin
import joblib

# Keras for image embeddings and transfer model
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.preprocessing import image as kimage
from tensorflow.keras import layers, models, optimizers


# ---------------------------- Utilities ----------------------------

def load_multimodal_csv(csv_path: str, text_col='text', image_col='image_path', label_col='label') -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    assert text_col in df.columns, f"No column {text_col}"
    assert image_col in df.columns, f"No column {image_col}"
    assert label_col in df.columns, f"No column {label_col}"
    df = df[[text_col, image_col, label_col]].dropna()
    df = df.reset_index(drop=True)
    return df


def extract_text_tfidf(texts, max_features=10000) -> Tuple[TfidfVectorizer, np.ndarray]:
    vect = TfidfVectorizer(max_features=max_features, ngram_range=(1,2), stop_words='english')
    X = vect.fit_transform(texts)
    return vect, X


def make_mobilenet_embedder(img_size=(160,160)):
    base = MobileNetV2(weights='imagenet', include_top=False, pooling='avg', input_shape=(img_size[0], img_size[1], 3))
    base.trainable = False

    def embed_batch(image_paths):
        arr = []
        for p in image_paths:
            try:
                img = kimage.load_img(p, target_size=img_size)
                x = kimage.img_to_array(img)
                x = np.expand_dims(x, axis=0)
                x = mobilenet_preprocess(x)
                arr.append(x)
            except Exception as e:
                # If image read fails, use zeros
                print(f"Warning: failed to read {p}: {e}")
                arr.append(np.zeros((1, img_size[0], img_size[1], 3), dtype=np.float32))
        batch = np.vstack(arr)
        emb = base.predict(batch, verbose=0)
        return emb

    return make_mobilenet_embedder, embed_batch if False else embed_batch


def extract_image_embeddings(image_paths, img_size=(160,160), batch_size=32):
    # Build the MobileNetV2 model once
    model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg', input_shape=(img_size[0], img_size[1], 3))
    model.trainable = False

    embeddings = []
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        batch_arr = []
        for p in batch_paths:
            try:
                img = kimage.load_img(p, target_size=img_size)
                x = kimage.img_to_array(img)
                x = np.expand_dims(x, axis=0)
                x = mobilenet_preprocess(x)
                batch_arr.append(x)
            except Exception as e:
                print(f"Warning: failed to read {p}: {e}")
                batch_arr.append(np.zeros((1, img_size[0], img_size[1], 3), dtype=np.float32))
        batch_arr = np.vstack(batch_arr)
        emb = model.predict(batch_arr, verbose=0)
        embeddings.append(emb)
    embeddings = np.vstack(embeddings)
    return embeddings


# ---------------------------- Models & CV ----------------------------

def evaluate_models(X_text, X_image, y, tfidf_vect, output_dir='models', cv_splits=5):
    os.makedirs(output_dir, exist_ok=True)

    # Prepare combined features
    from scipy.sparse import hstack
    if hasattr(X_text, 'toarray'):
        X_text_arr = X_text
    else:
        X_text_arr = X_text

    # If sparse, convert to CSR for hstack
    if hasattr(X_text_arr, 'shape') and hasattr(X_text_arr, 'tocsc'):
        # sparse
        X_combined = hstack([X_text_arr, X_image]) if hasattr(X_image, 'ndim') else hstack([X_text_arr, X_image])
    else:
        # dense
        X_combined = np.hstack([X_text_arr.toarray() if hasattr(X_text_arr, 'toarray') else X_text_arr, X_image])

    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)

    models = {
        'logistic': LogisticRegression(max_iter=2000),
        'svm': LinearSVC(max_iter=20000),
        'random_forest': RandomForestClassifier(n_estimators=200),
        # Naive Bayes will be trained on text only
        'naive_bayes': MultinomialNB()
    }

    results = {}

    def run_cv(estimator, X, y):
        accs, precs, recs, f1s = [], [], [], []
        for train_idx, test_idx in skf.split(X, y):
            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]

            # Some classifiers need dense
            if hasattr(estimator, 'fit'):
                estimator.fit(X_tr, y_tr)
                y_pred = estimator.predict(X_te)
            else:
                estimator.fit(X_tr, y_tr)
                y_pred = estimator.predict(X_te)

            accs.append(accuracy_score(y_te, y_pred))
            precs.append(precision_score(y_te, y_pred, average='macro', zero_division=0))
            recs.append(recall_score(y_te, y_pred, average='macro', zero_division=0))
            f1s.append(f1_score(y_te, y_pred, average='macro', zero_division=0))
        return np.mean(accs), np.mean(precs), np.mean(recs), np.mean(f1s)

    # Naive Bayes on text only
    print("Running 5-fold CV for MultinomialNB on text features...")
    if hasattr(X_text, 'toarray'):
        X_text_for_nb = X_text
    else:
        X_text_for_nb = X_text
    nb_metrics = run_cv(models['naive_bayes'], X_text_for_nb, y)
    results['naive_bayes'] = nb_metrics

    # Logistic, SVM, RandomForest on combined
    for name in ['logistic', 'svm', 'random_forest']:
        print(f"Running 5-fold CV for {name} on combined text+image features...")
        estimator = models[name]
        # many sklearn estimators require dense arrays
        if hasattr(X_combined, 'toarray'):
            Xc = X_combined.tocsr()
        else:
            Xc = X_combined
        metrics = run_cv(estimator, Xc, y)
        results[name] = metrics

    # Transfer learning (simple Keras MLP combining image embeddings + dense text vector)
    print("Running 5-fold CV for Transfer Learning (Keras combined model)...")
    tl_metrics = run_transfer_learning_cv(X_text, X_image, y, skf)
    results['transfer_learning'] = tl_metrics

    # Save vectorizer
    joblib.dump(tfidf_vect, os.path.join(output_dir, 'tfidf_vect.joblib'))
    # Note: saving large models like RandomForest, SVM
    # Optionally re-train on full data and save final models

    return results


def run_transfer_learning_cv(X_text, X_image, y, skf: StratifiedKFold):
    # Convert X_text sparse to dense (may be memory heavy) — project to lower-dimensional dense via TruncatedSVD if sparse
    from sklearn.decomposition import TruncatedSVD
    if hasattr(X_text, 'shape') and hasattr(X_text, 'tocsc'):
        svd = TruncatedSVD(n_components=128, random_state=42)
        X_text_dense = svd.fit_transform(X_text)
    else:
        # if already dense
        X_text_dense = X_text if isinstance(X_text, np.ndarray) else X_text.toarray()

    metrics_acc, metrics_prec, metrics_rec, metrics_f1 = [], [], [], []

    for train_idx, test_idx in skf.split(X_text_dense, y):
        X_text_tr, X_text_te = X_text_dense[train_idx], X_text_dense[test_idx]
        X_img_tr, X_img_te = X_image[train_idx], X_image[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        # Build Keras model
        # Image branch
        img_input = layers.Input(shape=(X_img_tr.shape[1],), name='img_emb')
        x1 = layers.Dense(256, activation='relu')(img_input)
        x1 = layers.Dropout(0.3)(x1)

        # Text branch
        text_input = layers.Input(shape=(X_text_tr.shape[1],), name='text_vec')
        x2 = layers.Dense(128, activation='relu')(text_input)
        x2 = layers.Dropout(0.3)(x2)

        # Combine
        x = layers.Concatenate()([x1, x2])
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        out = layers.Dense(1, activation='sigmoid')(x)

        model = models.Model([img_input, text_input], out)
        model.compile(optimizer=optimizers.Adam(1e-4), loss='binary_crossentropy', metrics=['accuracy'])

        # Convert labels to 0/1
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y_tr_enc = le.fit_transform(y_tr)
        y_te_enc = le.transform(y_te)

        model.fit([X_img_tr, X_text_tr], y_tr_enc, epochs=8, batch_size=32, verbose=0)
        preds = model.predict([X_img_te, X_text_te]).ravel()
        preds_bin = (preds >= 0.5).astype(int)

        metrics_acc.append(accuracy_score(y_te_enc, preds_bin))
        metrics_prec.append(precision_score(y_te_enc, preds_bin, average='binary', zero_division=0))
        metrics_rec.append(recall_score(y_te_enc, preds_bin, average='binary', zero_division=0))
        metrics_f1.append(f1_score(y_te_enc, preds_bin, average='binary', zero_division=0))

        # Free memory
        tf.keras.backend.clear_session()

    return (np.mean(metrics_acc), np.mean(metrics_prec), np.mean(metrics_rec), np.mean(metrics_f1))


# ---------------------------- Train & Save final models ----------------------------

def train_and_save_final(X_text, X_image, y, tfidf_vect, output_dir='models'):
    os.makedirs(output_dir, exist_ok=True)

    # Convert combined
    from scipy.sparse import hstack
    X_combined = hstack([X_text, X_image]) if hasattr(X_text, 'tocsc') else np.hstack([X_text, X_image])

    # Train logistic
    log = LogisticRegression(max_iter=2000)
    log.fit(X_combined, y)
    joblib.dump(log, os.path.join(output_dir, 'logistic.joblib'))

    # Train SVM
    svm = LinearSVC(max_iter=20000)
    svm.fit(X_combined, y)
    joblib.dump(svm, os.path.join(output_dir, 'svm.joblib'))

    # Random Forest
    rf = RandomForestClassifier(n_estimators=200)
    rf.fit(X_combined, y)
    joblib.dump(rf, os.path.join(output_dir, 'random_forest.joblib'))

    # Naive Bayes on text
    nb = MultinomialNB()
    nb.fit(X_text, y)
    joblib.dump(nb, os.path.join(output_dir, 'naive_bayes.joblib'))

    print("Saved models to", output_dir)


# ---------------------------- Simple Flask API ----------------------------

FLASK_APP_TEMPLATE = r"""
from flask import Flask, request, jsonify
import joblib
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.preprocessing import image as kimage
from tensorflow.keras.applications import MobileNetV2

app = Flask(__name__)

# Load models (assumes these files exist in ./models)
tfidf = joblib.load('models/tfidf_vect.joblib')
log = joblib.load('models/logistic.joblib')
nb = joblib.load('models/naive_bayes.joblib')

# MobileNet for embeddings
img_model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg', input_shape=(160,160,3))

def image_to_emb(path):
    img = kimage.load_img(path, target_size=(160,160))
    x = kimage.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = mobilenet_preprocess(x)
    emb = img_model.predict(x)
    return emb

@app.route('/api/analyze', methods=['POST'])
def analyze():
    data = request.json
    text = data.get('text', '')
    image_path = data.get('image_path', None)

    text_vec = tfidf.transform([text])
    nb_pred = nb.predict(text_vec)[0]

    if image_path:
        emb = image_to_emb(image_path)
        # Combine (text dense + emb)
        try:
            import scipy.sparse as sp
            text_dense = text_vec.toarray()
        except:
            text_dense = text_vec
        X = np.hstack([text_dense, emb])
        log_pred = log.predict(X)[0]
    else:
        log_pred = None

    return jsonify({'naive_bayes': str(nb_pred), 'logistic': str(log_pred)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
"""


# ---------------------------- CLI ----------------------------

def main():
    parser = argparse.ArgumentParser(description='Train multimodal fake-news models with 5-fold CV')
    parser.add_argument('--csv', type=str, required=True, help='Path to CSV with columns text,image_path,label')
    parser.add_argument('--out', type=str, default='models', help='Output directory to save models')
    parser.add_argument('--max_text_features', type=int, default=5000)
    args = parser.parse_args()

    df = load_multimodal_csv(args.csv)
    texts = df['text'].astype(str).tolist()
    img_paths = df['image_path'].astype(str).tolist()
    labels = df['label'].astype(str).values

    print('Extracting TF-IDF text features...')
    tfidf, X_text = extract_text_tfidf(texts, max_features=args.max_text_features)

    print('Extracting image embeddings (MobileNetV2)...')
    X_image = extract_image_embeddings(img_paths, img_size=(160,160), batch_size=32)

    print('Running cross-validation for models...')
    results = evaluate_models(X_text, X_image, labels, tfidf, output_dir=args.out, cv_splits=5)

    print('\nCV Results (accuracy, precision_macro, recall_macro, f1_macro):')
    for k,v in results.items():
        print(f"{k}: {v}")

    print('\nTraining final models on full dataset and saving...')
    train_and_save_final(X_text, X_image, labels, tfidf, output_dir=args.out)

    # Save small Flask app
    with open(os.path.join(args.out, 'serve_app.py'), 'w') as f:
        f.write(FLASK_APP_TEMPLATE)

    print(f"Wrote a basic Flask serving app to {os.path.join(args.out, 'serve_app.py')}.\nStart it with: python {os.path.join(args.out, 'serve_app.py')}")


if __name__ == '__main__':
    main()


# ---------------------------- README / Deployment Notes ----------------------------
"""README (usage & deployment)

1) Dataset format
   - CSV with columns: text,image_path,label
   - image_path may be local filesystem paths relative to where you run the script
   - label should be binary (e.g. 'fake' or 'real') — script encodes labels automatically

2) Run training & CV locally
   python backend_fake_news_multimodal.py --csv path/to/dataset.csv --out models

3) After running, models/ will contain saved sklearn models, tfidf vectorizer, and serve_app.py
   Start the API locally:
     cd models
     python serve_app.py
   The API listens on http://0.0.0.0:5000/api/analyze and accepts JSON {"text":"...", "image_path":"/full/path/to/img.jpg"}

4) Deploying to a host (Render / Railway / Heroku)
   - Create a requirements.txt listing needed packages
   - Add a Procfile: web: python models/serve_app.py
   - Push to GitHub and connect the repo to Render or Railway (they will provide a public URL)

5) Notes & improvements
   - For production, store images in cloud storage (S3) and fetch them in the server
   - Add proper error handling, authentication, batching
   - For Naive Bayes on text only: keep the pipeline separate
   - Consider fine-tuning a transformer (HuggingFace) for text instead of TF-IDF

"""
