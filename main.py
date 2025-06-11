import json
import io
import time
import numpy as np
import torch
import torch.nn as nn
from flask import Flask, request, jsonify
import mysql.connector
import os
import sys
import traceback

# -----------------------------------------
# Configuration
# -----------------------------------------
DB_USER        = os.environ.get("DB_USER", "appuser")
DB_PASS        = os.environ.get("DB_PASS", "secure_app_password")
DB_NAME        = os.environ.get("DB_NAME", "myappdb")
DB_SOCKET      = os.environ.get("DB_SOCKET")            # e.g. "/cloudsql/project:region:instance"
CLICK_MODEL_PATH = os.environ.get("CLICK_MODEL_PATH", "/app/click_model.pt")
EMB_DIM        = int(os.environ.get("EMB_DIM", 128))
NEG_RATIO      = int(os.environ.get("NEG_RATIO", 3))
BATCH_SIZE     = int(os.environ.get("BATCH_SIZE", 64))
NUM_EPOCHS     = int(os.environ.get("NUM_EPOCHS", 5))
LEARNING_RATE  = float(os.environ.get("LEARNING_RATE", 1e-3))
GCS_BUCKET     = os.environ.get("GCS_BUCKET")            # Optional: Google Cloud Storage bucket
GCS_MODEL_PATH = os.environ.get("GCS_MODEL_PATH", "models/click_model.pt")

# Device 설정 (GPU 우선)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

# -----------------------------------------
# Database helper
# -----------------------------------------
def get_db_connection():
    try:
        if DB_SOCKET:
            return mysql.connector.connect(
                user=DB_USER,
                password=DB_PASS,
                database=DB_NAME,
                unix_socket=DB_SOCKET
            )
        return mysql.connector.connect(
            user=DB_USER,
            password=DB_PASS,
            database=DB_NAME,
            host="127.0.0.1",
            port=3306
        )
    except mysql.connector.Error:
        traceback.print_exc(file=sys.stderr)
        raise


def load_ndarray(blob: bytes) -> np.ndarray:
    if not blob:
        return None
    buf = io.BytesIO(blob)
    return np.load(buf, allow_pickle=False)

# -----------------------------------------
# Model Definition
# -----------------------------------------
class ClickMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128]
        layers = []
        dims = [input_dim] + hidden_dims
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(dims[-1], 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return torch.sigmoid(self.net(x)).squeeze(-1)

# -----------------------------------------
# GCS Helpers (optional)
# -----------------------------------------
from google.cloud import storage

def upload_to_gcs(local_path, bucket_name, blob_path):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    blob.upload_from_filename(local_path)


def download_from_gcs(local_path, bucket_name, blob_path):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    blob.download_to_filename(local_path)

# -----------------------------------------
# 1) Train & Save Model
# -----------------------------------------
def train_and_save_model():
    conn = get_db_connection()
    cursor = conn.cursor()

    # load embeddings
    cursor.execute("SELECT id, gcn_vec FROM users WHERE gcn_vec IS NOT NULL")
    users = {uid: load_ndarray(blob) for uid, blob in cursor.fetchall()}
    cursor.execute("SELECT id, gcn_vec FROM issues WHERE gcn_vec IS NOT NULL")
    issues = {iid: load_ndarray(blob) for iid, blob in cursor.fetchall()}

    # load positive clicks
    cursor.execute("SELECT user_id, issue_id FROM custom_events WHERE eventname='click'")
    positives = set(cursor.fetchall())
    conn.close()

    # prepare samples
    X, y = [], []
    for uid, iid in positives:
        uvec, ivec = users.get(uid), issues.get(iid)
        if uvec is None or ivec is None: continue
        X.append(np.concatenate([uvec, ivec]))
        y.append(1)
        for _ in range(NEG_RATIO):
            neg_iid = np.random.choice(list(issues.keys()))
            if (uid, neg_iid) in positives: continue
            nvec = issues[neg_iid]
            X.append(np.concatenate([uvec, nvec]))
            y.append(0)

    X = torch.tensor(np.stack(X), dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    from torch.utils.data import DataLoader, TensorDataset
    loader = DataLoader(TensorDataset(X, y), batch_size=BATCH_SIZE, shuffle=True)

    # training
    model = ClickMLP(input_dim=2*EMB_DIM).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    model.train()
    for epoch in range(NUM_EPOCHS):
        total_loss = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb)
            loss = criterion(preds, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} - loss: {total_loss/len(loader):.4f}")

    # save model
    torch.save(model.state_dict(), CLICK_MODEL_PATH)
    print(f"Model saved to {CLICK_MODEL_PATH}")
    if GCS_BUCKET:
        upload_to_gcs(CLICK_MODEL_PATH, GCS_BUCKET, GCS_MODEL_PATH)
        print(f"Model uploaded to gs://{GCS_BUCKET}/{GCS_MODEL_PATH}")

# -----------------------------------------
# Initialize inference model
# -----------------------------------------
model = ClickMLP(input_dim=2*EMB_DIM).to(device)
if GCS_BUCKET:
    download_from_gcs(CLICK_MODEL_PATH, GCS_BUCKET, GCS_MODEL_PATH)
checkpoint = torch.load(CLICK_MODEL_PATH, map_location=device)
model.load_state_dict(checkpoint)
model.eval()

# -----------------------------------------
# Flask app
# -----------------------------------------
app = Flask(__name__)

@app.route('/set-recommand', methods=['POST'])
def set_recommendations():
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT id, gcn_vec FROM users WHERE gcn_vec IS NOT NULL")
    users = cursor.fetchall()
    cursor.execute("SELECT id, gcn_vec FROM issues WHERE gcn_vec IS NOT NULL")
    issues = cursor.fetchall()

    user_vecs  = {uid: load_ndarray(blob) for uid, blob in users}
    issue_vecs = {iid: load_ndarray(blob) for iid, blob in issues}

    for user_id, uvec in user_vecs.items():
        feats, iids = [], []
        for iid, ivec in issue_vecs.items():
            feats.append(np.concatenate([uvec, ivec], axis=0))
            iids.append(iid)
        if not feats:
            continue
        X = torch.from_numpy(np.stack(feats)).float().to(device)
        with torch.no_grad():
            probs = model(X)
        probs = probs.cpu().numpy()

        top = np.argsort(probs)[::-1][:100]
        rec_json = json.dumps([iids[i] for i in top], ensure_ascii=False)

        cursor.execute(
            "UPDATE users SET recommendations = %s WHERE id = %s",
            (rec_json, user_id)
        )

    version = int(time.time())
    cursor.execute(
        "INSERT INTO kv_int_store (`key`,`value`) VALUES (%s,%s) ON DUPLICATE KEY UPDATE `value`=VALUES(`value`)",
        ("recommendation_version", version)
    )

    conn.commit()
    cursor.close()
    conn.close()

    return jsonify({"message": "user recommendations updated (GPU click-MLP)", "version": version}), 200

@app.route('/train-model', methods=['POST'])
def train_model_route():
    try:
        train_and_save_model()
        return jsonify({"message": "model training triggered"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
