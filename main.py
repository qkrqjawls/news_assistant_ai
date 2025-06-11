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
DB_USER         = os.environ.get("DB_USER", "appuser")
DB_PASS         = os.environ.get("DB_PASS", "secure_app_password")
DB_NAME         = os.environ.get("DB_NAME", "myappdb")
DB_SOCKET       = os.environ.get("DB_SOCKET")            # e.g. "/cloudsql/project:region:instance"
CLICK_MODEL_PATH= os.environ.get("CLICK_MODEL_PATH", "/app/click_model.pt")
EMB_DIM         = int(os.environ.get("EMB_DIM", 128))      # GCN embedding dim
SENT_DIM        = int(os.environ.get("SENT_DIM", 384))     # sentence_embedding dim
CAT_DIM_USER    = int(os.environ.get("CAT_DIM_USER", 12))  # user category_vec dim
CAT_DIM_ISSUE   = int(os.environ.get("CAT_DIM_ISSUE", 12)) # issue category_vec dim
NEG_RATIO       = int(os.environ.get("NEG_RATIO", 3))
BATCH_SIZE      = int(os.environ.get("BATCH_SIZE", 64))
NUM_EPOCHS      = int(os.environ.get("NUM_EPOCHS", 5))
LEARNING_RATE   = float(os.environ.get("LEARNING_RATE", 1e-3))
GCS_BUCKET      = os.environ.get("GCS_BUCKET")             # Optional: GCS bucket
GCS_MODEL_PATH  = os.environ.get("GCS_MODEL_PATH", "models/click_model.pt")

# Device 설정 (GPU 우선)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

# -----------------------------------------
# Database helper
# -----------------------------------------
def get_db_connection():
    try:
        if DB_SOCKET:
            return mysql.connector.connect(user=DB_USER, password=DB_PASS,
                                           database=DB_NAME, unix_socket=DB_SOCKET)
        return mysql.connector.connect(user=DB_USER, password=DB_PASS,
                                       database=DB_NAME, host="127.0.0.1", port=3306)
    except mysql.connector.Error:
        traceback.print_exc(file=sys.stderr)
        raise


def load_ndarray(blob: bytes) -> np.ndarray:
    if not blob:
        return None
    buf = io.BytesIO(blob)
    return np.load(buf, allow_pickle=False)

# -----------------------------------------
# Model definition
# -----------------------------------------
class ClickMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128]
        layers = []
        dims = [input_dim] + hidden_dims
        for in_dim, out_dim in zip(dims, dims[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(dims[-1], 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return torch.sigmoid(self.net(x)).squeeze(-1)

# -----------------------------------------
# GCS helpers (optional)
# -----------------------------------------
from google.cloud import storage

def upload_to_gcs(local_path, bucket, blob_path):
    client = storage.Client()
    bucket = client.bucket(bucket)
    bucket.blob(blob_path).upload_from_filename(local_path)

def download_from_gcs(local_path, bucket, blob_path):
    client = storage.Client()
    client.bucket(bucket).blob(blob_path).download_to_filename(local_path)

# -----------------------------------------
# Train & Save Model
# -----------------------------------------
def train_and_save_model():
    conn = get_db_connection()
    cursor = conn.cursor()
    # load users (gcn, category)
    cursor.execute("SELECT id, gcn_vec, category_vec FROM users")
    users = {uid: (load_ndarray(g), load_ndarray(c)) for uid, g, c in cursor.fetchall()}
    # load issues (gcn, sentence, category)
    cursor.execute("SELECT id, gcn_vec, sentence_embedding, category_vec FROM issues")
    issues = {iid: (load_ndarray(g), load_ndarray(s), load_ndarray(c)) for iid, g, s, c in cursor.fetchall()}
    # positive clicks
    cursor.execute("SELECT user_id, issue_id FROM custom_events WHERE eventname='click'")
    positives = set(cursor.fetchall())
    conn.close()

    # prepare training samples
    X_list, y_list = [], []
    for uid, iid in positives:
        u_g, u_c = users.get(uid, (None, None))
        i_g, i_s, i_c = issues.get(iid, (None, None, None))
        if u_c is None or i_s is None:
            continue
        u_g = u_g if u_g is not None else np.zeros(EMB_DIM)
        i_g = i_g if i_g is not None else np.zeros(EMB_DIM)
        i_c = i_c if i_c is not None else np.zeros(CAT_DIM_ISSUE)
        feat = np.concatenate([u_g, u_c, i_g, i_s, i_c], axis=0)
        X_list.append(feat); y_list.append(1)
        for _ in range(NEG_RATIO):
            neg_iid = np.random.choice(list(issues.keys()))
            if (uid, neg_iid) in positives: continue
            ng_g, ng_s, ng_c = issues[neg_iid]
            ng_g = ng_g if ng_g is not None else np.zeros(EMB_DIM)
            ng_c = ng_c if ng_c is not None else np.zeros(CAT_DIM_ISSUE)
            neg_feat = np.concatenate([u_g, u_c, ng_g, ng_s, ng_c], axis=0)
            X_list.append(neg_feat); y_list.append(0)

    X = torch.tensor(np.stack(X_list), dtype=torch.float32)
    y = torch.tensor(y_list, dtype=torch.float32)
    from torch.utils.data import DataLoader, TensorDataset
    loader = DataLoader(TensorDataset(X, y), batch_size=BATCH_SIZE, shuffle=True)

    # train
    input_dim = 2*EMB_DIM + CAT_DIM_USER + SENT_DIM + CAT_DIM_ISSUE
    model = ClickMLP(input_dim=input_dim).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    model.train()
    for epoch in range(NUM_EPOCHS):
        total_loss = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb)
            loss = criterion(preds, yb)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} loss={total_loss/len(loader):.4f}")

    # save model
    torch.save(model.state_dict(), CLICK_MODEL_PATH)
    print(f"Model saved to {CLICK_MODEL_PATH}")
    if GCS_BUCKET:
        upload_to_gcs(CLICK_MODEL_PATH, GCS_BUCKET, GCS_MODEL_PATH)
        print(f"Uploaded to gs://{GCS_BUCKET}/{GCS_MODEL_PATH}")

# -----------------------------------------
# Initialize inference model
# -----------------------------------------
if GCS_BUCKET:
    download_from_gcs(CLICK_MODEL_PATH, GCS_BUCKET, GCS_MODEL_PATH)
input_dim = 2*EMB_DIM + CAT_DIM_USER + SENT_DIM + CAT_DIM_ISSUE
model = ClickMLP(input_dim=input_dim).to(device)
model.load_state_dict(torch.load(CLICK_MODEL_PATH, map_location=device))
model.eval()

# -----------------------------------------
# Flask app
# -----------------------------------------
app = Flask(__name__)

@app.route('/train-model', methods=['POST'])
def train_route():
    try:
        train_and_save_model()
        return jsonify({"message": "training complete"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/set-recommand', methods=['POST'])
def recommend_route():
    conn = get_db_connection(); cursor = conn.cursor()
    # load user & issue data
    cursor.execute("SELECT id, gcn_vec, category_vec FROM users")
    users = {uid: (load_ndarray(g), load_ndarray(c)) for uid, g, c in cursor.fetchall()}
    cursor.execute("SELECT id, gcn_vec, sentence_embedding, category_vec FROM issues")
    issues = {iid: (load_ndarray(g), load_ndarray(s), load_ndarray(c))
              for iid, g, s, c in cursor.fetchall()}

    # clear existing recommendations
    cursor.execute("DELETE FROM user_recommendations")

    # compute and batch insert
    recs = []
    for uid, (u_g, u_c) in users.items():
        if u_c is None: continue
        feats, iids = [], []
        for iid, (i_g, i_s, i_c) in issues.items():
            if i_s is None: continue
            u_gv = u_g if u_g is not None else np.zeros(EMB_DIM)
            i_gv = i_g if i_g is not None else np.zeros(EMB_DIM)
            i_cv = i_c if i_c is not None else np.zeros(CAT_DIM_ISSUE)
            feat = np.concatenate([u_gv, u_c, i_gv, i_s, i_cv], axis=0)
            feats.append(feat); iids.append(iid)
        if not feats: continue
        X = torch.tensor(np.stack(feats), dtype=torch.float32).to(device)
        with torch.no_grad(): probs = model(X).cpu().numpy()
        for iid, score in zip(iids, probs):
            recs.append((uid, iid, float(score)))

    # bulk insert recommendations
    if recs:
        cursor.executemany(
            "INSERT INTO user_recommendations (user_id, issue_id, score)"
            " VALUES (%s, %s, %s)"
            " ON DUPLICATE KEY UPDATE score = VALUES(score), updated_at = CURRENT_TIMESTAMP",
            recs
        )

    conn.commit(); cursor.close(); conn.close()
    return jsonify({"message": "recommendations updated"}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))

from GCN_embedding import user_item_GCN_embedding

def arr_to_blob(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, arr)
    return buf.getvalue()


def load_ndarray(blob: bytes) -> np.ndarray:
    if not blob:
        return None
    buf = io.BytesIO(blob)
    return np.load(buf, allow_pickle=False)

@app.route('/gcn-embedd', methods=['POST'])
def set_gcn_embedding():
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT id FROM users")
    rows = cursor.fetchall()
    user_ids = [row[0] for row in rows]

    cursor.execute("SELECT id FROM issues")
    rows = cursor.fetchall()
    issue_ids = [row[0] for row in rows]

    cursor.execute("SELECT user_id, issue_id FROM custom_events WHERE eventname = %s", ("click",))
    rows = cursor.fetchall()

    raw_edges = rows

    user_vec, issue_vec = user_item_GCN_embedding(user_ids=user_ids, item_ids=issue_ids, raw_edges=raw_edges)

    user_data = [(arr_to_blob(user_vec[uid]), uid) for uid in user_ids]
    cursor.executemany("UPDATE users SET gcn_vec=%s WHERE id=%s", user_data)

    issue_data = [(arr_to_blob(issue_vec[iid]), iid) for iid in issue_ids]
    cursor.executemany("UPDATE issues SET gcn_vec=%s WHERE id=%s", issue_data)

    conn.commit()
    
    cursor.close()
    conn.close()

    return jsonify({
        "message" : "everything ok my man"
    }), 200