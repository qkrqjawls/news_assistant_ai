import json
import io
import time
import os
import sys
import traceback

import numpy as np
import torch
import torch.nn as nn
from flask import Flask, request, jsonify
import mysql.connector
# Optional GCS client
from google.cloud import storage

# -----------------------------------------
# Configuration
# -----------------------------------------
DB_USER          = os.environ.get("DB_USER", "appuser")
DB_PASS          = os.environ.get("DB_PASS", "secure_app_password")
DB_NAME          = os.environ.get("DB_NAME", "myappdb")
DB_SOCKET        = os.environ.get("DB_SOCKET")
CLICK_MODEL_PATH = os.environ.get("CLICK_MODEL_PATH", "/app/click_model.pt")
EMB_DIM          = int(os.environ.get("EMB_DIM", 128))
SENT_DIM         = int(os.environ.get("SENT_DIM", 384))
CAT_DIM_USER     = int(os.environ.get("CAT_DIM_USER", 32))
CAT_DIM_ISSUE    = int(os.environ.get("CAT_DIM_ISSUE", 32))
NEG_RATIO        = int(os.environ.get("NEG_RATIO", 3))
BATCH_SIZE       = int(os.environ.get("BATCH_SIZE", 64))
NUM_EPOCHS       = int(os.environ.get("NUM_EPOCHS", 5))
LEARNING_RATE    = float(os.environ.get("LEARNING_RATE", 1e-3))
GCS_BUCKET       = os.environ.get("GCS_BUCKET")
GCS_MODEL_PATH   = os.environ.get("GCS_MODEL_PATH", "models/click_model.pt")

# Device 설정 (GPU 우선)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

# -----------------------------------------
# DB helper
# -----------------------------------------
def get_db_connection():
    try:
        if DB_SOCKET:
            return mysql.connector.connect(
                user=DB_USER, password=DB_PASS,
                database=DB_NAME, unix_socket=DB_SOCKET)
        return mysql.connector.connect(
            user=DB_USER, password=DB_PASS,
            database=DB_NAME, host="127.0.0.1", port=3306)
    except mysql.connector.Error:
        traceback.print_exc(file=sys.stderr)
        raise

def load_ndarray(blob: bytes):
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
        hidden_dims = hidden_dims or [256, 128]
        layers = []
        dims = [input_dim] + hidden_dims
        for in_dim, out_dim in zip(dims, dims[1:]):
            layers += [nn.Linear(in_dim, out_dim), nn.ReLU()]
        layers.append(nn.Linear(dims[-1], 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return torch.sigmoid(self.net(x)).squeeze(-1)

# -----------------------------------------
# GCS helpers
# -----------------------------------------

def upload_to_gcs(local_path, bucket_name, blob_path):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    bucket.blob(blob_path).upload_from_filename(local_path)

def download_from_gcs(local_path, bucket_name, blob_path):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    bucket.blob(blob_path).download_to_filename(local_path)

# -----------------------------------------
# Model load utility
# -----------------------------------------
model = None

def load_model():
    global model
    # skip if already loaded
    if model is not None:
        return True
    # download if GCS configured
    if GCS_BUCKET:
        try:
            download_from_gcs(CLICK_MODEL_PATH, GCS_BUCKET, GCS_MODEL_PATH)
        except Exception as e:
            print(f"[WARN] Model download failed: {e}")
    # load if exists
    if os.path.exists(CLICK_MODEL_PATH):
        try:
            input_dim = 2*EMB_DIM + CAT_DIM_USER + SENT_DIM + CAT_DIM_ISSUE
            m = ClickMLP(input_dim=input_dim).to(device)
            m.load_state_dict(torch.load(CLICK_MODEL_PATH, map_location=device))
            m.eval()
            model = m
            print("[INFO] Model loaded.")
            return True
        except Exception as e:
            print(f"[ERROR] Model load failed: {e}")
    else:
        print(f"[INFO] No model file at {CLICK_MODEL_PATH}")
    return False

# -----------------------------------------
# Train & save model
# -----------------------------------------
def train_and_save_model():
    # prepare data
    conn = get_db_connection(); cur = conn.cursor()
    cur.execute("SELECT id, gcn_vec, category_vec FROM users")
    users = {uid: (load_ndarray(g), load_ndarray(c)) for uid,g,c in cur.fetchall()}
    cur.execute("SELECT id, gcn_vec, sentence_embedding, category_vec FROM issues")
    issues = {iid: (load_ndarray(g), load_ndarray(s), load_ndarray(c)) for iid,g,s,c in cur.fetchall()}
    cur.execute("SELECT user_id, issue_id FROM custom_events WHERE eventname='click'")
    positives = set(cur.fetchall()); conn.close()
    X_list, y_list = [], []
    for uid,iid in positives:
        u_g, u_c = users.get(uid,(None,None))
        i_g, i_s, i_c = issues.get(iid,(None,None,None))
        if u_c is None or i_s is None: continue
        u_g = u_g if u_g is not None else np.zeros(EMB_DIM)
        i_g = i_g if i_g is not None else np.zeros(EMB_DIM)
        i_c = i_c if i_c is not None else np.zeros(CAT_DIM_ISSUE)
        X_list.append(np.concatenate([u_g,u_c,i_g,i_s,i_c])); y_list.append(1)
        for _ in range(NEG_RATIO):
            nid = np.random.choice(list(issues));
            if (uid,nid) in positives: continue
            ng,ns,nc = issues[nid]
            ng = ng if ng is not None else np.zeros(EMB_DIM)
            nc = nc if nc is not None else np.zeros(CAT_DIM_ISSUE)
            X_list.append(np.concatenate([u_g,u_c,ng,ns,nc])); y_list.append(0)
    X = torch.tensor(np.stack(X_list),dtype=torch.float32); y=torch.tensor(y_list,dtype=torch.float32)
    from torch.utils.data import DataLoader, TensorDataset
    loader = DataLoader(TensorDataset(X,y), batch_size=BATCH_SIZE, shuffle=True)
    input_dim=2*EMB_DIM+CAT_DIM_USER+SENT_DIM+CAT_DIM_ISSUE
    m = ClickMLP(input_dim).to(device)
    opt=torch.optim.Adam(m.parameters(),lr=LEARNING_RATE); loss_fn=nn.BCELoss()
    m.train()
    for e in range(NUM_EPOCHS):
        tot=0
        for xb,yb in loader:
            xb,yb=xb.to(device),yb.to(device)
            p=m(xb); l=loss_fn(p,yb)
            opt.zero_grad();l.backward();opt.step();tot+=l.item()
        print(f"Epoch{e+1}/{NUM_EPOCHS} loss={tot/len(loader):.4f}")
    torch.save(m.state_dict(),CLICK_MODEL_PATH);
    print("Model saved")
    if GCS_BUCKET: upload_to_gcs(CLICK_MODEL_PATH,GCS_BUCKET,GCS_MODEL_PATH)
    # reload into memory
    load_model()

# -----------------------------------------
# Flask app
# -----------------------------------------
app=Flask(__name__)

@app.route('/train-model',methods=['POST'])
def train_route():
    try:
        train_and_save_model()
        return jsonify({"message":"training complete"}),200
    except Exception as e:
        return jsonify({"error":str(e)}),500

@app.route('/set-recommand',methods=['POST'])
def recommend_route():
    if not load_model():
        return jsonify({"error":"No model available"}),503
    conn=get_db_connection();cur=conn.cursor()
    cur.execute("DELETE FROM user_recommendations")
    cur.execute("SELECT id,gcn_vec,category_vec FROM users"); users={uid:(load_ndarray(g),load_ndarray(c)) for uid,g,c in cur.fetchall()}
    cur.execute("SELECT id,gcn_vec,sentence_embedding,category_vec FROM issues"); issues={iid:(load_ndarray(g),load_ndarray(s),load_ndarray(c)) for iid,g,s,c in cur.fetchall()}
    recs=[]
    for uid,(ug,uc) in users.items():
        if uc is None: continue
        feats,iids=[],[]
        for iid,(ig,is_,ic) in issues.items():
            if is_ is None: continue
            ugv=ug if ug is not None else np.zeros(EMB_DIM)
            igv=ig if ig is not None else np.zeros(EMB_DIM)
            icv=ic if ic is not None else np.zeros(CAT_DIM_ISSUE)
            feats.append(np.concatenate([ugv,uc,igv,is_,icv])); iids.append(iid)
        if not feats: continue
        X=torch.tensor(np.stack(feats),dtype=torch.float32).to(device)
        with torch.no_grad(): ps=model(X).cpu().numpy()
        for iid,sc in zip(iids,ps): recs.append((uid,iid,float(sc)))
    if recs:
        cur.executemany("INSERT INTO user_recommendations(user_id,issue_id,score) VALUES(%s,%s,%s) "
                        "ON DUPLICATE KEY UPDATE score=VALUES(score),updated_at=CURRENT_TIMESTAMP",recs)
    conn.commit();cur.close();conn.close()
    return jsonify({"message":"recommendations updated"}),200

from GCN_embedding import user_item_GCN_embedding

def arr_to_blob(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, arr)
    return buf.getvalue()

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