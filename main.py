import json
import io
import time
import os
import sys
import traceback
from datetime import timezone
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
DECAY_RATE   = float(os.environ.get("RECENCY_DECAY", 1e-6)) # 점수 7일 당 2배로 감소
BOOST_FACTOR = float(os.environ.get("RECENCY_BOOST", 1.0))

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
        input_dim = (
            2*EMB_DIM
            + (CAT_DIM_USER + 1)
            + (SENT_DIM + CAT_DIM_ISSUE + 2)
        )
        try:
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
    now_ts = time.time()  # 훈련 시점 타임스탬프

    # DB에서 users, issues, events 로드
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT id, gcn_vec, category_vec FROM users")
    users = {}
    for uid, g_blob, c_blob in cur.fetchall():
        g_arr = load_ndarray(g_blob)
        g = g_arr if g_arr is not None else np.zeros(EMB_DIM)
        c_arr = load_ndarray(c_blob)
        c = c_arr if c_arr is not None else np.zeros(CAT_DIM_USER)
        # 유저 feature: [gcn_vec, category_vec..., now_ts]
        users[uid] = (g, np.concatenate([c, [now_ts]]))

    cur.execute("""
        SELECT id, gcn_vec, sentence_embedding, category_vec, `date`, related_news_list
        FROM issues
    """)
    issues = {}
    for iid, g_blob, s_blob, c_blob, date_obj, rel_str in cur.fetchall():
        g_arr = load_ndarray(g_blob)
        g = g_arr if g_arr is not None else np.zeros(EMB_DIM)
        s_arr = load_ndarray(s_blob)
        s = s_arr if s_arr is not None else np.zeros(SENT_DIM)
        c_arr = load_ndarray(c_blob)
        c = c_arr if c_arr is not None else np.zeros(CAT_DIM_ISSUE)

        # 추가 feature 계산
        date_ts = date_obj.replace(tzinfo=timezone.utc).timestamp()
        related_count = len(rel_str.split())
        # 이슈 feature: [gcn_vec, sentence_emb..., category_vec..., date_ts, related_count]
        issues[iid] = (
            g,
            np.concatenate([s, c, [date_ts, related_count]])
        )

    cur.execute("SELECT user_id, issue_id FROM custom_events WHERE eventname='click'")
    positives = set(cur.fetchall())
    conn.close()

    # 데이터셋 생성
    X_list, y_list = [], []
    for uid, iid in positives:
        u_g, u_feat = users.get(uid, (np.zeros(EMB_DIM), np.zeros(CAT_DIM_USER+1)))
        i_g, i_feat = issues.get(iid, (np.zeros(EMB_DIM), np.zeros(SENT_DIM+CAT_DIM_ISSUE+2)))
        # positive sample
        X_list.append(np.concatenate([u_g, u_feat, i_g, i_feat])); y_list.append(1)
        # negative sampling
        for _ in range(NEG_RATIO):
            nid = np.random.choice(list(issues))
            if (uid, nid) in positives: continue
            ng, nfeat = issues[nid]
            X_list.append(np.concatenate([u_g, u_feat, ng, nfeat])); y_list.append(0)

    X = torch.tensor(np.stack(X_list), dtype=torch.float32)
    y = torch.tensor(y_list, dtype=torch.float32)

    # 모델 정의 (input_dim 은 load_model 과 동일)
    input_dim = 2*EMB_DIM + (CAT_DIM_USER + 1) + (SENT_DIM + CAT_DIM_ISSUE + 2)
    

    from torch.utils.data import DataLoader, TensorDataset
    loader = DataLoader(TensorDataset(X,y), batch_size=BATCH_SIZE, shuffle=True)
    m = ClickMLP(input_dim).to(device)
    opt=torch.optim.Adam(m.parameters(),lr=LEARNING_RATE); loss_fn=nn.BCELoss()

    # 학습 루프
    
    m.train()
    for e in range(NUM_EPOCHS):
        tot=0
        for xb,yb in loader:
            xb,yb=xb.to(device),yb.to(device)
            p=m(xb); l=loss_fn(p,yb)
            opt.zero_grad();l.backward();opt.step();tot+=l.item()
        print(f"Epoch{e+1}/{NUM_EPOCHS} loss={tot/len(loader):.4f}")
    torch.save(m.state_dict(),CLICK_MODEL_PATH)
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

@app.route('/set-recommand', methods=['POST'])
def recommend_route():
    if not load_model():
        return jsonify({"error":"No model available"}),503

    conn = get_db_connection()
    cur  = conn.cursor()

    # 1) 이슈별 날짜 로드
    cur.execute("SELECT id, `date` FROM issues")
    date_map = {iid: date_obj.timestamp() for iid, date_obj in cur.fetchall()}
    now_ts = time.time()

    # 2) 유저/이슈 피처 로드
    cur.execute("SELECT id, gcn_vec, category_vec FROM users")
    users = {uid:(load_ndarray(g), load_ndarray(c)) for uid,g,c in cur.fetchall()}

    cur.execute("SELECT id, gcn_vec, sentence_embedding, category_vec FROM issues")
    issues = {iid:(load_ndarray(g), load_ndarray(s), load_ndarray(c))
              for iid,g,s,c in cur.fetchall()}

    # 3) 모델 추론 + recency weight 적용
    recs = []
    for uid, (ug, uc) in users.items():
        if uc is None: 
            continue

        feats, iids = [], []
        for iid, (ig, is_, ic) in issues.items():
            if is_ is None:
                continue

            # 명시적 None 체크로 결측 처리
            ugv = ug  if ug  is not None else np.zeros(EMB_DIM)
            igv = ig  if ig  is not None else np.zeros(EMB_DIM)
            icv = ic  if ic  is not None else np.zeros(CAT_DIM_ISSUE)
            # note: 'uc'와 'is_'는 이미 ndarray 여야 함
            feats.append(np.concatenate([ugv, uc, igv, is_, icv]))
            iids.append(iid)

        if not feats:
            continue

        X = torch.tensor(np.stack(feats), dtype=torch.float32).to(device)
        with torch.no_grad():
            base_scores = model(X).cpu().numpy()

        # recency 가중치
        for iid, base_sc in zip(iids, base_scores):
            age = now_ts - date_map.get(iid, now_ts)
            recency_w = np.exp(-DECAY_RATE * age)
            final_sc = base_sc * (1 + BOOST_FACTOR * recency_w)
            recs.append((uid, iid, float(final_sc)))

    # 4) DB 업데이트
    if recs:
        cur.executemany(
            "INSERT INTO user_recommendations(user_id,issue_id,score) "
            "VALUES(%s,%s,%s) "
            "ON DUPLICATE KEY UPDATE score=VALUES(score), updated_at=CURRENT_TIMESTAMP",
            recs
        )

    conn.commit()
    cur.close()
    conn.close()
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

    # 1) 유저 ID 목록
    cursor.execute("SELECT id FROM users")
    user_ids = [row[0] for row in cursor.fetchall()]

    # 2) 이슈 ID 목록
    cursor.execute("SELECT id FROM issues")
    issue_ids = [row[0] for row in cursor.fetchall()]

    # 3) 클릭 이벤트 엣지 로드
    cursor.execute(
        "SELECT user_id, issue_id FROM custom_events WHERE eventname = %s",
        ("click",)
    )
    raw_edges = cursor.fetchall()

    # 4) 유저 카테고리 벡터 로드 (or 대신 명시적 None 비교)
    cursor.execute("SELECT id, category_vec FROM users")
    user_feats = {}
    for uid, cat_blob in cursor.fetchall():
        arr = load_ndarray(cat_blob)
        # arr이 None 이면 0-벡터, 아니면 arr 그대로
        user_feats[uid] = arr if arr is not None else np.zeros(CAT_DIM_USER)

    # 5) 이슈 sentence_embedding + category_vec 로드
    cursor.execute("SELECT id, sentence_embedding, category_vec FROM issues")
    issue_feats = {}
    for iid, sent_blob, cat_blob in cursor.fetchall():
        s_arr = load_ndarray(sent_blob)
        c_arr = load_ndarray(cat_blob)
        # None 체크
        s_arr = s_arr if s_arr is not None else np.zeros(SENT_DIM)
        c_arr = c_arr if c_arr is not None else np.zeros(CAT_DIM_ISSUE)
        issue_feats[iid] = np.concatenate([s_arr, c_arr])

    # 6) GCN 임베딩 계산
    user_vecs, issue_vecs = user_item_GCN_embedding(
        user_ids=user_ids,
        item_ids=issue_ids,
        raw_edges=raw_edges,
        user_feats=user_feats,
        item_feats=issue_feats,
    )

    # 7) DB에 저장
    user_data = [(arr_to_blob(user_vecs[uid]), uid) for uid in user_ids]
    cursor.executemany(
        "UPDATE users SET gcn_vec=%s WHERE id=%s",
        user_data
    )
    issue_data = [(arr_to_blob(issue_vecs[iid]), iid) for iid in issue_ids]
    cursor.executemany(
        "UPDATE issues SET gcn_vec=%s WHERE id=%s",
        issue_data
    )

    conn.commit()
    cursor.close()
    conn.close()

    return jsonify({"message": "GCN embeddings updated"}), 200