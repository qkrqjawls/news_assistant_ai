import json
import io
import torch
import torch.nn.functional as F
from datetime import datetime, timezone, timedelta
from flask import Flask, request, jsonify
import mysql.connector
import numpy as np
import os
import sys
from GCN_embedding import user_item_GCN_embedding
import traceback


app = Flask(__name__)

DB_USER     = os.environ.get("DB_USER", "appuser")
DB_PASS     = os.environ.get("DB_PASS", "secure_app_password")
DB_NAME     = os.environ.get("DB_NAME", "myappdb")
DB_SOCKET   = os.environ.get("DB_SOCKET")   # ex) "/cloudsql/project:region:instance"

def get_db_connection():
    try:
        if DB_SOCKET:
            return mysql.connector.connect(
                user=DB_USER,
                password=DB_PASS,
                database=DB_NAME,
                unix_socket=DB_SOCKET,
            )
        else:
            return mysql.connector.connect(
                user=DB_USER,
                password=DB_PASS,
                database=DB_NAME,
                host="127.0.0.1",
                port=3306
            )
    except mysql.connector.Error as err:
        print("(!) DB 연결 실패", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        raise
def predict_date(articles):
    dates = sorted(a['pub_date'] for a in articles)
    return dates[len(dates)//4] if dates else None


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

