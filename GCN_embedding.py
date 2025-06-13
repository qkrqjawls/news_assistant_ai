import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder


def user_item_GCN_embedding(
    user_ids: list,
    item_ids: list,
    raw_edges: list,
    user_feats: dict,
    item_feats: dict,
    emb_dim: int = 128,
    hidden_dim: int = 32,
    out_dim: int = None,
    bpr_epochs: int = 200,
    bpr_neg_ratio: int = 1,
    batch_size: int = 128,
    lr: float = 0.01,
    seed: int = 42 # 재현성을 위한 시드 추가
):
    """
    BPR 손실을 사용하여 GCN 기반 사용자 및 아이템 임베딩을 계산합니다.

    Args:
        user_ids (list): 고유한 사용자 ID 목록.
        item_ids (list): 고유한 아이템 ID 목록.
        raw_edges (list): 양의 상호작용을 나타내는 튜플 (사용자_ID, 아이템_ID) 목록.
        user_feats (dict): 사용자_ID를 특징 numpy 배열에 매핑하는 사전.
        item_feats (dict): 아이템_ID를 특징 numpy 배열에 매핑하는 사전.
        emb_dim (int): 원시 특징의 초기 투영을 위한 차원.
        hidden_dim (int): GCN의 숨겨진 계층의 차원.
        out_dim (int): 최종 임베딩의 차원. 기본값은 `emb_dim`.
        bpr_epochs (int): BPR 학습을 위한 에포크 수.
        bpr_neg_ratio (int): BPR을 위한 양의 샘플당 음의 샘플 수.
        batch_size (int): BPR 학습을 위한 배치 크기.
        lr (float): 옵티마이저 학습률.
        seed (int): 재현성을 위한 랜덤 시드.

    Returns:
        두 개의 딕셔너리 튜플: (user_id -> 임베딩 np.ndarray, item_id -> 임베딩 np.ndarray)
    """
    # 재현성을 위해 시드 설정
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 기본 out_dim 설정
    OUT_DIM = emb_dim if out_dim is None else out_dim

    # 빈 상호작용에 대한 입력 유효성 검사
    if not raw_edges:
        print("경고: `raw_edges`가 비어 있습니다. 학습할 상호작용이 없습니다.")
        return {uid: np.zeros(OUT_DIM) for uid in user_ids}, \
               {iid: np.zeros(OUT_DIM) for iid in item_ids}


    # ID 인코딩
    ue = LabelEncoder().fit(user_ids)
    ie = LabelEncoder().fit(item_ids)
    user_idx = ue.transform(user_ids)
    # 아이템 인덱스에 사용자 수만큼 오프셋을 더하여 고유하게 만듦
    item_idx = ie.transform(item_ids) + len(user_ids)
    user_to_enc = dict(zip(user_ids, user_idx))
    item_to_enc = dict(zip(item_ids, item_idx))
    inv_user = {enc: uid for uid, enc in user_to_enc.items()}
    inv_item = {enc: iid for iid, enc in item_to_enc.items()}

    num_users = len(user_ids)
    num_items = len(item_ids)
    num_nodes = num_users + num_items # 전체 노드 수 (사용자 + 아이템)

    # 원시 특징 차원 결정
    # 특징 딕셔너리가 비어 있을 수 있는 경우 처리
    dim_u = next(iter(user_feats.values())).shape[0] if user_feats else 0
    dim_i = next(iter(item_feats.values())).shape[0] if item_feats else 0
    # 사용자 및 아이템 특징 중 최대 차원을 원시 특징 차원으로 설정 (특징이 없는 경우 emb_dim 폴백)
    raw_feat_dim = max(dim_u, dim_i) if dim_u or dim_i else emb_dim

    # raw_feat_dim이 0인 경우 (예: 제공된 특징 없음) emb_dim으로 기본 설정
    if raw_feat_dim == 0:
        raw_feat_dim = emb_dim

    # 엣지 구성 (무방향 그래프를 위해 양방향으로 구성)
    src = [user_to_enc[u] for u, i in raw_edges] # 소스 노드 (사용자)
    dst = [item_to_enc[i] for u, i in raw_edges] # 대상 노드 (아이템)
    edge_index = torch.tensor([src + dst, dst + src], dtype=torch.long)

    # 원시 특징 행렬 초기화
    x_raw = torch.zeros((num_nodes, raw_feat_dim), dtype=torch.float32)
    # 사용자 특징 채우기
    for uid, enc in user_to_enc.items():
        # 사용자 특징을 가져오거나 기본값 (모든 0) 사용
        feat = user_feats.get(uid, np.zeros(dim_u if dim_u > 0 else raw_feat_dim))
        if feat.shape[0] < raw_feat_dim:
            # 특징 차원이 raw_feat_dim보다 작으면 패딩
            feat = np.pad(feat, (0, raw_feat_dim - feat.shape[0]))
        elif feat.shape[0] > raw_feat_dim: # 특징 차원이 raw_feat_dim보다 크면 자르기
            feat = feat[:raw_feat_dim]
        x_raw[enc] = torch.from_numpy(feat).float()
    # 아이템 특징 채우기
    for iid, enc in item_to_enc.items():
        # 아이템 특징을 가져오거나 기본값 (모든 0) 사용
        feat = item_feats.get(iid, np.zeros(dim_i if dim_i > 0 else raw_feat_dim))
        if feat.shape[0] < raw_feat_dim:
            # 특징 차원이 raw_feat_dim보다 작으면 패딩
            feat = np.pad(feat, (0, raw_feat_dim - feat.shape[0]))
        elif feat.shape[0] > raw_feat_dim: # 특징 차원이 raw_feat_dim보다 크면 자르기
            feat = feat[:raw_feat_dim]
        x_raw[enc] = torch.from_numpy(feat).float()

    # PyTorch Geometric Data 객체 생성
    data = Data(x=x_raw, edge_index=edge_index)

    # 입력 투영을 포함한 GCN 모델 정의
    class SimpleGCN(nn.Module):
        def __init__(self, raw_dim, emb_dim, hidden_dim, out_dim):
            super().__init__()
            # 원시 입력 특징을 임베딩 차원으로 투영
            self.input_proj = nn.Linear(raw_dim, emb_dim)
            # 첫 번째 GCN 계층
            self.conv1 = GCNConv(emb_dim, hidden_dim)
            # 두 번째 GCN 계층, 최종 임베딩 차원을 출력
            self.conv2 = GCNConv(hidden_dim, out_dim)

        def forward(self, x_raw, edge_index):
            # 원시 특징 투영
            x = self.input_proj(x_raw)
            # ReLU 활성화 함수와 함께 첫 번째 GCN 계층 적용
            x = F.relu(self.conv1(x, edge_index))
            # 두 번째 GCN 계층 적용
            return self.conv2(x, edge_index)

    # 학습을 위한 장치 설정 (GPU 사용 가능 시 GPU, 아니면 CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleGCN(raw_feat_dim, emb_dim, hidden_dim, OUT_DIM).to(device)
    data = data.to(device) # 그래프 데이터를 장치로 이동
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 양의 상호작용 맵 준비: 사용자_인코딩_ID -> 아이템_0_인덱싱_ID 목록
    pos_map = {}
    for u_enc, i_enc in zip(src, dst):
        pos_map.setdefault(u_enc, []).append(i_enc - num_users)

    # 양의 (사용자, 양의_아이템, 음의_아이템) 삼중항 샘플링을 위한 BPR 데이터셋
    class BPRDataset(Dataset):
        def __init__(self, pos_map, num_items, neg_ratio):
            # 모든 사용자-양의 아이템 쌍 목록 생성
            self.pairs = [(u, pos) for u, items in pos_map.items() for pos in items]
            self.num_items = num_items
            self.neg_ratio = neg_ratio
            self.pos_map = pos_map # 음의 샘플을 확인하기 위해 pos_map 저장

        def __len__(self):
            return len(self.pairs)

        def __getitem__(self, idx):
            u, pos = self.pairs[idx]
            negs = []
            # 각 양의 쌍에 대해 `neg_ratio`개의 음의 아이템 샘플링
            for _ in range(self.neg_ratio):
                neg = random.randrange(self.num_items)
                # 샘플링된 음의 아이템이 현재 사용자에 대한 양의 아이템이 아닌지 확인
                while neg in self.pos_map.get(u, []):
                    neg = random.randrange(self.num_items)
                negs.append(neg)
            return u, pos, negs

    # BPR 데이터셋 및 데이터 로더 초기화
    dataset = BPRDataset(pos_map, num_items, bpr_neg_ratio)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # BPR 손실을 위한 학습 루프
    model.train() # 모델을 학습 모드로 설정
    for epoch in range(1, bpr_epochs + 1):
        epoch_loss = 0.0
        for us, pos, negs_batch in loader:
            # 배치 데이터를 장치로 이동
            us = us.to(device)
            pos = pos.to(device)
            # negs_batch는 [B, neg_ratio] 형태로 오므로, long 타입으로 변환
            negs_batch = negs_batch.to(device).long()

            optimizer.zero_grad() # 기울기 초기화

            # GCN 모델에서 현재 임베딩 가져오기
            out = model(data.x, data.edge_index)
            z_user = out[:num_users] # 사용자 임베딩
            z_item = out[num_users:] # 아이템 임베딩 (아이템에 대한 0-인덱싱)

            # 현재 배치에 대한 임베딩 검색
            u_emb = z_user[us]  # [B, emb_dim]
            p_emb = z_item[pos]  # [B, emb_dim]
            n_emb = z_item[negs_batch] # [B, neg_ratio, emb_dim]

            # 점수 계산: 사용자 임베딩과 양의/음의 아이템 임베딩의 내적
            pos_scores = torch.sum(u_emb * p_emb, dim=1) # [B]
            
            # 음의 아이템 차원에 맞게 사용자 임베딩 확장하여 요소별 곱셈 수행
            expanded_u_emb = u_emb.unsqueeze(1) # [B, 1, emb_dim]
            neg_scores = torch.sum(expanded_u_emb * n_emb, dim=2) # [B, neg_ratio]

            # BPR 손실 계산: -log(sigmoid(양의_점수 - 음의_점수))
            # 브로드캐스팅을 위해 neg_scores에 맞게 pos_scores 확장
            expanded_pos_scores = pos_scores.unsqueeze(1) # [B, 1]
            loss = -F.logsigmoid(expanded_pos_scores - neg_scores).mean()

            loss.backward() # 손실 역전파
            optimizer.step() # 모델 파라미터 업데이트
            epoch_loss += loss.item() # 에포크 손실 누적
        print(f"[BPR] Epoch {epoch}/{bpr_epochs} loss={epoch_loss/len(loader):.4f}")

    # 학습 후 최종 임베딩 계산
    model.eval() # 모델을 평가 모드로 설정
    with torch.no_grad(): # 기울기 계산 비활성화
        # 임베딩 가져오기, CPU로 이동, numpy 배열로 변환
        final_out = model(data.x, data.edge_index).cpu().numpy()
    
    # 사용자 및 아이템 임베딩 분리
    users_emb = final_out[:num_users]
    items_emb = final_out[num_users:]

    # 인코딩된 인덱스를 원래 ID로 다시 매핑하여 최종 출력
    user_id_to_vec = {inv_user[i]: users_emb[i] for i in range(num_users)}
    item_id_to_vec = {inv_item[i + num_users]: items_emb[i] for i in range(num_items)}

    return user_id_to_vec, item_id_to_vec
