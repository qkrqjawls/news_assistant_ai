import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from sklearn.preprocessing import LabelEncoder
import numpy as np
import random


def user_item_GCN_embedding(user_ids : list, item_ids : list, raw_edges : list): # return (user id->vec(ndarray) 딕셔너리, item id->vec 딕셔너리) 튜플
    user_encoder = LabelEncoder()
    item_encoder = LabelEncoder()

    # 인코딩된 유저 ID는 0부터 시작
    user_idx_encoded = user_encoder.fit_transform(user_ids)
    # 인코딩된 아이템 ID는 유저 ID의 최대값 + 1부터 시작하여 유저-아이템 노드 ID 충돌 방지
    item_idx_encoded = item_encoder.fit_transform(item_ids) + len(user_ids)

    # 원본 ID와 인코딩된 ID 매핑 (확인용)
    user_id_to_encoded = {id: encoded for id, encoded in zip(user_ids, user_idx_encoded)}
    item_id_to_encoded = {id: encoded for id, encoded in zip(item_ids, item_idx_encoded)}
    encoded_to_user_id = {encoded: id for id, encoded in zip(user_ids, user_idx_encoded)}
    encoded_to_item_id = {encoded: id for id, encoded in zip(item_ids, item_idx_encoded)}

    # 원본 ID를 인코딩된 노드 ID로 변환하여 edge_index 생성
    src_nodes = [user_id_to_encoded[u] for u, i in raw_edges]
    dst_nodes = [item_id_to_encoded[i] for u, i in raw_edges]

    edge_index = torch.tensor([src_nodes, dst_nodes], dtype=torch.long)

    num_users = len(user_ids)
    num_items = len(item_ids)
    num_nodes = num_users + num_items

    # 2. 노드 특징 초기화: (유저 수 + 아이템 수) × 특징 차원
    # 여기서는 랜덤 초기화하지만 실제로는 ID 임베딩이나 기타 특징을 사용할 수 있습니다.
    node_features_dim = 16
    x = torch.randn((num_nodes, node_features_dim))

    data = Data(x=x, edge_index=edge_index)

    # 3. GCN 모델 (이전과 동일)
    class SimpleGCN(torch.nn.Module):
        def __init__(self, in_channels, hidden_channels, out_channels):
            super().__init__()
            self.conv1 = GCNConv(in_channels, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, out_channels)

        def forward(self, x, edge_index):
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = self.conv2(x, edge_index)
            return x

    # 4. 학습 설정 (이전과 동일)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleGCN(in_channels=node_features_dim, hidden_channels=32, out_channels=64).to(device)
    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # --- 5. BPR Loss 학습 루프 (수정된 부분) ---

    # 긍정 샘플 매핑 (인코딩된 ID 기준)
    # {user_encoded_id: [item_encoded_id, ...]}
    positive_interactions = {u_enc: [] for u_enc in user_idx_encoded}
    for i in range(edge_index.size(1)):
        u = edge_index[0, i].item()
        i = edge_index[1, i].item() - num_users # 아이템 인덱스를 0부터 시작하도록 조정
        positive_interactions[u].append(i)

    # 학습 에폭
    num_epochs = 200
    num_samples_per_user = 5 # 각 유저-아이템 쌍당 샘플링할 부정 아이템 수

    for epoch in range(1, num_epochs + 1):
        model.train()
        optimizer.zero_grad()

        out = model(data.x, data.edge_index)
        z_user = out[:num_users]  # 유저 임베딩 (인코딩된 ID 0 ~ num_users-1)
        z_item = out[num_users:]  # 아이템 임베딩 (인코딩된 ID num_users ~ num_nodes-1)

        total_loss = 0.0

        # 각 긍정 상호작용 (u, i_pos)에 대해 부정 샘플링
        for user_encoded_id, pos_items_encoded_ids in positive_interactions.items():
            if not pos_items_encoded_ids: # 상호작용이 없는 유저는 스킵
                continue

            # 해당 유저의 긍정 아이템 중 하나를 랜덤 선택
            positive_item_local_idx = random.choice(pos_items_encoded_ids) # 0부터 시작하는 아이템 인덱스
            
            # 부정 아이템 샘플링 (연결되지 않은 아이템 중에서)
            negative_item_local_idx = None
            all_item_local_indices = list(range(num_items))
            
            # 유저가 한 번도 상호작용하지 않은 아이템이 있을 경우, 그 중 하나를 선택
            # 모든 아이템과 상호작용했다면, 이 로직은 무의미해질 수 있음.
            # 현실에서는 거의 모든 아이템과 상호작용하는 유저는 드물다.
            possible_negative_items = list(set(all_item_local_indices) - set(pos_items_encoded_ids))
            
            if possible_negative_items: # 부정 샘플링 가능
                negative_item_local_idx = random.choice(possible_negative_items)
            else: # 모든 아이템과 상호작용했다면, 무작위로 하나 선택 (이상한 상황)
                negative_item_local_idx = random.choice(all_item_local_indices)
                
            # 선택된 유저, 긍정 아이템, 부정 아이템의 임베딩 가져오기
            u_emb = z_user[user_encoded_id]
            i_pos_emb = z_item[positive_item_local_idx]
            i_neg_emb = z_item[negative_item_local_idx]
            
            # BPR Loss 계산
            pos_score = (u_emb * i_pos_emb).sum()
            neg_score = (u_emb * i_neg_emb).sum()
            
            loss = -F.logsigmoid(pos_score - neg_score)
            total_loss += loss

        total_loss /= len(positive_interactions) # 평균 손실
        total_loss.backward()
        optimizer.step()

        if epoch % 20 == 0:
            print(f"Epoch {epoch:03d} | Average BPR Loss: {total_loss.item():.4f}")

    # 6. 학습 후 임베딩 확인 (이전과 동일)
    model.eval()
    with torch.no_grad():
        final_out = model(data.x, data.edge_index).cpu()
        users_emb = final_out[:num_users]
        items_emb = final_out[num_users:]

    user_id_to_vec = {encoded_to_user_id[i] : emb.numpy() for i, emb in enumerate(users_emb)}
    item_id_to_vec = {encoded_to_item_id[i + num_users] : emb.numpy() for i, emb in enumerate(items_emb)}

    return user_id_to_vec, item_id_to_vec



if __name__=="__main__":
    # --- 0. 유저·아이템 ID 매핑 (이전과 동일) ---
    user_ids = ["u123", "u456", "u789", "u001", "u002"] # 유저/아이템 수 늘림
    item_ids = ["i111", "i222", "i333", "i444", "i555", "i666"]
    raw_edges = [
        ("u123", "i111"),
        ("u456", "i111"),
        ("u789", "i222"),
        ("u123", "i222"),
        ("u789", "i333"),
        ("u001", "i444"),
        ("u002", "i555"),
        ("u123", "i333"), # 추가 예시
        ("u456", "i444"), # 추가 예시
    ]

    user_id_to_vec, item_id_to_vec = user_item_GCN_embedding(user_ids, item_ids, raw_edges)

    print("\n-- Learned User Embeddings --")
    for uid in user_id_to_vec:
        print(f"{uid}: {(user_id_to_vec[uid])[:5]}…") # 앞 5차원만 출력

    print("\n-- Learned Item Embeddings --")
    for iid in item_id_to_vec:
        print(f"{iid}: {(item_id_to_vec[iid])[:5]}…") # 앞 5차원만 출력
