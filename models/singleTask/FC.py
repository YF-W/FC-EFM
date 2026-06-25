import torch
import torch.nn as nn
from ..subNets import BertTextEncoder
__all__ = ['FF']
class SequenceAligner(nn.Module):
    def __init__(self, target_len):
        super().__init__()
        self.target_len = target_len
    def forward(self, x):
        # x: [B, T, D]
        B, T, D = x.shape
        if T == self.target_len:
            return x
        # 插值对齐
        x = x.transpose(1, 2)  # -> [B, D, T]，因为 interpolate 对最后两维做插值
        x = torch.nn.functional.interpolate(x, size=self.target_len, mode='linear', align_corners=False)
        x = x.transpose(1, 2)  # -> [B, target_len, D]
        return x
# ========== 基于注意力的模态对齐模块 ==========
class CrossAttentionAlign(nn.Module):
    def __init__(self, align_dim, num_heads=4):
        super().__init__()
        # ======== Multi-Head Attention 对齐器（六方向）========
        self.t2a = nn.MultiheadAttention(align_dim, num_heads, batch_first=True)
        self.a2t = nn.MultiheadAttention(align_dim, num_heads, batch_first=True)
        self.t2v = nn.MultiheadAttention(align_dim, num_heads, batch_first=True)
        self.v2t = nn.MultiheadAttention(align_dim, num_heads, batch_first=True)
        self.a2v = nn.MultiheadAttention(align_dim, num_heads, batch_first=True)
        self.v2a = nn.MultiheadAttention(align_dim, num_heads, batch_first=True)
        # ======== LayerNorm（按模态区分）========
        self.ln_t = nn.LayerNorm(align_dim)
        self.ln_a = nn.LayerNorm(align_dim)
        self.ln_v = nn.LayerNorm(align_dim)
    def forward(self,t_proj, v_proj, a_proj):
        t2a_out, _ = self.t2a(t_proj, a_proj, a_proj)# T → A
        T2A = self.ln_t(t_proj + t2a_out)
        a2t_out, _ = self.a2t(a_proj, t_proj, t_proj)# A → T
        A2T = self.ln_a(a_proj + a2t_out)
        t2v_out, _ = self.t2v(t_proj, v_proj, v_proj)# T → V
        T2V = self.ln_t(t_proj + t2v_out)
        v2t_out, _ = self.v2t(v_proj, t_proj, t_proj)# V → T
        V2T = self.ln_v(v_proj + v2t_out)
        a2v_out, _ = self.a2v(a_proj, v_proj, v_proj)# A → V
        A2V = self.ln_a(a_proj + a2v_out)
        v2a_out, _ = self.v2a(v_proj, a_proj, a_proj)# V → A
        V2A = self.ln_v(v_proj + v2a_out)
        return {
            "T2A": T2A, "A2T": A2T,
            "T2V": T2V, "V2T": V2T,
            "A2V": A2V, "V2A": V2A
        }
#冲突特征提取
class ConflictExtractor(nn.Module):
    def __init__(self,align_dim):
        super().__init__()
        self.proj = nn.Linear(align_dim, align_dim)
    def forward(self, T, A, V, T2A, A2T, T2V, V2T, A2V, V2A):
        conf_TA = T2A - T     # 文本对齐音频后的差异
        conf_TV = T2V - T     # 文本对齐视频后的差异
        conflict_T = conf_TA + conf_TV   # 综合文本冲突（来自音频＋视频）
        conf_AT = A2T - A
        conf_AV = A2V - A
        conflict_A = conf_AT + conf_AV
        conf_VT = V2T - V
        conf_VA = V2A - V
        conflict_V = conf_VT + conf_VA
        conflict_T = self.proj(conflict_T)
        conflict_A = self.proj(conflict_A)
        conflict_V = self.proj(conflict_V)
        return conflict_T, conflict_A, conflict_V
class ConflictTripletExtractor(nn.Module):
    def __init__(self, window_size=2, pool="mean"):
        super().__init__()
        self.window_size = window_size
        self.pool = pool
    def temporal_pool(self, x, D):
        if x.size(1) == 0:
            return torch.zeros((1,D), device=x.device)
        if self.pool == "mean":
            return x.mean(dim=1, keepdim=True)
        elif self.pool == "max":
            return x.max(dim=1, keepdim=True)[0]
        else:
            raise ValueError("pool must be 'mean' or 'max'")
    def forward(self, conflict_feat, conflict_mask):
        B, L, D = conflict_feat.shape
        Pre_list, Con_list, Post_list = [], [], []
        for b in range(B):
            idx = torch.where(conflict_mask[b] > 0)[0]
            if len(idx) == 0:
                # 没有冲突点 → 用序列中点
                c = min(L - 1, L // 2)
            else:
                # 取第一个冲突点
                c = idx[0].item()
                c = min(L - 1, c)  # 安全防止越界
            # ===== 切片 =====
            pre_start  = max(0, c - self.window_size)
            pre_end    = c
            post_start = c + 1
            post_end   = min(L, c + 1 + self.window_size)
            pre_feat  = conflict_feat[b:b+1, pre_start:pre_end, :]   # [1,T1,D]
            con_feat  = conflict_feat[b:b+1, c:c+1, :]               # [1,1,D]
            post_feat = conflict_feat[b:b+1, post_start:post_end, :] # [1,T2,D]
            # ===== 池化 =====
            Pre  = self.temporal_pool(pre_feat, D).squeeze(1)  # [1,D] -> [D]
            # 冲突特征为空时用零向量
            if con_feat.size(1) == 0:
                Con = torch.zeros((D,), device=conflict_feat.device)
            else:
                Con = con_feat.squeeze(1)  # [1,1,D] -> [1,D] -> squeeze -> [D]
            Post = self.temporal_pool(post_feat, D).squeeze(1) # [1,D] -> [D]
            Pre_list.append(Pre)
            Con_list.append(Con)
            Post_list.append(Post)
        # ===== 堆叠 =====
        Pre  = torch.stack(Pre_list, dim=0)   # [B,D]
        Con  = torch.stack(Con_list, dim=0)   # [B,D]
        Post = torch.stack(Post_list, dim=0)  # [B,D]
        return Pre, Con, Post
# ========== 结构算子模块 ==========
class StructuralOperators(nn.Module):
    #O_pre  : 结构基准算子（稳定参考系） O_con  : 结构偏移算子（相对结构） O_post : 结构响应算子（结构反馈）
    def __init__(self, dim, alpha=1.0, beta=1.0, gamma=1.0, use_norm=True):
        super().__init__()
        self.alpha = alpha   # 冲突强度系数
        self.beta = beta     # 响应强度系数
        self.gamma = gamma   # 平滑强度系数
        self.use_norm = use_norm# 去噪/稳定化投影（用于 O_pre）
        self.pre_proj = nn.Linear(dim, dim)
        # 结构映射（用于 O_con, O_post）
        self.con_proj = nn.Linear(dim, dim)
        self.post_proj = nn.Linear(dim, dim)
        if use_norm:
            self.ln_pre = nn.LayerNorm(dim)
            self.ln_con = nn.LayerNorm(dim)
            self.ln_post = nn.LayerNorm(dim)
    def O_pre(self, Pre):
        # 结构基准算子（稳定参考结构）
        Pre_ref = self.pre_proj(Pre)
        if self.use_norm:
            Pre_ref = self.ln_pre(Pre_ref)
        return Pre_ref
    def O_con(self, Con, Pre_ref):
        # 结构偏移算子（相对结构）
        Rel = Con - Pre_ref                      # 相对结构
        Rel = self.alpha * self.con_proj(Rel)
        if self.use_norm:
            Rel = self.ln_con(Rel)
        return Rel
    def O_post(self, Post, Rel):
        # 结构响应算子（结构反馈）
        Resp = Post - self.beta * Rel
        Resp = self.post_proj(Resp)
        if self.use_norm:
            Resp = self.ln_post(Resp)
        return Resp
    def forward(self, Pre, Con, Post):
        Pre_ref = self.O_pre(Pre)                # 结构基准
        Rel     = self.O_con(Con, Pre_ref)       # 相对冲突结构
        Resp    = self.O_post(Post, Rel)         # 结构响应
        return Pre_ref, Rel, Resp
def non_traditional_similarity(x, y,alpha):
    #非传统相似度：衡量方向/趋势一致性，而不是传统欧氏距离 x, y: [D] 或 [B, D]返回: 相似度标量或向量
    x_norm = torch.nn.functional.normalize(x, dim=-1)
    y_norm = torch.nn.functional.normalize(y, dim=-1)
    sim = torch.sum(x_norm * y_norm, dim=-1)  # 越大表示趋势一致
    return alpha * sim
class ConflictLocator(nn.Module):
    #谱偏移 + 真正冲突特征定位
    def __init__(self, spectral_metric="l2", top_k=1):
        #spectral_metric: 谱偏移计算方式, 'l2' 或 'cosine' top_k: 每条序列选取的冲突时间步数量
        super().__init__()
        self.spectral_metric = spectral_metric
        self.top_k = top_k
    def spectral_shift(self, matrix, stable_matrix):
        #计算谱偏移matrix: [B, 9, D] 当前结构矩阵stable_matrix: [B, 3, D] 稳定参考矩阵 (FakeStableMatrix)返回: [B, 9] 每个时间步的偏差分数
        B, N, D = matrix.shape
        if stable_matrix.dim() == 4 and stable_matrix.size(2) == 1:
            stable_matrix = stable_matrix.squeeze(2)
        ref = stable_matrix.repeat(1, 3, 1)  # 扩展到9行对齐
        if self.spectral_metric == "l2":
            shift = torch.norm(matrix - ref, dim=-1)
        elif self.spectral_metric == "cosine":
            x_norm = torch.nn.functional.normalize(matrix, dim=-1)
            y_norm = torch.nn.functional.normalize(ref, dim=-1)
            shift = 1 - (x_norm * y_norm).sum(dim=-1)
        else:
            raise NotImplementedError(f"{self.spectral_metric} not supported")
        return shift
    def forward(self, structure_matrix, stable_matrix):
        #structure_matrix: [B, 9, D]stable_matrix: [B, 3, D]返回:true_conflict_T/A/V: [B, top_k, D] 选取 top-k 冲突时间步
        structure_matrix = structure_matrix.squeeze(2)  # 压掉第3维
        B, N, D = structure_matrix.shape
        # 1️⃣ 谱偏移
        spectral_scores = self.spectral_shift(structure_matrix, stable_matrix)  # [B, 9]
        # 2️⃣ 冲突时间步筛选
        spectral_threshold = spectral_scores.mean(dim=1, keepdim=True)
        conflict_time_mask = (spectral_scores > spectral_threshold).float()  # [B, 9]
        # 3️⃣ 真正冲突特征提取（修正版）
        true_conflict_T, true_conflict_A, true_conflict_V = [], [], []
        for b in range(B):
            idx = torch.where(conflict_time_mask[b] > 0)[0]
            if len(idx) == 0:
                idx = [N // 2]  # 若无冲突，取中间时间步
            if len(idx) > self.top_k:
                idx = idx[:self.top_k]  # 取 top-k
            # 确定最大时间步数
            max_time_step = structure_matrix.size(1) // 3  # 每个时间步 3 行
            # 限制 idx 不超过最大时间步数
            idx = [i for i in idx if i < max_time_step]
            if len(idx) == 0:
                idx = [max_time_step // 2]  # 保底，防止空列表
            # 再生成 T/A/V 的行索引
            T_rows = [i * 3 + 0 for i in idx]
            A_rows = [i * 3 + 1 for i in idx]
            V_rows = [i * 3 + 2 for i in idx]
            true_conflict_T.append(structure_matrix[b, T_rows, :])  # [top_k, D]
            true_conflict_A.append(structure_matrix[b, A_rows, :])
            true_conflict_V.append(structure_matrix[b, V_rows, :])
        # 转为张量 [B, top_k, D]
        true_conflict_T = torch.stack(true_conflict_T, dim=0)
        true_conflict_A = torch.stack(true_conflict_A, dim=0)
        true_conflict_V = torch.stack(true_conflict_V, dim=0)
        return true_conflict_T, true_conflict_A, true_conflict_V
class LocalFeatureExtractor(nn.Module):
    #从冲突特征序列中提取局部粒度特征text: 按词/子词重要性聚合audio: 短帧音频特征video: ROI 或 patch 特征
    def __init__(self, window_size=3, pool="mean"):
        super().__init__()
        self.window_size = window_size
        self.pool = pool
    def temporal_pool(self, x):
        # x: [B, T, D]
        if x.size(1) == 0:
            return None
        if self.pool == "mean":
            return x.mean(dim=1)
        elif self.pool == "max":
            return x.max(dim=1)[0]
        else:
            raise ValueError("pool must be 'mean' or 'max'")
    def forward(self, conflict_feat):
        # conflict_feat: [B, L, D]  -> 局部级特征序列
        B, L, D = conflict_feat.shape
        local_feats = []
        for b in range(B):
            # 遍历每个时间步，构建局部窗口
            feat_windows = []
            for t in range(L):
                start = max(0, t - self.window_size // 2)
                end = min(L, t + self.window_size // 2 + 1)
                window_feat = conflict_feat[b:b + 1, start:end, :]  # [1, window_size, D]
                pooled = self.temporal_pool(window_feat)
                if pooled is None:
                    pooled = torch.zeros(D, device=conflict_feat.device)
                feat_windows.append(pooled)
            local_feats.append(torch.stack(feat_windows, dim=0))  # [L, D]
        local_feats = torch.stack(local_feats, dim=0)  # [B, L, D]
        return local_feats
class LocalFeature3DSpaceBuilder(nn.Module):
    #构建微观局部特征3D空间（Local Feature 3D Space）p_i = (t_i, e_i, δ_i)
    def __init__(self, eps=1e-8, normalize=True):
        super().__init__()
        self.eps = eps
        self.normalize = normalize
    def forward(self, local_feat):
        #local_feat: [B, L, D]return:P: [B, L, 3]  -> 3D空间点云dict: 结构分量
        B, L, D = local_feat.shape
        device = local_feat.device
        # ===== X轴：时间轴 t_i =====
        t = torch.linspace(0, 1, L, device=device)  # [L]
        t = t.view(1, L).repeat(B, 1)               # [B, L]
        # ===== Y轴：能量轴 e_i =====
        e = torch.norm(local_feat, dim=-1)          # [B, L]
        # ===== Z轴：结构扰动轴 δ_i =====
        delta = torch.zeros_like(e)
        delta[:, 1:] = torch.norm(local_feat[:, 1:] - local_feat[:, :-1], dim=-1)
        # ===== 归一化（可选）=====
        if self.normalize:
            e = (e - e.mean(dim=1, keepdim=True)) / (e.std(dim=1, keepdim=True) + self.eps)
            delta = (delta - delta.mean(dim=1, keepdim=True)) / (delta.std(dim=1, keepdim=True) + self.eps)
        # ===== 构建3D空间点云 =====
        P = torch.stack([t, e, delta], dim=-1)  # [B, L, 3]
        return P, {"t": t, "e": e, "delta": delta}
class TriModalConflictSpace(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps
    def forward(self, conflict_T, conflict_A, conflict_V):
        # SCon_* : [B, D]
        # ===== X轴：一致性 Consistency =====
        def cos(x, y):
            return torch.nn.functional.cosine_similarity(x, y, dim=-1)
        c_ta = cos(conflict_T, conflict_A)
        c_tv = cos(conflict_T, conflict_V)
        c_av = cos(conflict_A, conflict_V)
        c = (c_ta + c_tv + c_av) / 3.0          # [B]
        # ===== Z轴：冲突强度 Intensity =====
        k_ta = torch.norm(conflict_T - conflict_A, dim=-1)
        k_tv = torch.norm(conflict_T - conflict_V, dim=-1)
        k_av = torch.norm(conflict_A - conflict_V, dim=-1)
        k = (k_ta + k_tv + k_av) / 3.0          # [B]
        # ===== Y轴：模态贡献度 Contribution =====
        nT = torch.norm(conflict_T, dim=-1)
        nA = torch.norm(conflict_A, dim=-1)
        nV = torch.norm(conflict_V, dim=-1)
        denom = nT + nA + nV + self.eps
        wT = nT / denom
        wA = nA / denom
        wV = nV / denom
        # ===== 结构点 q =====
        q = torch.stack([c, k, wT, wA, wV], dim=-1)
        # q: [B, 5]
        # (一致性, 冲突强度, 文本贡献, 音频贡献, 视频贡献)
        return q, {"c": c, "k": k, "wT": wT, "wA": wA, "wV": wV}
# 融合层
class FusionModule(nn.Module):
    def __init__(self, text_dim_out, audio_dim_out, video_dim_out, hidden_dims):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Linear(text_dim_out + audio_dim_out + video_dim_out, hidden_dims),
            nn.GELU(),
            nn.LayerNorm(hidden_dims),
            nn.Linear(hidden_dims, hidden_dims),
            nn.GELU(),
            nn.LayerNorm(hidden_dims)
        )
    def forward(self, t_proj, v_proj, a_proj):
        x = torch.cat([t_proj, v_proj, a_proj], dim=-1)
        return self.fuse(x)
# 预测头
class FusionPredictor(nn.Module):
    def __init__(self, fusion_dim, hidden_dims, dropout_prob):
        super(FusionPredictor, self).__init__()
        # 构建多层全连接
        self.mlp = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dims),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dims, 1)
        )
    def forward(self, fusion):
        output = self.mlp(fusion)
        return output
class FF(nn.Module):
    def __init__(self, args):
        # 使用配置文件中的参数创建模
        super(FF, self).__init__()# 调用父类构造函数，初始化 nn.Module 基础结构
        self.matrix_alpha=args.matrix_alpha
        self.matrix_noise=args.matrix_noise
        self.query_proj = nn.Linear(5, 256)
        # 文本输入形状: (batch_size, 3, seq_len)
        self.text_model = BertTextEncoder(transformers=args.transformers,pretrained=args.pretrained)
        # ===== 视频特征提取器：BiLSTM  =====
        self.video_extractor = nn.LSTM(
            input_size=args.video_inputs, # 每一帧视频特征的维度
            hidden_size=args.video_hidden, # LSTM 隐层大小
            num_layers = args.lstm_num_layers, # 堆叠的 LSTM 层数
            dropout=args.dropouts, # dropout 概率
            batch_first=True, # 指定输入张量维度格式为 (batch, seq, feature)
            bidirectional=True# 双向 LSTM，用于捕获前后时序信息
        )
        # ===== 音频特征提取器：BiLSTM =====
        self.audio_extractor = nn.LSTM(input_size=args.audio_inputs,  # 每帧音频特征维度
                                           hidden_size=args.audio_hidden,  # LSTM 隐层大小
                                           num_layers=args.lstm_num_layers,  # LSTM 层数
                                           dropout=args.dropouts,  # dropout 概率
                                           batch_first=True,  # 输入格式同上
                                           bidirectional=True)  # 双向 LSTM
        # === 线性映射，将各模态序列特征映射到相同的 align_dim ===
        self.map_text = nn.Linear(args.post_text_inputs,args.align_dim)  # text BERT hidden dim 通常为 self.text_in (e.g. 768)
        self.map_audio = nn.Linear(2*args.audio_hidden, args.align_dim)  # LSTM 是双向 -> 输出通道为 2*hidden
        self.map_video = nn.Linear(2*args.video_hidden, args.align_dim)  # LSTM 是双向 -> 输出通道为 2*hidden
        # ===序列对齐器 ===
        self.seq_align_text = SequenceAligner(target_len=args.text_len)
        self.seq_align_audio = SequenceAligner(target_len=args.text_len)
        self.seq_align_video = SequenceAligner(target_len=args.text_len)
        # === 对齐模块 ===
        self.CrossAttentionAlign = CrossAttentionAlign(align_dim=args.align_dim)
        # === 冲突特征提取模块 ===
        self.ConflictExtractor = ConflictExtractor(align_dim=args.align_dim)
        # === 冲突时间三段式结构提取器 ===
        self.ConflictTripletExtractor = ConflictTripletExtractor(
            window_size=args.conflict_window,
            pool=args.conflict_pool
        )
        # === 结构算子模块 ===
        self.StructuralOperators = StructuralOperators(
            dim=args.align_dim,
            alpha=args.alpha_con,
            beta=args.beta_post,
            gamma=args.gamma_pre,
            use_norm=args.use_struct_norm
        )
        # 冲突特征定位器
        self.ConflictLocator = ConflictLocator(
            spectral_metric="l2",  # 谱偏移方法
            top_k=5  # 每条序列取1个冲突时间步，如果想保留多个可改为>1
        )
        # 添加局部特征提取
        self.local_extractor = LocalFeatureExtractor(window_size=3, pool="mean")
        # === 微观局部3D空间构建器 ===
        self.Local3DSpaceBuilder = LocalFeature3DSpaceBuilder(
            normalize=True)
        # === 三模态冲突结构空间 ===
        self.TriModalConflictSpace = TriModalConflictSpace()

        # 融合层
        self.FusionModule = FusionModule(
            text_dim_out=args.align_dim,  # 或者 encoder 输出维度
            audio_dim_out=args.align_dim,
            video_dim_out=args.align_dim,
            hidden_dims=args.fusion_dim,
        )
        # 预测层
        self.FusionPredictor = FusionPredictor(
            fusion_dim=args.fusion_dim,
            hidden_dims=args.pred_hidden,
            dropout_prob=args.dropouts
        )
    def forward(self, text_inputs,video_inputs,audio_inputs):
        # ===== 文本特征提取 =====
        text_feature = self.text_model(text_inputs)
        # ===== 音频特征提取 =====
        audio_out, _ = self.audio_extractor(audio_inputs)  # [B, T_audio, 2*hidden_dim]
        audio_feature = audio_out
        # ===== 视频特征提取 =====
        video_out, _ = self.video_extractor(video_inputs)  # [B, T_video, 2*hidden_dim]
        video_feature = video_out  # 直接保留时间维
        if isinstance(text_feature, tuple):
            text_feature = text_feature[0]
        if isinstance(audio_feature, tuple):
            audio_feature = audio_feature[0]
        if isinstance(video_feature, tuple):
            video_feature = video_feature[0]
        # ================== 投影到统一维度 ==================
        t_proj = self.map_text(text_feature)
        a_proj = self.map_audio(audio_feature)
        v_proj = self.map_video(video_feature)
        # ==================Cross-attention 对齐 ==================
        align_outputs = self.CrossAttentionAlign(t_proj, v_proj, a_proj)
        T2A = align_outputs["T2A"]
        A2T = align_outputs["A2T"]
        T2V = align_outputs["T2V"]
        V2T = align_outputs["V2T"]
        A2V = align_outputs["A2V"]
        V2A = align_outputs["V2A"]
        # 冲突特征提取模块
        conflict_T, conflict_A, conflict_V = self.ConflictExtractor(
            T=t_proj, A=a_proj, V=v_proj,
            T2A=T2A, A2T=A2T, T2V=T2V,
            V2T=V2T, A2V=A2V, V2A=V2A
        )
        # 1️⃣ 转为 [B, D, L]，确保是浮点型
        conflict_T_aligned = conflict_T.transpose(1, 2).contiguous().float()  # [B, D, L_T]
        # 2️⃣ 插值到音频长度 L_A
        conflict_T_aligned = torch.nn.functional.interpolate(
            conflict_T_aligned,
            size=conflict_A.size(1),  # L_A
            mode='linear',
            align_corners=False
        )
        # 3️⃣ 转回 [B, L_A, D]
        conflict_T_aligned = conflict_T_aligned.transpose(1, 2).contiguous()  # [B, L_A, D]
        # ================== 冲突时间定位 ==================
        # 以 T-A 冲突强度作为时间冲突定位依据
        conflict_score = torch.norm(conflict_T_aligned - conflict_A, dim=-1)  # [B, L]
        threshold = conflict_score.mean(dim=1, keepdim=True)
        conflict_mask = (conflict_score > threshold).float()  # [B, L]
        # ================== 三段式时间因果切片 ==================
        Pre_T, Con_T, Post_T = self.ConflictTripletExtractor(conflict_T, conflict_mask)
        Pre_A, Con_A, Post_A = self.ConflictTripletExtractor(conflict_A, conflict_mask)
        Pre_V, Con_V, Post_V = self.ConflictTripletExtractor(conflict_V, conflict_mask)
        # ================== 结构算子变换 ==================
        SPre_T, SCon_T, SPost_T = self.StructuralOperators(Pre_T, Con_T, Post_T)# 文本模态
        SPre_A, SCon_A, SPost_A = self.StructuralOperators(Pre_A, Con_A, Post_A)# 音频模态
        SPre_V, SCon_V, SPost_V = self.StructuralOperators(Pre_V, Con_V, Post_V)# 视频模态
        # ================== 矩阵构建 ==================
        B,M,D = SPre_T.shape
        # 1️⃣ Operator Matrix（算子下矩阵）
        OperatorMatrix = torch.stack([SPre_T, SPre_A, SPre_V,
                                      SCon_T, SCon_A, SCon_V,
                                      SPost_T, SPost_A, SPost_V], dim=1)  # [B, 9, D]
        # 2️⃣ Fake Stable Matrix（虚拟稳定矩阵）
        FakeStableMatrix = []
        for b in range(B):
            row_features = []
            for mod_feat in [SPre_T[b], SPre_A[b], SPre_V[b]]:
                candidates = [mod_feat + self.matrix_noise * torch.randn_like(mod_feat),
                              mod_feat + 2 * self.matrix_noise * torch.randn_like(mod_feat)]
                sims = torch.stack([non_traditional_similarity(mod_feat, candidate, alpha=self.matrix_alpha)
                                    for candidate in candidates])  # 不用 c, 避免与循环嵌套冲突
                weights = torch.nn.functional.softmax(sims, dim=0)  # 保持 F 是 functional
                virtual_post = torch.stack(
                    [w * candidate for w, candidate in zip(weights, candidates)],
                    dim=0
                ).sum(dim=0)
                row_features.append(virtual_post)
        FakeStableMatrix.append(torch.stack(row_features, dim=0))
        FakeStableMatrix = torch.stack(FakeStableMatrix, dim=0)  # [B, 3, D]
        # 3️⃣ Ideal Conflict Verification Matrix（理想冲突验证矩阵）
        IdealConflictMatrix = []
        for b in range(B):
            row_features = []
            for mod_pre, mod_rel in zip([SPre_T[b], SPre_A[b], SPre_V[b]],
                                        [SCon_T[b] - SPre_T[b], SCon_A[b] - SPre_A[b], SCon_V[b] - SPre_V[b]]):
                candidates = [mod_pre + mod_rel + self.matrix_noise * torch.randn_like(mod_pre),
                              mod_pre + mod_rel + 2 * self.matrix_noise * torch.randn_like(mod_pre)]
                sims = torch.stack([non_traditional_similarity(mod_feat, candidate, alpha=self.matrix_alpha)
                                    for candidate in candidates])  # 不用 c, 避免与循环嵌套冲突
                weights = torch.nn.functional.softmax(sims, dim=0)  # 保持 F 是 functional
                virtual_post_conflict = sum(w * c for w, c in zip(weights, candidates))
                row_features.append(virtual_post_conflict)
            IdealConflictMatrix.append(torch.stack(row_features, dim=0))
        IdealConflictMatrix = torch.stack(IdealConflictMatrix, dim=0)  # [B, 3, D]
        # #原始时间结构矩阵
        RawTemporalStructureMatrix = torch.stack([
            Pre_T, Pre_A, Pre_V,
            Con_T, Con_A, Con_V,
            Post_T, Post_A, Post_V
        ], dim=1)  # [B, 9, D]
        # 使用 ConflictLocator 获取真正冲突特征
        true_conflict_T, true_conflict_A, true_conflict_V = self.ConflictLocator(
            RawTemporalStructureMatrix, FakeStableMatrix
        )
        # 将真正冲突特征传入结构算子
        SPre_T, SCon_T, SPost_T = self.StructuralOperators(Pre_T, true_conflict_T.mean(dim=1), Post_T)
        SPre_A, SCon_A, SPost_A = self.StructuralOperators(Pre_A, true_conflict_A.mean(dim=1), Post_A)
        SPre_V, SCon_V, SPost_V = self.StructuralOperators(Pre_V, true_conflict_V.mean(dim=1), Post_V)
        conflict_modality = (SCon_T + SCon_A + SCon_V) / 3  # [B, D]
        # 添加局部特征提取
        local_conflict_T = self.local_extractor(conflict_T)  # [B, L, D]
        local_conflict_A = self.local_extractor(conflict_A)  # [B, L, D]
        local_conflict_V = self.local_extractor(conflict_V)  # [B, L, D]
        # ================= 微观局部3D空间构建 =================
        B, top_k, num_modalities, D = local_conflict_T.shape
        local_conflict_T_flat = local_conflict_T.view(B, top_k * num_modalities, D)
        P_T, info_T = self.Local3DSpaceBuilder(local_conflict_T_flat)
        B, top_k, num_modalities, D = local_conflict_A.shape
        local_conflict_A_flat = local_conflict_A.view(B, top_k * num_modalities, D)
        P_A, info_A = self.Local3DSpaceBuilder(local_conflict_A_flat)
        B, top_k, num_modalities, D = local_conflict_V.shape
        local_conflict_V_flat = local_conflict_V.view(B, top_k * num_modalities, D)
        P_V, info_V = self.Local3DSpaceBuilder(local_conflict_V_flat)
        # ================= 三模态冲突结构空间 =================
        # 找到当前 batch 各模态长度
        len_T = conflict_T.size(1)
        len_A = conflict_A.size(1)
        len_V = conflict_V.size(1)
        max_len = max(len_T, len_A, len_V)  # 对齐长度
        # pad 各模态到相同长度
        if len_T < max_len:
            pad_T = torch.zeros(conflict_T.size(0), max_len - len_T, conflict_T.size(2), device=conflict_T.device)
            conflict_T = torch.cat([conflict_T, pad_T], dim=1)
        if len_A < max_len:
            pad_A = torch.zeros(conflict_A.size(0), max_len - len_A, conflict_A.size(2), device=conflict_A.device)
            conflict_A = torch.cat([conflict_A, pad_A], dim=1)
        if len_V < max_len:
            pad_V = torch.zeros(conflict_V.size(0), max_len - len_V, conflict_V.size(2), device=conflict_V.device)
            conflict_V = torch.cat([conflict_V, pad_V], dim=1)
        q_struct, struct_info = self.TriModalConflictSpace(conflict_T, conflict_A, conflict_V)
        # q_struct: [B, 5] = (c, k, wT, wA, wV)
        # ================== 微观局部3D空间构建 =================
        if local_conflict_T.dim() == 4:
            B, L, top_k, D = local_conflict_T.shape
            local_conflict_T = local_conflict_T.view(B, L * top_k, D)
        if local_conflict_A.dim() == 4:
            B, L, top_k, D = local_conflict_A.shape
            local_conflict_A = local_conflict_A.view(B, L * top_k, D)
        if local_conflict_V.dim() == 4:
            B, L, top_k, D = local_conflict_V.shape
            local_conflict_V = local_conflict_V.view(B, L * top_k, D)
        P_T, info_T = self.Local3DSpaceBuilder(local_conflict_T)  # [B,L,3]
        P_A, info_A = self.Local3DSpaceBuilder(local_conflict_A)
        P_V, info_V = self.Local3DSpaceBuilder(local_conflict_V)
        # 将三个模态合并到一个微观局部空间
        P_micro = torch.cat([P_T, P_A, P_V], dim=1)  # [B, L*3, 3]
        # ================== 闭环流程：微观空间 ↔ 冲突空间 =================
        # 1️⃣ 计算结构控制量 G
        G = q_struct  # [B, 5] = (c, k, wT, wA, wV)
        # 2️⃣ 用 G 调整微观局部空间（平移+缩放）
        alpha_G_translation = 0.1  # X轴平移强度
        beta_G_scaling = 0.2  # Y/Z轴缩放强度
        gamma_feedback = 0.1  # 回传到冲突特征空间强度
        B, L3, _ = P_micro.shape
        P_micro_adjusted = P_micro.clone()
        # ===== 安全切片对齐=====
        # G: [B, LG, 5] P_micro_adjusted: [B, L3, 3]
        LG = G.size(1)
        if LG >= L3:
            G_used = G[:, :L3, :]  # [B, L3, 5]
        else:
            # 不够长就补零（极端情况防炸）
            pad_len = L3 - LG
            pad = torch.zeros(G.size(0), pad_len, G.size(2), device=G.device)
            G_used = torch.cat([G, pad], dim=1)  # [B, L3, 5]
        # 平移修正（只用前3维，对应x,y,z）
        P_micro_adjusted[:, :, 0] += alpha_G_translation * G_used[:, :, 0]
        P_micro_adjusted[:, :, 1] += alpha_G_translation * G_used[:, :, 1]
        P_micro_adjusted[:, :, 2] += alpha_G_translation * G_used[:, :, 2]
        # 3️⃣ 将调整后的微观空间回传到冲突特征
        # 取微观空间 Z 轴（扰动）作为回传
        feedback_delta = P_micro_adjusted[:, :, 2]  # [B, L3]
        # 根据每模态长度分片
        len_T = conflict_T.size(1)
        len_A = conflict_A.size(1)
        len_V = conflict_V.size(1)
        # 直接切片并 unsqueeze(-1) 对齐维度
        conflict_T = conflict_T + gamma_feedback * feedback_delta[:, :len_T].unsqueeze(-1)
        conflict_A = conflict_A + gamma_feedback * feedback_delta[:, len_T:len_T + len_A].unsqueeze(-1)
        conflict_V = conflict_V + gamma_feedback * feedback_delta[:, len_T + len_A:len_T + len_A + len_V].unsqueeze(-1)
        # 4️⃣ 基于调整后的冲突特征计算模态冲突贡献度
        q_struct, struct_info = self.TriModalConflictSpace(conflict_T, conflict_A, conflict_V)
        conflict_contrib = torch.stack([struct_info['wT'], struct_info['wA'], struct_info['wV']], dim=-1)  # [B,3]
        main_conflict_modal = conflict_contrib.argmax(dim=-1)  # [B]
        # 5️⃣ 在微观局部空间中定位主要冲突局部特征,这里选择 Z 最大点作为主要冲突局部特征
        max_Z_idx = P_micro_adjusted[:, :, 2].argmax(dim=1)  # [B]
        major_conflict_local = P_micro_adjusted[torch.arange(B), max_Z_idx, :]  # [B,3]
        # ================== 冲突特征放大与算子引导 ==================
        # P_micro_adjusted: [B, L*3, 3]，微观空间 major_conflict_local: [B, 3]，主要冲突局部特征 SCon_T/A/V: [B, D]，当前结构算子冲突部分
        B, L3, _ = P_micro_adjusted.shape
        D = SCon_T.shape[-1]
        # 1️⃣ 扩展 Query 到注意力计算
        if q_struct.dim() == 2:
            q_struct = q_struct.unsqueeze(1)  # [B,1,3]
        query = conflict_modality.unsqueeze(1)   # [B,1,D]
        # 2️⃣ Key/Value: 用 Z/Y轴信息来表示冲突特征能量/扰动
        # 可以简单用三维微观空间作特征为 Value 映射到 D 维度
        local_conflict = torch.cat([local_conflict_T, local_conflict_A, local_conflict_V], dim=1)
        keys = local_conflict  # [B, L*3, D]
        # 3️⃣ 点乘注意力
        # query: [B, 1, D], keys: [B, L*3, D]
        attn_scores = torch.bmm(query, keys.transpose(1, 2)) / (D ** 0.5)  # [B,1,L*3]
        attn_weights = torch.nn.functional.softmax(attn_scores, dim=-1)  # [B,1,L*3]
        # 点乘注意力得到放大冲突特征
        amplified_conflict = torch.bmm(attn_weights, keys)  # [B,1,D]
        amplified_conflict = amplified_conflict.squeeze(1)  # [B,D]
        # 4️⃣ 生成放大冲突特征
        # amplified_conflict: [B, D]
        # 直接广播减去
        # amplified_conflict: [B, L3, D] -> [B, D]
        amplified_conflict_mean = amplified_conflict.mean(dim=1)  # [B, D]
        SCon_T = SCon_T - gamma_feedback * amplified_conflict_mean
        SCon_A = SCon_A - gamma_feedback * amplified_conflict_mean
        SCon_V = SCon_V - gamma_feedback * amplified_conflict_mean
        # 5️⃣ 用放大后的冲突特征弱化结构算子冲突部分
        global_conflict = amplified_conflict.mean(dim=1)  # [B, D] 或 [B]
        SCon_T = SCon_T - gamma_feedback * global_conflict
        global_conflict = amplified_conflict.mean(dim=1)  # [B, D] 或 [B]
        SCon_A = SCon_A - gamma_feedback * global_conflict
        global_conflict = amplified_conflict.mean(dim=1)  # [B, D] 或 [B]
        SCon_V = SCon_V - gamma_feedback * global_conflict
        # ================== 将弱化后的冲突特征放回原模态序列 ==================
        # proj_T/A/V: [B, L, D]
        # conflict_time_idx: [B, num_conflict_steps] -> 每个 batch 的冲突时间步索引
        # amplified_conflict: [B, D] -> 当前 batch 的冲突特征（已弱化）
        B, L, D = t_proj.shape
        # 1️⃣ 生成每个 batch 的冲突时间步索引
        B = t_proj.shape[0]
        conflict_time_idx = [torch.where(conflict_mask[b] > 0)[0] for b in range(B)]
        num_conflict_steps = torch.tensor(
            [len(idx) for idx in conflict_time_idx],
            device=SCon_T.device
        )
        # 扩展 amplified_conflict 到冲突时间步数量
        T = t_proj.size(1)  # 序列长度
        B = amplified_conflict.size(0)  # batch
        D = amplified_conflict.size(-1)
        amplified_expanded_list = []
        for b in range(B):
            k = min(num_conflict_steps[b].item(), amplified_conflict.size(1))  # 确保不超过序列长度
            if k == 0:
                amplified_expanded_list.append(
                    torch.zeros(0, D, device=amplified_conflict.device)
                )
            else:
                # 取前 k 个时间步
                feat = amplified_conflict[b][:k, :]  # [k, D]
                amplified_expanded_list.append(feat)
        # 如果后面必须 tensor 化（例如 concat）
        amplified_expanded = torch.nn.utils.rnn.pad_sequence(
            amplified_expanded_list,
            batch_first=True
        )  # [B, max_k, D]
        import torch.nn.functional as F
        B, T_orig, D = amplified_conflict.size()
        seq_len = t_proj.size(1)
        # 线性插值到 seq_len
        amplified_conflict_resized = torch.nn.functional.interpolate(
            amplified_conflict.permute(0, 2, 1),  # [B, D, T_orig]
            size=seq_len,
            mode='linear',
            align_corners=False
        ).permute(0, 2, 1)  # [B, seq_len, D]
        # === 时间长度 ===
        Lt = t_proj.size(1)  # 50
        La = a_proj.size(1)  # 375
        Lv = v_proj.size(1)  # 500
        # amplified_conflict: [B, Lc, D]  (结构反馈时间序列)
        conflict = amplified_conflict  # [B, Lc, D]
        # === 插值对齐到各模态时间轴 ===
        conflict_T = F.interpolate(
            conflict.permute(0, 2, 1),  # [B, D, Lc]
            size=Lt,
            mode='linear',
            align_corners=False
        ).permute(0, 2, 1)  # [B, Lt, D]
        conflict_A = F.interpolate(
            conflict.permute(0, 2, 1),
            size=La,
            mode='linear',
            align_corners=False
        ).permute(0, 2, 1)  # [B, La, D]
        conflict_V = F.interpolate(
            conflict.permute(0, 2, 1),
            size=Lv,
            mode='linear',
            align_corners=False
        ).permute(0, 2, 1)  # [B, Lv, D]
        # === 结构反馈回填 ===
        t_proj = t_proj - gamma_feedback * conflict_T
        a_proj = a_proj - gamma_feedback * conflict_A
        v_proj = v_proj - gamma_feedback * conflict_V
        # 提取特征后
        t_proj = self.seq_align_text( t_proj)
        a_proj = self.seq_align_audio( a_proj)
        v_proj = self.seq_align_video( v_proj)
        fusion = self.FusionModule(t_proj, v_proj, a_proj)  # [B, L, fusion_dim]融合层
        # 最终预测
        fusion = fusion.mean(dim=1)  # 或 max, 或取最后一个时间步
        output = self.FusionPredictor(fusion)
        return {"M": output}
