from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel


class TransformerCaptionModel(nn.Module):
    """
    改进版：使用 Transformer 编码器处理图像特征 + Transformer 解码器生成 caption。
    """

    def __init__(
        self,
        bert_name: str = "bert-base-uncased",
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 2,  # 新增：图像编码器层数
        num_decoder_layers: int = 3,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained(bert_name)
        self.bert = BertModel.from_pretrained(bert_name)
        self.d_model = d_model

        # 图像特征投影
        self.img_proj = nn.Linear(2048, d_model)
        # 位置编码（用于图像特征）
        self.img_pos_encoding = nn.Parameter(torch.randn(1, 1000, d_model))  # 最大1000个网格
        
        # 图像 Transformer 编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.img_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        # 文本投影
        self.txt_proj = nn.Linear(self.bert.config.hidden_size, d_model)

        # Transformer 解码器
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        self.fc_out = nn.Linear(d_model, self.bert.config.vocab_size)

    def forward(
        self,
        img_feats: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        use_bert_embedding: bool = True,  # 是否使用 BERT embedding（训练时用，推理时不用）
    ) -> torch.Tensor:
        """
        img_feats: (B, L, 2048) 由图像编码器提供的网格特征
        input_ids, attention_mask: BERT tokenizer 输出
        use_bert_embedding: 如果 False，直接使用 token embedding 而不是 BERT 输出
        """
        if use_bert_embedding:
            with torch.no_grad():
                bert_out = self.bert(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                ).last_hidden_state  # (B, T, hidden)
            tgt = self.txt_proj(bert_out)  # (B, T, d_model)
        else:
            # 直接使用 BERT 的 embedding layer（不经过 BERT 编码器）
            tgt = self.bert.embeddings(input_ids)  # (B, T, hidden)
            tgt = self.txt_proj(tgt)  # (B, T, d_model)

        # 图像特征处理：投影 + 位置编码 + Transformer编码
        B, L, _ = img_feats.shape
        img_proj = self.img_proj(img_feats)  # (B, L, d_model)
        # 添加位置编码
        img_proj = img_proj + self.img_pos_encoding[:, :L, :]
        # 通过Transformer编码器
        memory = self.img_encoder(img_proj)  # (B, L, d_model)

        tgt_mask = nn.Transformer.generate_square_subsequent_mask(
            tgt.size(1),
            device=tgt.device,
        )
        out = self.decoder(
            tgt=tgt,
            memory=memory,
            tgt_mask=tgt_mask,
        )  # (B, T, d_model)

        logits = self.fc_out(out)  # (B, T, vocab_size)
        return logits


def greedy_decode(
    model: TransformerCaptionModel,
    img_feats: torch.Tensor,
    max_len: int = 50,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    使用 greedy search 生成 token 序列。
    返回 (input_ids, attention_mask)
    """
    tokenizer = model.tokenizer
    device = img_feats.device

    input_ids = torch.tensor(
        [[tokenizer.cls_token_id]],
        dtype=torch.long,
        device=device,
    )  # (1, 1)
    attn_mask = torch.ones_like(input_ids, device=device)

    for _ in range(max_len):
        with torch.no_grad():
            # 推理时使用 embedding 而不是 BERT 编码，强制依赖图像
            logits = model(
                img_feats=img_feats,
                input_ids=input_ids,
                attention_mask=attn_mask,
                use_bert_embedding=False,  # 推理时不用 BERT 编码
            )  # (1, T, vocab)
            next_token_logits = logits[:, -1, :]  # (1, vocab)
            next_token_id = next_token_logits.argmax(dim=-1, keepdim=True)  # (1,1)

        input_ids = torch.cat([input_ids, next_token_id], dim=1)
        attn_mask = torch.ones_like(input_ids, device=device)

        if next_token_id.item() == tokenizer.sep_token_id:
            break

    return input_ids, attn_mask


def beam_search_decode(
    model: TransformerCaptionModel,
    img_feats: torch.Tensor,
    beam_size: int = 5,
    max_len: int = 50,
    length_penalty: float = 0.6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    使用 beam search 生成 token 序列（简化版，使用列表存储不同长度的序列）。
    返回 (input_ids, attention_mask)
    """
    tokenizer = model.tokenizer
    device = img_feats.device
    vocab_size = tokenizer.vocab_size
    
    # 初始化：所有beam都以[CLS]开始
    beams = [([tokenizer.cls_token_id], 0.0)]  # [(sequence, score), ...]
    complete_beams = []
    
    for step in range(max_len):
        if len(beams) == 0:
            break
        
        # 准备当前所有beam的输入
        current_size = len(beams)
        input_ids_list = [torch.tensor([seq], dtype=torch.long, device=device) for seq, _ in beams]
        max_seq_len = max(ids.shape[1] for ids in input_ids_list)
        
        # Padding到相同长度
        input_ids_batch = torch.zeros((current_size, max_seq_len), dtype=torch.long, device=device)
        for i, ids in enumerate(input_ids_list):
            input_ids_batch[i, :ids.shape[1]] = ids
        
        img_feats_expanded = img_feats.expand(current_size, -1, -1)
        
        with torch.no_grad():
            attn_mask = (input_ids_batch != 0).long()
            logits = model(
                img_feats=img_feats_expanded,
                input_ids=input_ids_batch,
                attention_mask=attn_mask,
                use_bert_embedding=False,
            )  # (current_size, T, vocab_size)
            
            # 获取最后一个位置的logits
            next_token_logits = logits[:, -1, :]  # (current_size, vocab_size)
            log_probs = F.log_softmax(next_token_logits, dim=-1)  # (current_size, vocab_size)
        
        # 扩展所有beam（只考虑top-k个token，减少计算量）
        candidates = []
        for i, (seq, score) in enumerate(beams):
            # 只考虑top-k个最可能的token（k=beam_size*2，保留一些多样性）
            top_k = min(beam_size * 2, vocab_size)
            top_log_probs, top_indices = torch.topk(log_probs[i], top_k)
            for log_prob, token_id in zip(top_log_probs, top_indices):
                new_seq = seq + [token_id.item()]
                new_score = score + log_prob.item()
                candidates.append((new_seq, new_score))
        
        # 选择top-k
        candidates.sort(key=lambda x: x[1], reverse=True)
        beams = []
        
        for seq, score in candidates[:beam_size]:
            if seq[-1] == tokenizer.sep_token_id:
                # 完成的序列，应用长度惩罚
                seq_len = len(seq)
                final_score = score / (seq_len ** length_penalty)
                complete_beams.append((seq, final_score))
            else:
                beams.append((seq, score))
        
        # 如果完成的beam足够多，提前结束
        if len(complete_beams) >= beam_size:
            break
    
    # 选择最佳序列
    if len(complete_beams) > 0:
        complete_beams.sort(key=lambda x: x[1], reverse=True)
        best_seq = complete_beams[0][0]
    elif len(beams) > 0:
        beams.sort(key=lambda x: x[1], reverse=True)
        best_seq = beams[0][0]
    else:
        best_seq = [tokenizer.cls_token_id]
    
    input_ids = torch.tensor([best_seq], dtype=torch.long, device=device)
    attn_mask = torch.ones_like(input_ids, device=device)
    return input_ids, attn_mask


