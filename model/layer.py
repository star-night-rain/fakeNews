import torch.nn as nn
import torch
import math
import torch.nn.functional as F


class MaskAttention(torch.nn.Module):
    """
    Compute attention layer
    """

    def __init__(self, input_shape):
        super(MaskAttention, self).__init__()
        self.attention_layer = torch.nn.Linear(input_shape, 1)

    def forward(self, inputs, mask=None):
        scores = self.attention_layer(inputs).view(-1, inputs.size(1))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        scores = torch.softmax(scores, dim=-1).unsqueeze(1)
        outputs = torch.matmul(scores, inputs).squeeze(1)

        return outputs


class Attention(torch.nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """

    def forward(self, query, key, value, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        p_attn = F.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value)


class MultiHeadedAttention(torch.nn.Module):
    """
    Take in model size and number of heads.
    """

    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0

        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h

        self.linear_layers = torch.nn.ModuleList(
            [torch.nn.Linear(d_model, d_model) for _ in range(3)]
        )
        self.output_linear = torch.nn.Linear(d_model, d_model)
        self.attention = Attention()

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)  #
        if mask is not None:
            mask = mask.repeat(1, self.h, 1, 1)
        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [
            l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
            for l, x in zip(self.linear_layers, (query, key, value))
        ]

        # 2) Apply attention on all the projected vectors in batch.
        x = self.attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        return self.output_linear(x)


class FakeNewDetecter(nn.Module):
    def __init__(self, bert, hidden_size=768, num_heads=8):
        super().__init__()
        self.bert = bert

        self.bg_pred = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

        self.iss_pred = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

        self.attention = MultiHeadedAttention(h=num_heads, d_model=hidden_size)
        self.bg_cross_att = MultiHeadedAttention(h=num_heads, d_model=hidden_size)
        self.iss_cross_att = MultiHeadedAttention(h=num_heads, d_model=hidden_size)

        self.bg_score_mapper = nn.Sequential(
            nn.Linear(hidden_size * 2, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        self.iss_score_mapper = nn.Sequential(
            nn.Linear(hidden_size * 2, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        self.aggregator = MaskAttention(hidden_size)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, batch):
        # print(batch)
        text = self.bert(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        ).last_hidden_state

        background = self.bert(
            input_ids=batch["backgrounds_input_ids"],
            attention_mask=batch["backgrounds_attention_mask"],
        ).last_hidden_state

        issues = self.bert(
            input_ids=batch["issues_input_ids"],
            attention_mask=batch["issues_attention_mask"],
        ).last_hidden_state

        # prediction
        # print(background.shape)
        bg_pred = self.bg_pred(torch.mean(background, dim=1))
        iss_pred = self.iss_pred(torch.mean(issues, dim=1))

        # self attention
        ft = self.attention(query=text, key=text, value=text)
        ft = torch.mean(ft, dim=1)

        # cross attention
        # t->b
        ftb = self.bg_cross_att(query=background, key=text, value=text)
        ftb = torch.mean(ftb, dim=1)
        # b->t
        fbt = self.bg_cross_att(query=text, key=background, value=background)
        fbt = torch.mean(fbt, dim=1)

        # t->i
        fti = self.iss_cross_att(query=issues, key=text, value=text)
        fti = torch.mean(fti, dim=1)
        # i->t
        fit = self.iss_cross_att(query=text, key=issues, value=issues)
        fit = torch.mean(fit, dim=1)

        fbt_score = self.bg_score_mapper(torch.cat([fbt, fit], dim=1))
        fbt = fbt_score * fbt

        fit_score = self.iss_score_mapper(torch.cat([fit, fbt], dim=1))
        fit = fit_score * fit
        # concat and aggregate
        x = torch.cat([ft.unsqueeze(1), ftb.unsqueeze(1), fti.unsqueeze(1)], dim=1)

        fusion = self.aggregator(x)
        # classifier
        output = self.classifier(fusion)
        return bg_pred, iss_pred, output
