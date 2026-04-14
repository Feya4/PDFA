# models/pdfa.py
import open_clip
import torch
import torch.nn as nn
#import open_clip
from .film import FiLM
from .asgm import ASGM

class PDFA(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.d = config.d
        self.n_way = config.n_way

        # Load CLIP with correct model name
        self.clip_model, _, _ = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai"      # Use "ViT-B-32"
        )
        self.clip_model.eval()
        for p in self.clip_model.parameters():
            p.requires_grad = False

        self.f_theta = self.clip_model.visual
        self.tokenizer = open_clip.get_tokenizer("ViT-B-32")

        # Learnable prompt tokens
        self.prompt_tokens = nn.Parameter(torch.randn(config.M, self.d) * 0.02)

        self.B_alpha = nn.Sequential(
            nn.LayerNorm(self.d),
            nn.Linear(self.d, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_dim, self.d),
        )

        self.film = FiLM(self.d)
        self.asgm = ASGM(self.d, config.hidden_dim)

        self.Z_mu = nn.Linear(self.d, self.d)
        nn.init.normal_(self.Z_mu.weight, std=0.02)
        nn.init.zeros_(self.Z_mu.bias)
        for p in self.Z_mu.parameters():
            p.requires_grad = False

    def encode_text_with_prompt(self, class_names):
        device = next(self.parameters()).device
        N = len(class_names)

        text_tokens = self.tokenizer(class_names).to(device)   # (N, 77)

        x = self.clip_model.token_embedding(text_tokens)

        # Insert prompt tokens
        prompt = self.prompt_tokens.unsqueeze(0).expand(N, -1, -1)
        x = torch.cat([x[:, :1], prompt, x[:, 1:]], dim=1)

        x = x[:, :77]
        x = x + self.clip_model.positional_embedding[:77].to(x.dtype)

        x = x.permute(1, 0, 2)
        x = self.clip_model.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.clip_model.ln_final(x)

        eos_idx = text_tokens.argmax(dim=-1)
        text_features = x[torch.arange(N), eos_idx]

        return text_features

    def forward(self, support_imgs, query_imgs, support_labels, class_names, return_intermediates=False):
        device = support_imgs.device
        N = self.n_way

        V_s = self.Z_mu(self.f_theta(support_imgs))
        V_q = self.Z_mu(self.f_theta(query_imgs))

        e_i = self.encode_text_with_prompt(class_names)
        X_i = self.B_alpha(e_i) + 0.3 * e_i                     # Residual for stronger semantics

        hat_V_s = self.film(V_s, X_i[support_labels])

        hat_V_q_list = []
        for i in range(N):
            hat_V_q_i = self.film(V_q, X_i[i].unsqueeze(0).expand(V_q.shape[0], -1))
            hat_V_q_list.append(hat_V_q_i)
        hat_V_q = torch.stack(hat_V_q_list, dim=1)

        def fuse(v_mod, x_sem):
            v_norm = nn.functional.instance_norm(v_mod.unsqueeze(0)).squeeze(0)
            x_rep = x_sem.unsqueeze(0).expand(v_mod.shape[0], -1)
            return torch.cat([v_norm, x_rep], dim=1)

        h_s_list = []
        for i in range(N):
            mask = (support_labels == i)
            if mask.sum() > 0:
                h_s_list.append(fuse(hat_V_s[mask], X_i[i]))
        h_s = torch.cat(h_s_list, dim=0)

        h_q_per_class = torch.stack([
            fuse(hat_V_q[:, i], X_i[i]) for i in range(N)
        ], dim=1)

        prototypes = self.asgm.compute_prototypes(h_s, support_labels, N)
        S_star_ii = torch.ones(N, device=device) * 0.8

        logits = self.asgm(h_q_per_class, prototypes, S_star_ii)

        if return_intermediates:
            V_bar_list = []
            for i in range(N):
                mask = (support_labels == i)
                mean_v = V_s[mask].mean(dim=0) if mask.sum() > 0 else torch.zeros(self.d, device=device)
                V_bar_list.append(mean_v)
            V_bar_i = torch.stack(V_bar_list)
            return logits, X_i, V_bar_i, e_i

        return logits