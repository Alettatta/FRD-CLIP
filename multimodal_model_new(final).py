import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50
import clip
from transformers import BertModel
import numpy as np

class ECommerceMultimodalModel(nn.Module):
    def __init__(
        self,
        bert_model=None,
        bert_name="bert-base-chinese",
        clip_name="ViT-B/32",
        hidden_dim=256,
        num_classes=2,
        device = 'cpu'
    ):
        super().__init__()
        self.device = device

        # ===== 0. ResNet 视觉增强分支 (冻结) =====
        self.resnet = resnet50(pretrained=True)
        self.resnet.fc = nn.Identity()
        for p in self.resnet.parameters():
            p.requires_grad = False
        resnet_dim = 2048

        # ===== 1. BERT 文本编码器 (冻结) =====
        if bert_model is not None:
            self.bert = bert_model
        else:
            self.bert = BertModel.from_pretrained(bert_name)

        for p in self.bert.parameters():
            p.requires_grad = False
        bert_dim = self.bert.config.hidden_size 

        # ===== 2. CLIP (冻结) =====
        self.clip_model, _ = clip.load(clip_name, device = 'cpu')
        for p in self.clip_model.parameters():
            p.requires_grad = False
        clip_dim = self.clip_model.text_projection.shape[1] 

        # ===== 3. 可学习温度系数 tau =====
        # 初始化为 CLIP 官方建议值 log(1/0.07) ≈ 2.6593
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        # ===== 4. 单模态投影与归一化 =====
        self.proj_text = nn.Linear(bert_dim + clip_dim, hidden_dim)
        self.proj_img  = nn.Linear(resnet_dim + clip_dim, hidden_dim)
        self.proj_clip = nn.Linear(clip_dim + clip_dim, hidden_dim)

        self.ln_text = nn.LayerNorm(hidden_dim)
        self.ln_img  = nn.LayerNorm(hidden_dim)
        self.ln_clip = nn.LayerNorm(hidden_dim)

        # ===== 5. 门控网络 =====
        self.gate_fc = nn.Linear(hidden_dim * 3, 3) 

        # ===== 6. 分类器 =====
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, bert_input_ids, bert_attention_mask, clip_input_ids, image_tensor):
            
        # 1) TextPath
        bert_out = self.bert(input_ids=bert_input_ids, attention_mask=bert_attention_mask)
        bert_cls = bert_out.pooler_output
            
            # --- 修改点 1: 加上 .float() ---
        clip_text = self.clip_model.encode_text(clip_input_ids).float() 
        clip_text = F.normalize(clip_text, dim=-1)

        Ft = self.ln_text(self.proj_text(torch.cat([bert_cls, clip_text], dim=1)))

            # 2) ImagePath
        resnet_vis = self.resnet(image_tensor) # 这是 Float32
            
            # --- 修改点 2: 加上 .float() ---
        clip_vis = self.clip_model.encode_image(image_tensor).float() 
        clip_vis = F.normalize(clip_vis, dim=-1)

        Fi = self.ln_img(self.proj_img(torch.cat([resnet_vis, clip_vis], dim=1)))

            # 3) CLIP 融合
            # --- 这里拼接的是两个被强制转为 float 的变量，所以没问题 ---
        Fm = self.ln_clip(self.proj_clip(torch.cat([clip_text, clip_vis], dim=1)))


            # 4) 应用温度系数的余弦相似度门控
            # logit_scale.exp() 动态调整相似度对结果的影响力
        scale = self.logit_scale.exp()
        sim = F.cosine_similarity(clip_text, clip_vis, dim=1).unsqueeze(1)
        Fm_weighted = (sim * scale) * Fm 

            # 5) 动态三系数门控
        gate_input = torch.cat([Ft, Fi, Fm_weighted], dim=1)
        alpha = F.softmax(self.gate_fc(gate_input), dim=1)

        F_final = (alpha[:, 0:1] * Ft +
                    alpha[:, 1:2] * Fi +
                    alpha[:, 2:3] * Fm_weighted)

            # 6) 分类
        logits = self.classifier(F_final)
        return logits








# 消融实验1
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50
import clip
from transformers import BertModel
import numpy as np

class Ablation1(nn.Module):
    def __init__(
        self,
        bert_model=None,
        bert_name="bert-base-chinese",
        clip_name="ViT-B/32",
        hidden_dim=256,
        num_classes=2,
        device='cpu',
        mode='multimodal',
        debug=False  # 新增调试模式
    ):
        super().__init__()
        self.device = device
        self.mode = mode
        self.debug = debug

        # ===== CLIP (image_only 和多模态都需要) =====
        if mode in ['multimodal', 'image_only']:
            self.clip_model, self.preprocess = clip.load(clip_name, device='cpu')
            
            # 关键：冻结 CLIP，但确保它在 eval 模式（防止 BN/dropout 变化）
            self.clip_model.eval()
            for p in self.clip_model.parameters():
                p.requires_grad = False
            
            clip_dim = 512  # ViT-B/32
            
            # 关键：CLIP 的图像预处理参数（用于检查）
            self.clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
            self.clip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)
        else:
            clip_dim = 0

        # ... [其他初始化与之前相同] ...

        # ===== image_only 架构 =====
        if mode == 'image_only':
            # 更深的投影头，帮助学习
            self.proj_img = nn.Sequential(
                nn.Linear(clip_dim, hidden_dim * 2),
                nn.LayerNorm(hidden_dim * 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim * 2, hidden_dim)
            )
            self.ln_img = nn.LayerNorm(hidden_dim)
            
            # 初始化检查
            self._init_weights()

    def _init_weights(self):
        """更好的初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _check_image_preprocessing(self, image_tensor):
        """检查图像预处理是否正确"""
        if not self.debug:
            return
        
        print(f"\n=== Image Preprocessing Check ===")
        print(f"Shape: {image_tensor.shape}")
        print(f"Dtype: {image_tensor.dtype}")
        print(f"Device: {image_tensor.device}")
        print(f"Range: [{image_tensor.min():.4f}, {image_tensor.max():.4f}]")
        print(f"Mean: {image_tensor.mean():.4f}")
        print(f"Std: {image_tensor.std():.4f}")
        
        # 检查是否已经是 CLIP 预处理后的（范围约 [-3, 3]）
        if image_tensor.min() < -1.0:
            print("⚠️  警告：图像似乎已用 ImageNet normalize，CLIP 期望 [0,1] 范围！")
        elif image_tensor.max() > 1.0 and image_tensor.max() <= 255:
            print("⚠️  警告：图像范围 [0,255]，需要除以 255！")
        elif image_tensor.max() <= 1.0 and image_tensor.min() >= 0:
            print("✅ 图像范围 [0,1]，符合 CLIP 预期")
        
        # 样本检查
        sample = image_tensor[0]
        print(f"Sample channel means: R={sample[0].mean():.3f}, G={sample[1].mean():.3f}, B={sample[2].mean():.3f}")

    def forward(self, bert_input_ids, bert_attention_mask, clip_input_ids, image_tensor):
        
        if self.mode == 'image_only':
            # ========== 关键修复：确保图像预处理正确 ==========
            
            # 调试检查
            self._check_image_preprocessing(image_tensor)
            
            # 关键修复 1：确保值范围正确
            # 如果是 [0, 255]，归一化到 [0, 1]
            if image_tensor.max() > 1.0:
                image_tensor = image_tensor / 255.0
            
            # 关键修复 2：如果是 ImageNet 标准化后的，反标准化回 [0,1]（近似）
            # 或者更简单：直接用 CLIP 的预处理重新处理
            # 这里假设输入是原始像素或 [0,1]
            
            # 关键修复 3：CLIP 期望的预处理
            # CLIP 内部会做一次 Normalize(mean=[0.481..., 0.457..., 0.408...], ...)
            # 所以输入应该是 [0, 1] 范围
            
            with torch.no_grad():
                # 确保在正确的设备上
                if next(self.clip_model.parameters()).device != image_tensor.device:
                    self.clip_model = self.clip_model.to(image_tensor.device)
                
                clip_vis = self.clip_model.encode_image(image_tensor).float()
            
            if self.debug:
                print(f"CLIP output norm: {clip_vis.norm(dim=1).mean():.4f}")
                print(f"CLIP std: {clip_vis.std():.4f}")
                print(f"Has NaN: {torch.isnan(clip_vis).any().item()}")
            
            # 归一化 CLIP 特征
            clip_vis = F.normalize(clip_vis, dim=-1)
            
            # 投影
            Fi = self.ln_img(self.proj_img(clip_vis))
            
            if self.debug:
                print(f"Projected feature norm: {Fi.norm(dim=1).mean():.4f}")
            
            logits = self.classifier(Fi)
            
            if self.debug:
                print(f"Logits range: [{logits.min():.4f}, {logits.max():.4f}]")
                print(f"Logits mean: {logits.mean():.4f}")
            
            return logits
