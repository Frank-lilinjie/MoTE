# --------------------------------------------------------
# References:
# https://github.com/jxhe/unify-parameter-efficient-tuning
# --------------------------------------------------------

import math
import torch
import torch.nn as nn
from timm.models.layers import DropPath
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# MAE: https://github.com/facebookresearch/mae
# --------------------------------------------------------
import timm
from functools import partial
from collections import OrderedDict
import torch
import torch.nn as nn
from timm.models.vision_transformer import PatchEmbed
from timm.models.registry import register_model

import logging
import os
from collections import OrderedDict
import torch
import copy
from backbone.linears import CosineLinear

class Adapter(nn.Module):
    def __init__(self,
                 config=None,
                 d_model=None,
                 bottleneck=None,
                 dropout=0.0,
                 init_option="bert",
                 adapter_scalar="1.0",
                 adapter_layernorm_option="in"):
        super().__init__()
        self.n_embd = config.d_model if d_model is None else d_model
        self.down_size = config.attn_bn if bottleneck is None else bottleneck

        #_before
        self.adapter_layernorm_option = adapter_layernorm_option

        self.adapter_layer_norm_before = None
        if adapter_layernorm_option == "in" or adapter_layernorm_option == "out":
            self.adapter_layer_norm_before = nn.LayerNorm(self.n_embd)

        if adapter_scalar == "learnable_scalar":
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.scale = float(adapter_scalar)

        self.down_proj = nn.Linear(self.n_embd, self.down_size)
        self.non_linear_func = nn.ReLU()
        self.up_proj = nn.Linear(self.down_size, self.n_embd)

        self.dropout = dropout
        if init_option == "bert":
            raise NotImplementedError
        elif init_option == "lora":
            with torch.no_grad():
                nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
                nn.init.zeros_(self.up_proj.weight)
                nn.init.zeros_(self.down_proj.bias)
                nn.init.zeros_(self.up_proj.bias)

    def forward(self, x, add_residual=True, residual=None):
        residual = x if residual is None else residual
        if self.adapter_layernorm_option == 'in':
            x = self.adapter_layer_norm_before(x)

        down = self.down_proj(x)
        down = self.non_linear_func(down)
        down = nn.functional.dropout(down, p=self.dropout, training=self.training)
        up = self.up_proj(down)

        up = up * self.scale

        if self.adapter_layernorm_option == 'out':
            up = self.adapter_layer_norm_before(up)

        if add_residual:
            output = up + residual
        else:
            output = up

        return output


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.,):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(self, x):
        B, N, C = x.shape

        q = self.q_proj(x)
        k = self._shape(self.k_proj(x), -1, B).view(B * self.num_heads, -1, self.head_dim)
        v = self._shape(self.v_proj(x), -1, B).view(B * self.num_heads, -1, self.head_dim)
        q = self._shape(q, N, B).view(B * self.num_heads, -1, self.head_dim)

        # attn = (q @ k.transpose(-2, -1)) * self.scale
        attn_weights = torch.bmm(q, k.transpose(1, 2)) * self.scale

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_probs = self.attn_drop(attn_weights)
        attn_output = torch.bmm(attn_probs, v)

        attn_output = attn_output.view(B, self.num_heads, N, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(B, N, C)

        x = self.proj(attn_output)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, config=None, layer_id=None):
        super().__init__()
        self.config = config
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)

        self.fc1 = nn.Linear(dim, mlp_hidden_dim)
        self.fc2 = nn.Linear(mlp_hidden_dim, dim)
        self.act = act_layer()
        self.mlp_drop = nn.Dropout(drop)

    def forward(self, x, adapt=None):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        if adapt is not None:
            adapt_x = adapt(x, add_residual=False)
        else:
            adapt_x = None
            # print("use PTM backbone without adapter.")

        residual = x
        x = self.mlp_drop(self.act(self.fc1(self.norm2(x))))
        x = self.drop_path(self.mlp_drop(self.fc2(x)))

        if adapt_x is not None:
            if self.config.ffn_adapt:
                if self.config.ffn_option == 'sequential':
                    x = adapt(x)
                elif self.config.ffn_option == 'parallel':
                    x = x + adapt_x
                else:
                    raise ValueError(self.config.ffn_adapt)

        x = residual + x

        return x



class VisionTransformer(nn.Module):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, global_pool=False, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init='', tuning_config=None):
        super().__init__()

        print("I'm using ViT with adapters.")
        self.tuning_config = tuning_config
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer,
                config=tuning_config, layer_id=i,
            )
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Representation layer
        if representation_size and not distilled:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

        # Classifier head(s)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

        # self.init_weights(weight_init)

        ######### MAE begins ############
        self.global_pool = global_pool
        if self.global_pool:
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm

        ######## Adapter begins #########
        if tuning_config.vpt_on:
            assert tuning_config.vpt_num > 0, tuning_config.vpt_num
            # properly registered
            self.embeddings = nn.ParameterList(  # batch, num_prompt, embed_dim
                [nn.Parameter(torch.empty(1, self.tuning_config.vpt_num, embed_dim)) for _ in
                 range(depth)])
            for eee in self.embeddings:
                torch.nn.init.xavier_uniform_(eee.data)
        
        self.config = tuning_config
        self._device = tuning_config._device
        self.proto_list = None
        self.adapter_list = []
        self.init = tuning_config.init
        self.inc = tuning_config.inc
        self.cur_adapter = nn.ModuleList()
        self.get_new_adapter()
        self.multicount = 0
        self.zerocount = 0

    def init_weights(self, mode=''):
        raise NotImplementedError()

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'dist_token'}

    def get_classifier(self):
        if self.dist_token is None:
            return self.head
        else:
            return self.head, self.head_dist           
    
    def update_proto_list(self, num_classes, weight):
        proto_list = CosineLinear(self.embed_dim, num_classes)
        weight = copy.deepcopy(weight)
        proto_list.weight = nn.Parameter(weight)
        del self.proto_list
        self.proto_list = proto_list.requires_grad_(False)
        self.proto_list.to(self._device)


    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        if self.num_tokens == 2:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        
        for i in range(len(self.cur_adapter)):
            self.cur_adapter[i].requires_grad = True
        
    def get_new_adapter(self):
        config = self.config
        self.cur_adapter = nn.ModuleList()
        if config.ffn_adapt:
            for i in range(len(self.blocks)):
                adapter = Adapter(self.config, dropout=0.1, bottleneck=config.ffn_num,
                                        init_option=config.ffn_adapter_init_option,
                                        adapter_scalar=config.ffn_adapter_scalar,
                                        adapter_layernorm_option=config.ffn_adapter_layernorm_option,
                                        ).to(self._device)
                self.cur_adapter.append(adapter)
            self.cur_adapter.requires_grad_(True)
        else:
            print("====Not use adapter===")

    def add_adapter_to_list(self):
        self.adapter_list.append(copy.deepcopy(self.cur_adapter.requires_grad_(False)))
        self.get_new_adapter()
    
    def forward_train(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for idx, blk in enumerate(self.blocks):
            if self.config.vpt_on:
                eee = self.embeddings[idx].expand(B, -1, -1)
                x = torch.cat([eee, x], dim=1)
            x = blk(x, self.cur_adapter[idx])
            if self.config.vpt_on:
                x = x[:, self.config.vpt_num:, :]

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        return outcome

    def forward_test(self, x, use_init_ptm=False):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x_init = self.pos_drop(x)
        
        features = []
        logits = []
        if use_init_ptm:
            x = copy.deepcopy(x_init)
            x = self.blocks(x)
            x = self.norm(x)
            features.append(x)
        expert_num = len(self.adapter_list)
        # 每个旧的adapter的遍历
        for i in range(expert_num):
            x = copy.deepcopy(x_init)
            # 针对每个block
            for j in range(len(self.blocks)): 
                adapt = self.adapter_list[i][j]
                x = self.blocks[j](x, adapt)
            x = self.norm(x)
            cls = x[:, 0, :]
            features.append(cls)
            logit = self.proto_list(cls)["logits"]
            logit = logit * 10
            logits.append(logit)

        # 这里嵌入投票法和专家合并
        features = self.merge_and_reweight(B, features, logits)
        return features


    def vote(self, batch_size, features, logits):
        """
        根据 logits 的置信度选择最合适的专家，并提取该专家输出的特征。
        
        Args:
            batch_size (int): Batch 中样本的数量。
            features (list of tensors): 每个专家输出的特征列表，每个张量形状为 (batch_size, feature_dim)。
            logits (list of tensors): 每个专家输出的 logits 列表，每个张量形状为 (batch_size, num_classes)。
        
        Returns:
            Tensor: 重组后的特征，形状为 (batch_size, feature_dim)。
        """
        # 初始化保存最终选定特征的列表
        final_features = []

        # 动态生成专家任务范围
        expert_ranges = []
        start = 0
        for i in range(len(logits)):  # 遍历当前所有专家
            # 第一个任务类别范围为 [0, self.init)，后续任务每次增加 self.inc 个类别
            end = start + (self.init if i == 0 else self.inc)
            expert_ranges.append(range(start, end))
            start = end

        # 遍历 Batch 中的每个样本

        for i in range(batch_size):
            # 获取当前样本的 logits 和特征
            sample_logits = [logits[exp][i] for exp in range(len(logits))]  # 每个专家的 logits
            sample_features = [features[exp][i] for exp in range(len(logits))]  # 每个专家的特征
            
            # 计算每个专家的预测结果及置信度
            predictions = []
            confidences = []
            for exp in range(len(logits)):
                # 预测类别 (argmax) 和对应的置信度 (softmax 最大值)
                prob = torch.nn.functional.softmax(sample_logits[exp], dim=-1)
                pred = torch.argmax(prob).item()
                confidence = prob[pred].item()
                predictions.append(pred) # 这个是预测结果
                confidences.append(confidence) # 这个是置信度
            
            # 根据任务范围选择最合适的专家 如果有两个专家预测在任务范围内，则选取更高置信度的专家。
            selected_expert = None
            for exp in range(len(logits)):
                # 检查专家预测是否在任务范围内
                if predictions[exp] in expert_ranges[exp]:
                    if selected_expert is None or confidences[exp] > confidences[selected_expert]:
                        selected_expert = exp
            
            # 如果没有专家预测在任务范围内，选择置信度最高的专家
            if selected_expert is None:
                selected_expert = torch.argmax(torch.tensor(confidences)).item()
            
            # 保存选定专家的特征
            final_features.append(sample_features[selected_expert])
        
        # 将最终特征组合成一个张量
        final_features = torch.stack(final_features, dim=0)
        
        return final_features

    def vote_re(self, batch_size, features, logits):
        """
        根据 logits 的置信度选择最合适的专家，并提取该专家输出的特征。
        
        Args:
            batch_size (int): Batch 中样本的数量。
            features (list of tensors): 每个专家输出的特征列表，每个张量形状为 (batch_size, feature_dim)。
            logits (list of tensors): 每个专家输出的 logits 列表，每个张量形状为 (batch_size, num_classes)。
        
        Returns:
            Tensor: 重组后的特征，形状为 (batch_size, feature_dim)。
        """
        # 初始化保存最终选定特征的列表
        final_features = []

        # 动态生成专家任务范围
        expert_ranges = []
        start = 0
        for i in range(len(logits)):  # 遍历当前所有专家
            # 第一个任务类别范围为 [0, self.init)，后续任务每次增加 self.inc 个类别
            end = start + (self.init if i == 0 else self.inc)
            expert_ranges.append(range(start, end))
            start = end

        # 遍历 Batch 中的每个样本
        for i in range(batch_size):
            # 获取当前样本的 logits 和特征
            sample_logits = [logits[exp][i] for exp in range(len(logits))]  # 每个专家的 logits
            sample_features = [features[exp][i] for exp in range(len(logits))]  # 每个专家的特征
            
            # 计算每个专家的预测结果及置信度
            predictions = []
            confidences = []
            for exp in range(len(logits)):
                # 预测类别 (argmax) 和对应的置信度 (softmax 最大值)
                prob = torch.nn.functional.softmax(sample_logits[exp], dim=-1)
                pred = torch.argmax(prob).item()
                confidence = prob[pred].item()
                predictions.append(pred)  # 预测结果
                confidences.append(confidence)  # 置信度
            
            # 找到所有任务范围内的专家
            valid_experts = [
                exp for exp in range(len(logits)) if predictions[exp] in expert_ranges[exp]
            ]
            
            if len(valid_experts) > 1:
                # 如果多个专家预测在任务范围内，进行加权混合
                self.multicount += 1
                weights = []
                max_conf_idx = torch.argmax(torch.tensor([confidences[exp] for exp in valid_experts])).item()
                for idx, exp in enumerate(valid_experts):
                    if idx == max_conf_idx:
                        weights.append(confidences[exp])  # 保持最大置信度专家的权重不变
                    else:
                        weights.append(confidences[exp] * 0.1)  # 其他专家的置信度 * 0.1
                
                # 对权重进行 Softmax 归一化
                weights = torch.nn.functional.softmax(torch.tensor(weights), dim=0)
                
                # 根据权重混合特征
                mixed_feature = torch.stack(
                    [sample_features[exp] * weights[idx] for idx, exp in enumerate(valid_experts)]
                ).sum(dim=0)
                final_features.append(mixed_feature)
            elif len(valid_experts) == 1:
                # 如果只有一个专家在任务范围内，直接选取该专家的特征
                final_features.append(sample_features[valid_experts[0]])
            else:
                # 如果没有专家在任务范围内，选择置信度最高的专家
                self.zerocount += 1
                selected_expert = torch.argmax(torch.tensor(confidences)).item()
                final_features.append(sample_features[selected_expert])
        
        # 将最终特征组合成一个张量
        final_features = torch.stack(final_features, dim=0)
        
        return final_features
    
    def vote_v2(self, batch_size, features, logits):
        """
        根据 logits 和任务范围选择最合适的专家，通过投票和加权混合机制选择最终输出。
        
        Args:
            batch_size (int): Batch 中样本的数量。
            features (list of tensors): 每个专家输出的特征列表，每个张量形状为 (batch_size, feature_dim)。
            logits (list of tensors): 每个专家输出的 logits 列表，每个张量形状为 (batch_size, num_classes)。
            
        Returns:
            Tensor: 重组后的特征，形状为 (batch_size, feature_dim)。
        """
        
        # 初始化保存最终选定特征的列表
        final_features = []

        # 动态生成专家任务范围
        expert_ranges = []
        start = 0
        for i in range(len(logits)):  # 遍历当前所有专家
            # 第一个任务类别范围为 [0, self.init)，后续任务每次增加 self.inc 个类别
            end = start + (self.init if i == 0 else self.inc)
            expert_ranges.append(range(start, end))
            start = end

        # 遍历 Batch 中的每个样本
        for i in range(batch_size):
            # 获取当前样本的 logits 和特征
            sample_logits = [logits[exp][i] for exp in range(len(logits))]  # 每个专家的 logits
            sample_features = [features[exp][i] for exp in range(len(logits))]  # 每个专家的特征
            
            # 计算每个专家的预测结果及置信度
            predictions = []
            confidences = []
            for exp in range(len(logits)):
                # 预测类别 (argmax) 和对应的置信度 (softmax 最大值)
                prob = torch.nn.functional.softmax(sample_logits[exp], dim=-1)
                pred = torch.argmax(prob).item()
                confidence = prob[pred].item()
                predictions.append(pred)
                confidences.append(confidence)
            
            # 找到所有在任务范围内的专家
            valid_experts = [
                exp for exp in range(len(logits)) if predictions[exp] in expert_ranges[exp]
            ]
            
            if len(valid_experts) == 1:
                # 只有一个专家在任务范围内，直接输出该专家的表征
                selected_expert = valid_experts[0]
                final_features.append(sample_features[selected_expert])
            
            elif len(valid_experts) > 1:
                # 多个专家在任务范围内，进行投票和加权混合
                # 选择任务范围内所有专家的类原型
                self.multicount += 1
                prototype_predictions = [predictions[exp] for exp in valid_experts]
                
                # 计算投票得分，其他专家对当前专家的类原型投票
                votes = [0] * len(valid_experts)
                for exp_idx, exp in enumerate(valid_experts):
                    for other_exp in range(len(logits)):
                        if other_exp != exp:
                            # 其他专家对当前专家的预测可信度投票
                            prob = torch.nn.functional.softmax(sample_logits[other_exp], dim=-1)
                            votes[exp_idx] += prob[prototype_predictions[exp_idx]]

                best_expert_idx = torch.argmax(torch.tensor(votes)).item()
                weighted_votes = torch.tensor([votes[idx] * (1 if idx == best_expert_idx else 0.1) for idx in range(len(valid_experts))])
                weights = torch.nn.functional.softmax(weighted_votes, dim=0)  # 使用 softmax 进行归一化
                
                # 加权融合特征
                mixed_feature = torch.stack(
                    [sample_features[exp] * weights[idx] for idx, exp in enumerate(valid_experts)]
                ).sum(dim=0)
                final_features.append(mixed_feature)
            
            else:
                # 没有专家在任务范围内，选择置信度最高的专家
                self.zerocount += 1
                selected_expert = torch.argmax(torch.tensor(confidences)).item()
                final_features.append(sample_features[selected_expert])
        
        # 将最终特征组合成一个张量
        final_features = torch.stack(final_features, dim=0)
        
        return final_features

    def vote_comb(self, batch_size, features, logits):
        """
        将基于置信度和投票法的两种策略结合，选择最优专家的特征。

        Args:
            batch_size (int): Batch 中样本的数量。
            features (list of tensors): 每个专家输出的特征列表，每个张量形状为 (batch_size, feature_dim)。
            logits (list of tensors): 每个专家输出的 logits 列表，每个张量形状为 (batch_size, num_classes)。

        Returns:
            Tensor: 融合后的特征，形状为 (batch_size, feature_dim)。
        """
        # 初始化保存最终选定特征的列表
        final_features = []

        # 动态生成专家任务范围
        expert_ranges = []
        start = 0
        for i in range(len(logits)):  # 遍历当前所有专家
            # 第一个任务类别范围为 [0, self.init)，后续任务每次增加 self.inc 个类别
            end = start + (self.init if i == 0 else self.inc)
            expert_ranges.append(range(start, end))
            start = end

        # 遍历 Batch 中的每个样本
        for i in range(batch_size):
            # 获取当前样本的 logits 和特征
            sample_logits = [logits[exp][i] for exp in range(len(logits))]  # 每个专家的 logits
            sample_features = [features[exp][i] for exp in range(len(logits))]  # 每个专家的特征
            
            # 计算每个专家的预测结果及置信度
            predictions = []
            confidences = []
            for exp in range(len(logits)):
                # 预测类别 (argmax) 和对应的置信度 (softmax 最大值)
                prob = torch.nn.functional.softmax(sample_logits[exp], dim=-1)
                pred = torch.argmax(prob).item()
                confidence = prob[pred].item()
                predictions.append(pred)
                confidences.append(confidence)
            
            # 找到所有在任务范围内的专家
            valid_experts = [
                exp for exp in range(len(logits)) if predictions[exp] in expert_ranges[exp]
            ]

            if len(valid_experts) == 1:
                # 如果只有一个专家在任务范围内，直接选取该专家的特征
                final_features.append(sample_features[valid_experts[0]])
            
            elif len(valid_experts) > 1:
                # 多个专家在任务范围内，使用投票结合置信度加权
                self.multicount += 1
                prototype_predictions = [predictions[exp] for exp in valid_experts]
                
                # 计算投票得分
                votes = [0] * len(valid_experts)
                for exp_idx, exp in enumerate(valid_experts):
                    for other_exp in range(len(logits)):
                        if other_exp != exp:
                            prob = torch.nn.functional.softmax(sample_logits[other_exp], dim=-1)
                            votes[exp_idx] += prob[prototype_predictions[exp_idx]]

                # 根据投票得分和置信度进行加权
                weighted_votes = torch.tensor([
                    votes[idx] + confidences[valid_experts[idx]] for idx in range(len(valid_experts))
                ])
                weights = torch.nn.functional.softmax(weighted_votes, dim=0)  # 使用 softmax 归一化

                # 加权融合特征
                mixed_feature = torch.stack(
                    [sample_features[exp] * weights[idx] for idx, exp in enumerate(valid_experts)]
                ).sum(dim=0)
                final_features.append(mixed_feature)

            else:
                # 如果没有专家在任务范围内，选择置信度最高的专家
                self.zerocount += 1
                selected_expert = torch.argmax(torch.tensor(confidences)).item()
                final_features.append(sample_features[selected_expert])
    
        # 将最终特征组合成一个张量
        final_features = torch.stack(final_features, dim=0)
        
        return final_features


    def vote_re_merge_only(self, batch_size, features, logits):
        """
        只进行专家合并的函数，通过置信度加权合并所有专家的特征。
        
        Args:
            batch_size (int): Batch 中样本的数量。
            features (list of tensors): 每个专家输出的特征列表，每个张量形状为 (batch_size, feature_dim)。
            logits (list of tensors): 每个专家输出的 logits 列表，每个张量形状为 (batch_size, num_classes)。
        
        Returns:
            Tensor: 合并后的特征，形状为 (batch_size, feature_dim)。
        """
        # 初始化保存最终选定特征的列表
        final_features = []

        # 遍历 Batch 中的每个样本
        for i in range(batch_size):
            # 获取当前样本的 logits 和特征
            sample_logits = [logits[exp][i] for exp in range(len(logits))]  # 每个专家的 logits
            sample_features = [features[exp][i] for exp in range(len(logits))]  # 每个专家的特征

            # 计算每个专家的置信度（softmax 最大值）
            confidences = []
            for exp in range(len(logits)):
                prob = torch.nn.functional.softmax(sample_logits[exp], dim=-1)
                confidence = torch.max(prob).item()  # 取 softmax 的最大值作为置信度
                confidences.append(confidence)

            # 对置信度进行 Softmax 归一化，得到权重
            weights = torch.nn.functional.softmax(torch.tensor(confidences), dim=0)

            # 根据权重加权合并特征
            mixed_feature = torch.stack(
                [sample_features[exp] * weights[exp] for exp in range(len(logits))]
            ).sum(dim=0)

            # 将合并后的特征加入最终特征列表
            final_features.append(mixed_feature)

        # 将最终特征组合成一个张量
        final_features = torch.stack(final_features, dim=0)

        return final_features
    
    def vote_re_reweight_only(self, batch_size, features, logits):
        """
        只进行重加权的函数，直接选取置信度最高的专家进行输出。
        
        Args:
            batch_size (int): Batch 中样本的数量。
            features (list of tensors): 每个专家输出的特征列表，每个张量形状为 (batch_size, feature_dim)。
            logits (list of tensors): 每个专家输出的 logits 列表，每个张量形状为 (batch_size, num_classes)。
        
        Returns:
            Tensor: 选取的特征，形状为 (batch_size, feature_dim)。
        """
        # 初始化保存最终选定特征的列表
        final_features = []

        # 遍历 Batch 中的每个样本
        for i in range(batch_size):
            # 获取当前样本的 logits 和特征
            sample_logits = [logits[exp][i] for exp in range(len(logits))]  # 每个专家的 logits
            sample_features = [features[exp][i] for exp in range(len(logits))]  # 每个专家的特征

            # 计算每个专家的置信度（softmax 最大值）
            confidences = []
            for exp in range(len(logits)):
                prob = torch.nn.functional.softmax(sample_logits[exp], dim=-1)
                confidence = torch.max(prob).item()  # 取 softmax 的最大值作为置信度
                confidences.append(confidence)

            # 选取置信度最高的专家
            selected_expert = torch.argmax(torch.tensor(confidences)).item()
            final_features.append(sample_features[selected_expert])

        # 将最终特征组合成一个张量
        final_features = torch.stack(final_features, dim=0)

        return final_features

    def filter_and_merge(self, batch_size, features, logits):
        """
        进行专家过滤后，选择任务范围内的专家并根据置信度进行加权，最后合并其特征。
        
        Args:
            batch_size (int): Batch 中样本的数量。
            features (list of tensors): 每个专家输出的特征列表，每个张量形状为 (batch_size, feature_dim)。
            logits (list of tensors): 每个专家输出的 logits 列表，每个张量形状为 (batch_size, num_classes)。
        
        Returns:
            Tensor: 重组后的特征，形状为 (batch_size, feature_dim)。
        """
        # 初始化保存最终选定特征的列表
        final_features = []

        # 动态生成专家任务范围
        expert_ranges = []
        start = 0
        for i in range(len(logits)):  # 遍历当前所有专家
            # 第一个任务类别范围为 [0, self.init)，后续任务每次增加 self.inc 个类别
            end = start + (self.init if i == 0 else self.inc)
            expert_ranges.append(range(start, end))
            start = end

        # 遍历 Batch 中的每个样本
        for i in range(batch_size):
            # 获取当前样本的 logits 和特征
            sample_logits = [logits[exp][i] for exp in range(len(logits))]  # 每个专家的 logits
            sample_features = [features[exp][i] for exp in range(len(logits))]  # 每个专家的特征
            
            # 计算每个专家的预测结果及置信度
            predictions = []
            confidences = []
            for exp in range(len(logits)):
                # 预测类别 (argmax) 和对应的置信度 (softmax 最大值)
                prob = torch.nn.functional.softmax(sample_logits[exp], dim=-1)
                pred = torch.argmax(prob).item()
                confidence = prob[pred].item()
                predictions.append(pred)  # 预测结果
                confidences.append(confidence)  # 置信度
            
            # 找到任务范围内的专家
            valid_experts = [
                exp for exp in range(len(logits)) if predictions[exp] in expert_ranges[exp]
            ]
            
            if valid_experts:
                # 对任务范围内的专家按置信度加权
                weights = [confidences[exp] for exp in valid_experts]
                # 对权重进行 Softmax 归一化
                weights = torch.nn.functional.softmax(torch.tensor(weights), dim=0)
                
                # 根据权重混合特征
                mixed_feature = torch.stack(
                    [sample_features[exp] * weights[idx] for idx, exp in enumerate(valid_experts)]
                ).sum(dim=0)
                final_features.append(mixed_feature)
            else:
                # 没有专家在任务范围内，选择置信度最高的专家
                self.zerocount += 1
                selected_expert = torch.argmax(torch.tensor(confidences)).item()
                final_features.append(sample_features[selected_expert])
        
        # 将最终特征组合成一个张量
        final_features = torch.stack(final_features, dim=0)
        
        return final_features

    def merge_and_reweight(self, batch_size, features, logits):
        """
        进行专家合并后，根据每个专家的置信度进行重加权，赋予置信度最高的专家更高的权重，
        其余专家的权重乘以 0.1。
        
        Args:
            batch_size (int): Batch 中样本的数量。
            features (list of tensors): 每个专家输出的特征列表，每个张量形状为 (batch_size, feature_dim)。
            logits (list of tensors): 每个专家输出的 logits 列表，每个张量形状为 (batch_size, num_classes)。
        
        Returns:
            Tensor: 重组后的特征，形状为 (batch_size, feature_dim)。
        """
        # 初始化保存最终选定特征的列表
        final_features = []

        # 遍历 Batch 中的每个样本
        for i in range(batch_size):
            # 获取当前样本的 logits 和特征
            sample_logits = [logits[exp][i] for exp in range(len(logits))]  # 每个专家的 logits
            sample_features = [features[exp][i] for exp in range(len(logits))]  # 每个专家的特征
            
            # 计算每个专家的预测结果及置信度
            confidences = []
            for exp in range(len(logits)):
                # 预测类别 (argmax) 和对应的置信度 (softmax 最大值)
                prob = torch.nn.functional.softmax(sample_logits[exp], dim=-1)
                pred = torch.argmax(prob).item()
                confidence = prob[pred].item()
                confidences.append(confidence)  # 置信度
            
            # 获取最大置信度专家的索引
            max_conf_idx = torch.argmax(torch.tensor(confidences)).item()  # 获取最大置信度专家的索引
            
            # 对所有专家的置信度进行加权
            weights = []
            for exp in range(len(logits)):
                if exp == max_conf_idx:
                    weights.append(confidences[exp])  # 保持最大置信度专家的权重不变
                else:
                    weights.append(confidences[exp] * 0.1)  # 其他专家的置信度 * 0.1
            
            weights = torch.tensor(weights)  # 转换为张量
            
            # 对权重进行 Softmax 归一化
            weights = torch.nn.functional.softmax(weights, dim=0)
            
            # 根据加权后的权重合并特征
            mixed_feature = torch.stack(
                [sample_features[exp] * weights[exp] for exp in range(len(logits))]
            ).sum(dim=0)
            
            final_features.append(mixed_feature)
        
        # 将最终特征组合成一个张量
        final_features = torch.stack(final_features, dim=0)
        
        return final_features
    # 这个不需要
    def filter_and_reweight(self, batch_size, features, logits):
        """
        进行专家过滤后，根据每个专家的置信度进行重加权，并选取置信度最高的专家输出其特征。
        
        Args:
            batch_size (int): Batch 中样本的数量。
            features (list of tensors): 每个专家输出的特征列表，每个张量形状为 (batch_size, feature_dim)。
            logits (list of tensors): 每个专家输出的 logits 列表，每个张量形状为 (batch_size, num_classes)。
        
        Returns:
            Tensor: 重组后的特征，形状为 (batch_size, feature_dim)。
        """
        # 初始化保存最终选定特征的列表
        final_features = []

        # 动态生成专家任务范围
        expert_ranges = []
        start = 0
        for i in range(len(logits)):  # 遍历当前所有专家
            # 第一个任务类别范围为 [0, self.init)，后续任务每次增加 self.inc 个类别
            end = start + (self.init if i == 0 else self.inc)
            expert_ranges.append(range(start, end))
            start = end

        # 遍历 Batch 中的每个样本
        for i in range(batch_size):
            # 获取当前样本的 logits 和特征
            sample_logits = [logits[exp][i] for exp in range(len(logits))]  # 每个专家的 logits
            sample_features = [features[exp][i] for exp in range(len(logits))]  # 每个专家的特征
            
            # 计算每个专家的预测结果及置信度
            predictions = []
            confidences = []
            for exp in range(len(logits)):
                # 预测类别 (argmax) 和对应的置信度 (softmax 最大值)
                prob = torch.nn.functional.softmax(sample_logits[exp], dim=-1)
                pred = torch.argmax(prob).item()
                confidence = prob[pred].item()
                predictions.append(pred)  # 预测结果
                confidences.append(confidence)  # 置信度
            
            # 找到所有任务范围内的专家
            valid_experts = [
                exp for exp in range(len(logits)) if predictions[exp] in expert_ranges[exp]
            ]
            
            if len(valid_experts) > 0:
                # 如果有专家在任务范围内，进行重加权
                # 获取最大置信度专家的索引
                max_conf_idx = torch.argmax(torch.tensor([confidences[exp] for exp in valid_experts])).item()
                
                # 对所有有效专家的置信度进行加权
                weights = []
                for exp in valid_experts:
                    if exp == valid_experts[max_conf_idx]:
                        weights.append(confidences[exp])  # 保持最大置信度专家的权重不变
                    else:
                        weights.append(confidences[exp] * 0.1)  # 其他专家的置信度 * 0.1
                
                weights = torch.tensor(weights)  # 转换为张量
                # 归一化权重
                weights = torch.nn.functional.softmax(weights, dim=0)
                
                # 选取置信度最高的专家的特征
                selected_expert = valid_experts[max_conf_idx]
                final_features.append(sample_features[selected_expert])
            else:
                # 如果没有专家在任务范围内，选择置信度最高的专家
                self.zerocount += 1
                selected_expert = torch.argmax(torch.tensor(confidences)).item()
                final_features.append(sample_features[selected_expert])

        # 将最终特征组合成一个张量
        final_features = torch.stack(final_features, dim=0)
        
        return final_features


    def forward(self, x, test=False, use_init_ptm=False):
        if not test:
            output = self.forward_train(x)
        else:
            output = self.forward_test(x, use_init_ptm)
            # output = torch.Tensor().to(features[0].device)
            # for x in features:
            #     cls = x[:, 0, :]
            #     output = torch.cat((
            #         output,
            #         cls
            #     ), dim=1)

        return output

    def forward_proto(self, x, adapt_index):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x_init = self.pos_drop(x)
        
        # the init_PTM's feature
        if adapt_index == -1:
            x = copy.deepcopy(x_init)
            x = self.blocks(x)
            x = self.norm(x)
            output = x[:, 0, :]
            return output

        i = adapt_index
        x = copy.deepcopy(x_init)
        for j in range(len(self.blocks)):
            if i < len(self.adapter_list):
                adapt = self.adapter_list[i][j]
            else:
                adapt = self.cur_adapter[j]
            x = self.blocks[j](x, adapt)
        x = self.norm(x)
        output = x[:, 0, :]
        
        return output
        

def vit_base_patch16_224_mote(pretrained=False, **kwargs):
    
    model = VisionTransformer(patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)

    # checkpoint_model = torch.load('./pretrained_models/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0.npz')
    checkpoint_model=timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=0, checkpoint_path='/pretrains/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_224.npz')
    state_dict = checkpoint_model.state_dict()
    # modify the checkpoint state dict to match the model
    # first, split qkv weight into q, k, v
    for key in list(state_dict.keys()):
        if 'qkv.weight' in key:
            qkv_weight = state_dict.pop(key)
            q_weight = qkv_weight[:768]
            k_weight = qkv_weight[768:768*2]
            v_weight = qkv_weight[768*2:]
            state_dict[key.replace('qkv.weight', 'q_proj.weight')] = q_weight
            state_dict[key.replace('qkv.weight', 'k_proj.weight')] = k_weight
            state_dict[key.replace('qkv.weight', 'v_proj.weight')] = v_weight
        elif 'qkv.bias' in key:
            qkv_bias = state_dict.pop(key)
            q_bias = qkv_bias[:768]
            k_bias = qkv_bias[768:768*2]
            v_bias = qkv_bias[768*2:]
            state_dict[key.replace('qkv.bias', 'q_proj.bias')] = q_bias
            state_dict[key.replace('qkv.bias', 'k_proj.bias')] = k_bias
            state_dict[key.replace('qkv.bias', 'v_proj.bias')] = v_bias
    # second, modify the mlp.fc.weight to match fc.weight
    for key in list(state_dict.keys()):
        if 'mlp.fc' in key:
            fc_weight = state_dict.pop(key)
            state_dict[key.replace('mlp.', '')] = fc_weight

    msg = model.load_state_dict(state_dict, strict=False)
    print(msg)

    # freeze all but the adapter
    for name, p in model.named_parameters():
        if name in msg.missing_keys:
            p.requires_grad = True
        else:
            p.requires_grad = False 
    return model

def vit_base_patch16_224_in21k_mote(pretrained=False, **kwargs):
    
    model = VisionTransformer(patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)

    # checkpoint_model = torch.load('./pretrained_models/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0.npz')
    checkpoint_model=timm.create_model("vit_base_patch16_224_in21k", pretrained=False, num_classes=0, checkpoint_path='/pretrains/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0.npz')
    state_dict = checkpoint_model.state_dict()
    # modify the checkpoint state dict to match the model
    # first, split qkv weight into q, k, v
    for key in list(state_dict.keys()):
        if 'qkv.weight' in key:
            qkv_weight = state_dict.pop(key)
            q_weight = qkv_weight[:768]
            k_weight = qkv_weight[768:768*2]
            v_weight = qkv_weight[768*2:]
            state_dict[key.replace('qkv.weight', 'q_proj.weight')] = q_weight
            state_dict[key.replace('qkv.weight', 'k_proj.weight')] = k_weight
            state_dict[key.replace('qkv.weight', 'v_proj.weight')] = v_weight
        elif 'qkv.bias' in key:
            qkv_bias = state_dict.pop(key)
            q_bias = qkv_bias[:768]
            k_bias = qkv_bias[768:768*2]
            v_bias = qkv_bias[768*2:]
            state_dict[key.replace('qkv.bias', 'q_proj.bias')] = q_bias
            state_dict[key.replace('qkv.bias', 'k_proj.bias')] = k_bias
            state_dict[key.replace('qkv.bias', 'v_proj.bias')] = v_bias
    # second, modify the mlp.fc.weight to match fc.weight
    for key in list(state_dict.keys()):
        if 'mlp.fc' in key:
            fc_weight = state_dict.pop(key)
            state_dict[key.replace('mlp.', '')] = fc_weight

    msg = model.load_state_dict(state_dict, strict=False)
    print(msg)

    # freeze all but the adapter
    for name, p in model.named_parameters():
        if name in msg.missing_keys:
            p.requires_grad = True
        else:
            p.requires_grad = False 
    return model