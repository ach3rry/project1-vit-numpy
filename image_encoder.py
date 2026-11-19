"""Student implementation of the final project's image encoder.

Complete TODO-01 through TODO-06 using NumPy. Model parameters are supplied.
The public classes retain the final project's interface. helpers.py is provided.
"""
import numpy as np
from helpers import LayerNorm, GELU, attention_weights, causal_mask, unit_vector


class MultiHeadAttention:
    def __init__(self, weights, prefix, heads):
        self.qkv_weight = weights[prefix + '.in_proj_weight']  # [3D,D]
        self.qkv_bias = weights[prefix + '.in_proj_bias']      # [3D]
        self.out_weight = weights[prefix + '.out_proj.weight']  # [D,D]
        self.out_bias = weights[prefix + '.out_proj.bias']
        self.heads = heads
        self.width = self.out_bias.shape[0]
        self.head_dim = self.width // heads

    def split_heads(self, x):
        """TODO-02. Input [B,N,D], return [B,H,N,D/H].

        In this image model D=768, H=12 and D/H=64. Use reshape/transpose.
        Research the head organization independently. Preserve token order.
        """
        # TODO-02 contract: [B,N,D] to [B,H,N,D/H].
        # Image model: D=768, H=12, D/H=64.
        B, N, D = x.shape 
        # 768 = 12 * 64：把每个 token 的特征分成 12 个头
        x = x.reshape(B, N, self.heads, self.head_dim)#[B, 50, 768] -> [B, 50, 12, 64]
        # head 轴提到 token 轴前面，每个头都能看到全部 50 个 token
        return x.transpose(0, 2, 1, 3)#[B, 50, 12, 64] -> [B, 12, 50, 64]

    def __call__(self, x, causal=False):
        # Given: stored linear weights use [out,in]. Parameters stay fixed.
        qkv = x @ self.qkv_weight.T + self.qkv_bias #qkv三个权重矩阵拼起来
        q = self.split_heads(qkv[:, :, :self.width])
        k = self.split_heads(qkv[:, :, self.width:2*self.width])
        v = self.split_heads(qkv[:, :, 2*self.width:])

        # TODO-03a: query/key comparisons, scaled by sqrt(head_dim).
        # q,k,v: [B,H,N,64]. scores must have [B,H,N,N].
        k_t = k.transpose(0, 1, 3, 2) # 只换最后两轴，保留 batch 和 head 分组
        scores = q @ k_t #[B, 12, 50, 50]，第 i 个 token 对第 j 个 token 的分数
        # 除以 sqrt(64)=8 缩放；先转 float32，否则整数开方会把结果提升成 float64
        scores = scores / np.sqrt(np.float32(self.head_dim))

        # Given for later text-encoder compatibility. Images use causal=False.
        if causal:
            scores = causal_mask(scores)
        coefficients = attention_weights(scores)  # supplied numerical helper

        # TODO-03b: mix value rows using coefficients; then join head outputs.
        # context: [B,H,N,64], merged: [B,N,D].
        context = coefficients @ v # 每个 head 内按系数对 v 加权求和
        context = context.transpose(0, 2, 1, 3)#换回 [B, 50, 12, 64]
        B, N, D = x.shape
        merged = context.reshape(B, N, D)#12 个头拼回 768 维

        # TODO-03c: output projection, using out_weight and out_bias.
        # Return [B,N,D]. Remember the stored linear-weight orientation.
        output = merged @ self.out_weight.T + self.out_bias
        return output


class TransformerBlock:
    def __init__(self, weights, prefix, heads):
        self.norm1 = LayerNorm(weights[prefix+'.ln_1.weight'], weights[prefix+'.ln_1.bias'])
        self.norm2 = LayerNorm(weights[prefix+'.ln_2.weight'], weights[prefix+'.ln_2.bias'])
        self.attention = MultiHeadAttention(weights, prefix+'.attn', heads)
        self.fc1_weight = weights[prefix+'.mlp.c_fc.weight']  # [4D,D]
        self.fc1_bias = weights[prefix+'.mlp.c_fc.bias']
        self.fc2_weight = weights[prefix+'.mlp.c_proj.weight']  # [D,4D]
        self.fc2_bias = weights[prefix+'.mlp.c_proj.bias']
        self.gelu = GELU()

    def __call__(self, x, causal=False):
        # Given: normalize each token and calculate its attention update.
        update = self.attention(self.norm1(x), causal=causal)

        # TODO-04a: first residual addition. Name the result residual.
        residual = x + update # 第一条残差：注意力输出加回 block 输入

        # Given: the second Norm and the exact checkpoint activation helper.
        u = self.norm2(residual)

        # TODO-04b: MLP = GELU(u @ W1.T + b1) @ W2.T + b2.
        # hidden: [B,N,3072], mlp_update: [B,N,768] for the image tower.
        # Return residual + mlp_update. Both residual paths retain [B,N,D].
        hidden = u @ self.fc1_weight.T + self.fc1_bias # 768 -> 3072
        hidden = self.gelu(hidden)
        mlp_update = hidden @ self.fc2_weight.T + self.fc2_bias # 3072 -> 768
        return residual + mlp_update # 第二条残差


class ImageEncoder:
    def __init__(self, weights):
        self.patch_weight = weights['visual.conv1.weight'].reshape(768, 3072)
        self.cls = weights['visual.class_embedding'].reshape(1, 1, 768)
        self.position = weights['visual.positional_embedding']  # [50,768]
        self.pre_norm = LayerNorm(weights['visual.ln_pre.weight'], weights['visual.ln_pre.bias'])
        self.blocks = [TransformerBlock(weights, 'visual.transformer.resblocks.' + str(i), 12)
                       for i in range(12)]
        self.post_norm = LayerNorm(weights['visual.ln_post.weight'], weights['visual.ln_post.bias'])
        self.projection = weights['visual.proj']  # [768,512], already [in,out]

    def patchify(self, images):
        """TODO-01. [B,3,224,224] -> [B,49,3072].

        Visit patches left to right, then top to bottom. Within a patch, flatten
        C,H,W in that order. Each 32x32 RGB patch contributes 3*32*32 values.
        Use reshape/transpose. Do not resize the already-prepared input.
        """ 
        # Input [B,3,224,224], return [B,49,3072].
        B = images.shape[0]
        # 224 = 7*32：拆成 7x7 网格和块内 32x32 像素
        x = images.reshape(B, 3, 7, 32, 7, 32)
        # 把网格轴提到通道轴前面，顺序为 batch, 行, 列, 通道, 高, 宽
        x = x.transpose(0, 2, 4, 1, 3, 5)
        return x.reshape(B, 49, 3072) # 每个块按 C,H,W 展平

    def embed(self, images):
        patches = self.patchify(images)  # [B,49,3072]

        # TODO-05a: project patches to [B,49,768]. There is no patch bias.
        tokens = patches @ self.patch_weight.T # 3072 -> 768，共享投影
        
        # Given: one supplied CLS row per image.
        cls = np.repeat(self.cls, images.shape[0], axis = 0) # 每个样本一份 CLS

        # TODO-05b: prepend CLS on the token axis, then add learned positions.
        # Return [B,50,768]. self.position broadcasts over B.
        tokens = np.concatenate((cls, tokens), axis = 1) # CLS 放在第 0 位
        tokens += self.position # [50,768] 按 batch 广播
        return tokens

    def encode_tokens(self, images):
        """Provided assembly of token input and the 12 blocks; [B,50,768]."""
        x = self.pre_norm(self.embed(images))
        for block in self.blocks:
            x = block(x)
        return x

    def __call__(self, images):
        """TODO-06. Return one normalized output token [B,512] per image."""
        tokens = self.encode_tokens(images)

        # Select CLS, apply post_norm, and project with self.projection.
        # [B,50,768] -> [B,768] -> [B,512]. visual.proj is already [in,out].
        # Name the projected array output. The final unit-vector helper is given.
        cls = tokens[:, 0, :] # 只取 CLS，形状 [B,768]
        cls = self.post_norm(cls)
        output = cls @ self.projection # visual.proj 已是 [in,out]，不用转置

        return unit_vector(output) # 每行缩放到长度 1
