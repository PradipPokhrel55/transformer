import pandas as pd
import numpy as np

def softmax(x, axis=-1):
    e_x = np.exp(x-np.max(x,axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


class Embedding:
    def __init__(self, vocab_size, d_model):
        self.weight = np.random.randn(vocab_size, d_model) * 0.01

    def __call__(self, x):
        return self.weight[x]
    

class MultiHeadAttention:
    def __init__(self, d_model, num_heads):
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        self.Wq = np.random.randn(d_model, d_model) * 0.01
        self.Wk = np.random.randn(d_model, d_model) * 0.01
        self.Wv = np.random.randn(d_model, d_model) * 0.01
        self.Wo = np.random.randn(d_model, d_model) * 0.01

    def split_heads(self, x):
        batch, seq_len, _ = x.shape
        x = x.reshape(batch, seq_len, self.num_heads, self.d_head)
        return x.transpose(0,2,1,3)
    
    def scaled_dot_product_attention(self, Q, K, V):
        scores = np.matmul(Q, K.tarnspose(0,1,3,2)) / np.sqrt(self.d_head)
        weights = softmax(scores, axis=-1)
        output = np.matmul(weights, V)
        return output, weights
    
    def __call__(self, x):
        Q = np.matmul(x, self.Wq)
        K = np.matmul(x, self.Wk)
        V = np.matmul(x, self.Wv)

        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)

        out, weights = self.scaled_dot_product_attention(Q,K,V)

        out = out.transpose(0,2,1,3).reshape(x.shape)

        out = np.matmul(out, self.Wo)

        return out, weights
    

class FeedForward:
    def __init__(self,d_model, d_ff):
        self.W1 = np.random.randn(d_model, d_ff) * 0.01
        self.b1 = np.zeros(d_ff)
        self.W2 = np.random.randn(d_ff, d_model) * 0.01
        self.b2 = np.zeros(d_model)

    def __call__(self, x):
        x = np.matmul(x, self.W1) + self.b1
        x = np.maximum(0,x)
        x = np.matmul(x, self.W2) + self.b2
        return x
    

class EncoderBlock:
    def __init__(self, d_model, num_heads, d_ff):
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff)
        self.eps = 1e-6

    def layer_norm(self, x):
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        return (x-mean) / np.sqrt(var + self.eps)
    
    def __call__(self, x):
        attn_out, attn_weights = self.mha(x)
        x = self.layer_norm(x + attn_out)

        ffn_out = self.ffn(x)
        x = self.layer_norm(x + ffn_out)

        return x, attn_weights
    

class Encoder:
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers):
        self.embedding = Embedding(vocab_size, d_model)
        self.layers = [EncoderBlock(d_model, num_heads, d_ff) for _ in range(num_layers)]
        self.d_model = d_model
        self.vocab_size = vocab_size

    def __call__(self, x):
        x = self.embedding
        all_attn = []

        for layer in self.layers:
            x, attn = layer(x)
            all_attn.append(attn)

        return x, all_attn
    

    

