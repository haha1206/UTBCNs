import torch
import torch.nn as nn
import torch.nn.functional as F
from argumentparser import ArgumentParser
arg = ArgumentParser()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class ScaledDotProductAttention(nn.Module):
    """Scaled dot-product attention mechanism."""

    def __init__(self, attention_dropout=0.0):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, scale=None):

        attention = torch.bmm(q, k.transpose(1, 2))
        if scale:
        	attention = attention * scale

        attention = self.softmax(attention)

        attention = self.dropout(attention)

        context = torch.bmm(attention, v)

        return context, attention

class MultiHeadAttention(nn.Module):

    def __init__(self, model_dim=512, num_heads=8, dropout=0.2):
        super(MultiHeadAttention, self).__init__()

        self.dim_per_head = model_dim // num_heads
        self.num_heads = num_heads
        self.linear_k = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_v = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_q = nn.Linear(model_dim, self.dim_per_head * num_heads)

        self.dot_product_attention = ScaledDotProductAttention(dropout)
        self.linear_final = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(dropout)

        self.layer_norm = nn.LayerNorm(model_dim)
        self._initialize_weights()

    def forward(self, key, value, query):

        residual = query

        dim_per_head = self.dim_per_head
        num_heads = self.num_heads
        batch_size = key.size(0)

        # linear projection
        key = self.linear_k(key)
        value = self.linear_v(value)
        query = self.linear_q(query)

        # split by heads
        key = key.view(batch_size * num_heads, -1, dim_per_head)
        value = value.view(batch_size * num_heads, -1, dim_per_head)
        query = query.view(batch_size * num_heads, -1, dim_per_head)

        # scaled dot product attention
        scale = (key.size(-1) // num_heads) ** -0.5
        context, attention = self.dot_product_attention(
          query, key, value, scale)

        # concat heads
        context = context.view(batch_size, -1, dim_per_head * num_heads)

        # final linear projection
        output = self.linear_final(context)

        # dropout
        output = self.dropout(output)

        # add residual and norm layer
        output = self.layer_norm(residual + output)

        return output, attention
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class PositionalWiseFeedForward(nn.Module):

    def __init__(self, model_dim=512, ffn_dim=2048, dropout=0.2):
        super(PositionalWiseFeedForward, self).__init__()
        self.w1 = nn.Linear(model_dim, ffn_dim)
        self.w2 = nn.Linear(ffn_dim, model_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)
        self._initialize_weights()

    def forward(self, x):
        output = x
        output = self.w2(F.relu(self.w1(output)))
        output = self.dropout(output)

        # add residual and norm layer
        output = self.layer_norm(x + output)
        return output

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

class EncoderLayer(nn.Module):

    def __init__(self, model_dim=512, num_heads=8, ffn_dim=2018, dropout=0.2):

        super(EncoderLayer, self).__init__()

        self.attention = MultiHeadAttention(model_dim, num_heads, dropout)

        self.feed_forward = PositionalWiseFeedForward(model_dim, ffn_dim, dropout)

    def forward(self, inputs):

        # self attention
        context, attention = self.attention(inputs, inputs, inputs)

        # feed forward network
        output = self.feed_forward(context)

        return output, attention

class Encoder(nn.Module):
    def __init__(self,
               num_size,
               num_layers=3,
               model_dim=512,
               num_heads=8,
               ffn_dim=2048,
               dropout=0.2):
        super(Encoder, self).__init__()

        self.encoder_layers = nn.ModuleList(
          [EncoderLayer(model_dim, num_heads, ffn_dim, dropout) for _ in
           range(num_layers)])

        self.seq_embedding = nn.Linear(num_size, model_dim)

    def forward(self, inputs):
        output = self.seq_embedding(inputs)

        attentions = []

        for encoder in self.encoder_layers:
            output, attention = encoder(output)
            attentions.append(attention)

        return output, attentions


class DecoderLayer(nn.Module):

    def __init__(self, model_dim, num_heads=8, ffn_dim=2048, dropout=0.2):
        super(DecoderLayer, self).__init__()

        self.attention = MultiHeadAttention(model_dim, num_heads, dropout)

        self.feed_forward = PositionalWiseFeedForward(model_dim, ffn_dim, dropout)

    def forward(self,
              dec_inputs,
              enc_outputs):
        # self attention, all inputs are decoder inputs
        dec_output, self_attention = self.attention(
          dec_inputs, dec_inputs, dec_inputs)

        # context attention
        # query is decoder's outputs, key and value are encoder's inputs
        dec_output, context_attention = self.attention(
          enc_outputs, enc_outputs, dec_output)

        # decoder's output, or context
        dec_output = self.feed_forward(dec_output)

        return dec_output, self_attention, context_attention


class Decoder(nn.Module):

    def __init__(self,
               num_size,
               num_layers=6,
               model_dim=512,
               num_heads=8,
               ffn_dim=2048,
               dropout=0.2):
        super(Decoder, self).__init__()

        self.num_layers = num_layers

        self.decoder_layers = nn.ModuleList(
          [DecoderLayer(model_dim, num_heads, ffn_dim, dropout) for _ in
           range(num_layers)])

        self.seq_embedding = nn.Linear(num_size, model_dim)

    def forward(self, inputs,data):

        output = self.seq_embedding(data)

        self_attentions = []

        context_attentions = []

        for decoder in self.decoder_layers:
            output, self_attn, context_attn = decoder(output,inputs)
            self_attentions.append(self_attn)
            context_attentions.append(context_attn)

        return output, self_attentions, context_attentions

class Transformer(nn.Module):

    def __init__(self,
               num_size,
               num_layers=6,
               model_dim=512,
               num_heads=8,
               ffn_dim=2048,
               dropout=0.2,
               ):
        super(Transformer, self).__init__()

        self.encoder = Encoder(num_size, num_layers, model_dim,
                               num_heads, ffn_dim, dropout)

        self.decoder = Decoder(num_size,num_layers, model_dim,
                               num_heads, ffn_dim, dropout)

        self.G_ranm = torch.randn(arg.batch_size, arg.window, arg.code_size).normal_(0,1).to(device)

        self.linear = nn.Linear(model_dim, num_size)

        self.linear1 = nn.Linear(arg.window*arg.code_size,arg.code_size)

        self.linear2 = nn.Linear(model_dim,1000)

        self.linear3 = nn.Linear(1000,500)

        self.linear4 = nn.Linear(500,arg.code_size)

        self.linear5 = nn.Linear(arg.code_size,num_size)

        self.dropout = nn.Dropout(dropout)

        self._initialize_weights()


    def forward(self, src_seq):

        output, enc_self_attn = self.encoder(src_seq)

        output1 = self.linear2(output)

        output1 = self.dropout(output1)

        output1 = nn.Tanh()(output1)

        output1 = self.linear3(output1)

        output1 = self.dropout(output1)

        output1 = nn.Tanh()(output1)

        output1 = self.linear4(output1)

        output1 = self.dropout(output1)

        output3 = nn.Tanh()(output1)

        output3 = output3.reshape(arg.batch_size,-1)

        output3 = self.linear1(output3)

        output3 = self.dropout(output3)

        output3 = nn.Tanh()(output3)

        output2 = torch.cat((self.G_ranm,output1),2)

        inputs = self.linear5(output1)

        inputs = self.dropout(inputs)

        output, dec_self_attn, ctx_attn = self.decoder(output,inputs)

        output = self.linear(output)

        output = nn.Tanh()(output)

        return output,output2,output3,output1

    def _initialize_weights(self):
        for m in self.modules():

            if isinstance(m, nn.Linear):

                nn.init.normal_(m.weight, 0, 0.01)

                nn.init.constant_(m.bias, 0)



class Discriminator(nn.Module):
    def __init__(self, len_code=arg.code_size, data_channel=1, nf=64):
        """ Defines the discriminator and encoder model.
        Args:
            len_code: The length of output hash code.
            nf: The unit number of filters. Default is 64.
        Returns:
            None.
        """
        super(Discriminator, self).__init__()
        self.len_code = len_code

        self.nf = nf

        self.network = nn.Sequential(nn.Conv2d(data_channel, nf, 2, 2, 1),
                                     nn.LeakyReLU(2e-1),

                                     nn.Conv2d(nf, nf * 2, 2, 2, 1),
                                     nn.BatchNorm2d(nf * 2),
                                     nn.LeakyReLU(2e-1),

                                     nn.Conv2d(nf * 2, nf * 4, 2, 2, 1),
                                     nn.BatchNorm2d(nf * 4),
                                     nn.LeakyReLU(2e-1),

                                     nn.Conv2d(nf * 4, nf * 8, 2, 2, 1),
                                     nn.BatchNorm2d(nf * 8),
                                     nn.LeakyReLU(2e-1))

        self.discriminate = nn.Linear(8 * nf * 2 * 5, 1)

        self.encode = nn.Linear(8 * nf * 2 * 5, len_code)

        self._initialize_weights()

    def forward(self, x):
        feat = self.network(x)
        feat = feat.view(feat.size(0), -1)
        disc = self.discriminate(feat)
        disc = nn.Sigmoid()(disc)
        code = self.encode(feat)
        code = nn.Sigmoid()(code)
        return disc, code, feat

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            nn.Linear(2*arg.code_size, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256,arg.code_size),
            nn.Tanh()
        )
        self._initialize_weights()

    def forward(self, x):
        x = torch.reshape(x,(arg.batch_size,1,arg.window,-1))
        x = self.gen(x)
        return x
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)