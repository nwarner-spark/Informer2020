import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from nbm_bench.models.base_ae import BaseAutoencoder, Prediction
from nbm_bench.models.informer.utils.timefeatures import time_features
from nbm_bench.models.informer.utils.masking import TriangularCausalMask, ProbMask
from nbm_bench.models.informer.encoder import Encoder, EncoderLayer, ConvLayer, EncoderStack
from nbm_bench.models.informer.decoder import Decoder, DecoderLayer
from nbm_bench.models.informer.attn import FullAttention, ProbAttention, AttentionLayer
from nbm_bench.models.informer.embed import DataEmbedding
from nbm_bench.utils import WindowedDataset, collate_seq_len_first
from nbm_bench.models.informer.fnet import FNetEncoder, FNetDecoder

class InformerAE(BaseAutoencoder):
    """
    Using informer for forecasting.
    """
    def __init__(
            self,
            input_dim,
            seq_len=100,
            forc_len=0,
            forc_weight=0,
            d_model=512,
            n_heads=8,
            e_layers=2,
            d_layers=1,
            d_ff=2048,
            dropout=0.05,
            attn='full',
            embed='timeF',
            freq='h',
            activation='gelu',
            kernel_size=3,
            train_config=None,
            encoder_type='attention',
            decoder_type='attention',
    ):
        super().__init__()

        self.input_dim = input_dim
        self.seq_len = seq_len
        self.forc_len = forc_len
        self.forc_weight = forc_weight
        self.d_model = d_model
        self.n_heads = n_heads
        self.e_layers = e_layers
        self.d_layers = d_layers
        self.d_ff = d_ff
        self.dropout = dropout
        self.attn = attn
        self.embed = embed
        self.freq = freq
        self.activation = activation
        self.kernel_size = kernel_size
        self.train_config = train_config
        self.encoder_type = encoder_type
        self.decoder_type= decoder_type

        self.check_forc_params()

        # Offset inside seq_len to start forecasting,
        # this seems to be the most common case.
        self.label_len = self.seq_len

        # Can also be InformerStack
        self.informer = Informer(
            self.input_dim, # enc_in
            self.input_dim, # dec_in
            self.input_dim, # c_out
            self.seq_len,   # seq_len
            self.label_len, # label_len: offset for y_future
            self.forc_len,  # pred_len: number of steps to forecast ahead
            d_model=self.d_model,
            n_heads=self.n_heads,
            e_layers=self.e_layers,
            d_layers=self.d_layers,
            d_ff=self.d_ff,
            dropout=self.dropout,
            attn=self.attn,
            embed=self.embed,
            freq=self.freq,
            activation=self.activation,
            kernel_size=self.kernel_size,
            encoder_type=self.encoder_type,
            decoder_type=self.decoder_type,
        )

    def forward(self, batch):
        x = batch['x']
        x_dt = batch['x_dt']
        y = batch.get('y', None)
        y_dt = batch.get('y_dt', None)

        # x should be [seq_len x batch_size x input_dim]
        assert x.shape[0] == self.seq_len, x.shape
        assert x.shape[2] == self.input_dim, x.shape

        if y is not None:
            # y should be [forc_len x batch_size x input_dim]
            assert y.shape[0] == self.forc_len, y.shape
            assert y.shape[2] == self.input_dim, y.shape

            # informer needs stacked past and future
            y = torch.cat((x, y))
            y_dt = torch.cat((x_dt, y_dt))
        else:
            assert self.forc_len == 0

        # make batch dimension first instead of seq_len
        x = x.permute(1,0,2)
        x_dt = x_dt.permute(1,0,2)
        if y is not None:
            y = y.permute(1,0,2)
            y_dt = y_dt.permute(1,0,2)
        else:
            y_dt = x_dt

        # decoder input
        # This is just seq_len of past `x` with `forc_len` zeros stacked at the end
        device = self.train_config.device
        dec_inp = torch.zeros((x.shape[0], self.forc_len, x.shape[2])).float().to(device)
        dec_inp = torch.cat([x, dec_inp], dim=1).float().to(device)

        if y is not None:
            # original version
            dec_inp0 = torch.zeros_like(y[:, -self.forc_len:, :]).float()
            dec_inp0 = torch.cat([y[:, :self.label_len, :], dec_inp0], dim=1).float().to(device)
            assert np.allclose(dec_inp0.cpu().numpy(), dec_inp.cpu().numpy())

        # note that future data in y is not passed
        # future data in y_dt is passed, but the date features can be known ahead of time
        x_hat, y_hat, attns = self.informer(x, x_dt, dec_inp, y_dt)

        if self.forc_weight < 1:
            loss_rec = nn.functional.mse_loss(x_hat, x)
            x_hat = x_hat.permute(1,0,2)
        else:
            loss_rec = 0
            x_hat = None

        if self.forc_weight > 0:
            y = y[:, -self.forc_len:, :]
            # permuting dims gives same MSE loss
            loss_forc = nn.functional.mse_loss(y_hat, y)
            y_hat = y_hat.permute(1,0,2)
        else:
            loss_forc = 0
            assert y is None
            y_hat = None

        loss = (1 - self.forc_weight) * loss_rec + self.forc_weight * loss_forc

        return x_hat, y_hat, loss

    def get_time_feats(self, date_index):
        assert isinstance(date_index, pd.DatetimeIndex), type(date_index)
        timeenc = 1 if self.embed == 'timeF' else 0
        return time_features(date_index, timeenc=timeenc, freq=self.freq)

    def create_dataloader(self, df, shuffle=False):
        if not isinstance(df, pd.DataFrame):
            raise Exception('Expecting DataFrame')

        # don't add time features globally
        df = df.copy()

        if isinstance(df.index, pd.DatetimeIndex):
            date_index = df.index
        else:
            date_index = pd.date_range('1900-01-01 00:00', periods=len(df), freq='15min')

        t_feats, t_names = self.get_time_feats(date_index)
        t_cols = []
        for i in range(t_feats.shape[1]):
            c = 't_feat_%02d' % i
            t_cols.append(c)
            # assert c not in df.columns
            # if c in df.columns:
            #     assert np.allclose(df[c].values, t_feats[:, i]) 
            df[c] = t_feats[:,i]

        input_cols = [c for c in df.columns if str(c) not in t_cols]

        if self.input_cols is not None:
            assert len(input_cols) == len(self.input_cols)
            assert all([input_cols[i] == self.input_cols[i] for i in range(len(input_cols))])
        else:
            self.input_cols = input_cols

        x = df[input_cols]
        dates = df[t_cols]

        x = torch.tensor(x.values).float()
        dates = torch.tensor(dates.values).float()

        return DataLoader(
            WindowedDataset(x, dates=dates, seq_len=self.seq_len, forc_len=self.forc_len),
            pin_memory=True,
            shuffle=shuffle,
            batch_size=self.train_config.batch_size,
            num_workers=self.train_config.num_workers,
            collate_fn=collate_seq_len_first, # this flips batch and sequence length
        )


class Informer(nn.Module):
    """
    embed: [timeF, embed]
    activation: [relu, gelu] - gelu needs torch>=1.8
    attn: [full, prob] - prob needs torch>=1.8

    Forked from https://github.com/zhouhaoyi/Informer2020 at this commit:
    commit 99b59c181f871a1521a798a37287a27ba5b742a7
    """
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len,
                 factor=5, d_model=512, n_heads=8, e_layers=2, d_layers=1, d_ff=2048,
                 dropout=0.05, attn='prob', embed='timeF', freq='h', activation='gelu',
                 output_attention=False, distil=True, kernel_size=3, encoder_type='attention',
                 decoder_type='attention'):
        super(Informer, self).__init__()

        print('Informer vars',
              enc_in, dec_in, c_out, seq_len, label_len, out_len,
              factor, d_model, n_heads, e_layers, d_layers, d_ff,
              dropout, attn, embed, freq, activation,
              output_attention, distil, encoder_type, decoder_type)

        self.seq_len = seq_len
        self.pred_len = out_len
        self.attn = attn
        self.output_attention = output_attention
        self.encoder_type=encoder_type
        self.decoder_type=decoder_type

        # Encodings
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout, kernel_size=kernel_size)
        self.dec_embedding = DataEmbedding(dec_in, d_model, embed, freq, dropout, kernel_size=kernel_size)

        # Attention type
        Attn = ProbAttention if attn=='prob' else FullAttention

        # Encoder
        self.encoder = None
        if self.encoder_type == 'attention':
            self.encoder = Encoder(
                [
                    EncoderLayer(
                        AttentionLayer(Attn(False, factor,
                                            attention_dropout=dropout, output_attention=output_attention),
                                    d_model, n_heads),
                        d_model,
                        d_ff,
                        dropout=dropout,
                        activation=activation
                    ) for l in range(e_layers)
                ],
                [
                    ConvLayer(
                        d_model
                    ) for l in range(e_layers-1)
                ] if distil else None,
                norm_layer=torch.nn.LayerNorm(d_model)
            )
        else:
            self.encoder = FNetEncoder(
                d_model, e_layers, d_ff, dropout=dropout
            )

        # Decoder
        self.decoder = None
        if self.decoder_type == 'attention':
            self.decoder = Decoder(
                [
                    DecoderLayer(
                        AttentionLayer(Attn(True, factor, attention_dropout=dropout, output_attention=False),
                                    d_model, n_heads),
                        AttentionLayer(FullAttention(False, factor, attention_dropout=dropout,
                                                    output_attention=False),
                                    d_model, n_heads),
                        d_model,
                        d_ff,
                        dropout=dropout,
                        activation=activation,
                    )
                    for l in range(d_layers)
                ],
                norm_layer=torch.nn.LayerNorm(d_model)
            )
        else:
            self.decoder = FNetDecoder(
                d_model, d_layers, d_ff, out_len, dropout=dropout
            )

        self.projection = nn.Linear(d_model, c_out, bias=True)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        dec_out = self.projection(dec_out)

        assert dec_out.shape[1] == self.seq_len + self.pred_len, dec_out.shape
        x_hat = dec_out[:, :self.seq_len, :]
        y_hat = dec_out[:, -self.pred_len:, :]

        return x_hat, y_hat, attns