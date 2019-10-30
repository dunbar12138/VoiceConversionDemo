from pathlib import Path

import imageio
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import itertools

from scipy.fftpack import dct
from tensorboardX import SummaryWriter
from torch.autograd import Variable

from audioUtils.audio import wav2seg, inv_preemphasis, preemphasis
from model_video import VideoGenerator, STAGE2_G
from audioUtils.hparams import hparams
from audioUtils import audio
from vocoder.models.fatchord_version import WaveRNN
import cv2

_inv_mel_basis = np.linalg.pinv(audio._build_mel_basis(hparams))
mel_basis = librosa.filters.mel(hparams.sample_rate, hparams.n_fft, n_mels=40)

class LinearNorm(torch.nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super(LinearNorm, self).__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)

        torch.nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        return self.linear_layer(x)


class ConvNorm(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=None, dilation=1, bias=True, w_init_gain='linear'):
        super(ConvNorm, self).__init__()
        if padding is None:
            assert(kernel_size % 2 == 1)
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = torch.nn.Conv1d(in_channels, out_channels,
                                    kernel_size=kernel_size, stride=stride,
                                    padding=padding, dilation=dilation,
                                    bias=bias)

        torch.nn.init.xavier_uniform_(
            self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, signal):
        conv_signal = self.conv(signal)
        return conv_signal


class Encoder(nn.Module):
    """Encoder module:
    """
    def __init__(self, dim_neck, dim_emb, freq):
        super(Encoder, self).__init__()
        self.dim_neck = dim_neck
        self.freq = freq
        
        convolutions = []
        for i in range(3):
            conv_layer = nn.Sequential(
                ConvNorm(80+dim_emb if i==0 else 512,
                         512,
                         kernel_size=5, stride=1,
                         padding=2,
                         dilation=1, w_init_gain='relu'),
                nn.BatchNorm1d(512))
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)
        
        self.lstm = nn.LSTM(512, dim_neck, 2, batch_first=True, bidirectional=True)

    def forward(self, x, c_org):
        # (B, T, n_mel)
        x = x.squeeze(1).transpose(2,1)
        c_org = c_org.unsqueeze(-1).expand(-1, -1, x.size(-1))
        x = torch.cat((x, c_org), dim=1)
        
        for conv in self.convolutions:
            x = F.relu(conv(x))
        x = x.transpose(1, 2)
        
        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)
        out_forward = outputs[:, :, :self.dim_neck]
        out_backward = outputs[:, :, self.dim_neck:]

        # print(outputs.shape)

        codes = []
        for i in range(0, outputs.size(1), self.freq):
            codes.append(torch.cat((out_forward[:,i+self.freq-1,:],out_backward[:,i,:]), dim=-1))

        return codes


class MyEncoder(nn.Module):
    '''Encoder without speaker embedding'''

    def __init__(self, dim_neck, freq, num_mel=80):
        super(MyEncoder, self).__init__()
        self.dim_neck = dim_neck
        self.freq = freq

        convolutions = []
        for i in range(3):
            conv_layer = nn.Sequential(
                ConvNorm(num_mel if i == 0 else 512,
                         512,
                         kernel_size=5, stride=1,
                         padding=2,
                         dilation=1, w_init_gain='relu'),
                nn.BatchNorm1d(512))
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)

        self.lstm = nn.LSTM(512, dim_neck, 2, batch_first=True, bidirectional=True)

    def forward(self, x, c_org=None, return_unsample=False):
        # (B, T, n_mel)
        x = x.squeeze(1).transpose(2, 1)

        for conv in self.convolutions:
            x = F.relu(conv(x))
        x = x.transpose(1, 2)

        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)
        out_forward = outputs[:, :, :self.dim_neck]
        out_backward = outputs[:, :, self.dim_neck:]

        # print(outputs.shape)

        codes = []
        for i in range(0, outputs.size(1), self.freq):
            codes.append(torch.cat((out_forward[:, i + self.freq - 1, :], out_backward[:, i, :]), dim=-1))
        if return_unsample:
            return codes, outputs
        return codes

class MultiEncoder(nn.Module):
    def __init__(self, num_encoders, dim_neck, freq, fully_separate=True):
        super(MultiEncoder, self).__init__()
        if fully_separate:
            self.encoders = nn.ModuleList([MyEncoder(dim_neck, freq) for i in range(num_encoders)])
        else:
            self.encoder = MyEncoder(dim_neck*num_encoders, freq)
        self.fully_separate = fully_separate
        self.dim_neck = dim_neck
        self.num_encoders = num_encoders

    def forward(self, x, c_org):
        index = c_org[0].max(0)[1]
        # print(index)
        if self.fully_separate:
            return self.encoders[index](x, c_org)
        else:
            mid = self.dim_neck * self.num_encoders
            return [torch.cat((code[:, index*self.dim_neck:(index+1)*self.dim_neck],
                              code[:, mid+index*self.dim_neck:mid+(index+1)*self.dim_neck]), dim=-1)
                    for code in self.encoder(x, c_org)]


class Decoder(nn.Module):
    """Decoder module:
    """
    def __init__(self, dim_neck, dim_emb, dim_pre, num_mel=80):
        super(Decoder, self).__init__()
        
        self.lstm1 = nn.LSTM(dim_neck*2+dim_emb, dim_pre, 1, batch_first=True)
        
        convolutions = []
        for i in range(3):
            conv_layer = nn.Sequential(
                ConvNorm(dim_pre,
                         dim_pre,
                         kernel_size=5, stride=1,
                         padding=2,
                         dilation=1, w_init_gain='relu'),
                nn.BatchNorm1d(dim_pre))
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)
        
        self.lstm2 = nn.LSTM(dim_pre, 1024, 2, batch_first=True)
        
        self.linear_projection = LinearNorm(1024, num_mel)

    def forward(self, x):
        
        #self.lstm1.flatten_parameters()
        x, _ = self.lstm1(x)
        x = x.transpose(1, 2)
        
        for conv in self.convolutions:
            x = F.relu(conv(x))
        x = x.transpose(1, 2)
        
        outputs, _ = self.lstm2(x)
        
        decoder_output = self.linear_projection(outputs)

        return decoder_output   


class MultiDecoder(nn.Module):
    def __init__(self, num_decoders, dim_neck, dim_pre, multigpu):
        super(MultiDecoder, self).__init__()

        self.decoders = [Decoder(dim_neck, 0, dim_pre) for i in range(num_decoders)]
        #
        # if multigpu:
        #     self.decoders = [nn.DataParallel(decoder.cuda()) for decoder in self.decoders]
        self.decoders = nn.ModuleList(self.decoders)

    def forward(self, x, speaker_onehot):
        index = speaker_onehot[0].max(0)[1]
        # print(index)
        return self.decoders[index](x)



class LocationLayer(nn.Module):
    def __init__(self, attention_n_filters, attention_kernel_size,
                 attention_dim):
        super(LocationLayer, self).__init__()
        padding = int((attention_kernel_size - 1) / 2)
        self.location_conv = ConvNorm(2, attention_n_filters,
                                      kernel_size=attention_kernel_size,
                                      padding=padding, bias=False, stride=1,
                                      dilation=1)
        self.location_dense = LinearNorm(attention_n_filters, attention_dim,
                                         bias=False, w_init_gain='tanh')

    def forward(self, attention_weights_cat):
        processed_attention = self.location_conv(attention_weights_cat)
        processed_attention = processed_attention.transpose(1, 2)
        processed_attention = self.location_dense(processed_attention)
        return processed_attention


class Attention(nn.Module):
    def __init__(self, attention_rnn_dim, embedding_dim, attention_dim,
                 attention_location_n_filters, attention_location_kernel_size):
        super(Attention, self).__init__()
        self.query_layer = LinearNorm(attention_rnn_dim, attention_dim,
                                      bias=False, w_init_gain='tanh')
        self.memory_layer = LinearNorm(embedding_dim, attention_dim, bias=False,
                                       w_init_gain='tanh')
        self.v = LinearNorm(attention_dim, 1, bias=False)
        self.location_layer = LocationLayer(attention_location_n_filters,
                                            attention_location_kernel_size,
                                            attention_dim)
        self.score_mask_value = -float("inf")

    def get_alignment_energies(self, query, processed_memory,
                               attention_weights_cat):
        """
        PARAMS
        ------
        query: decoder output (batch, n_mel_channels * n_frames_per_step)
        processed_memory: processed encoder outputs (B, T_in, attention_dim)
        attention_weights_cat: cumulative and prev. att weights (B, 2, max_time)

        RETURNS
        -------
        alignment (batch, max_time)
        """

        processed_query = self.query_layer(query.unsqueeze(1))
        processed_attention_weights = self.location_layer(attention_weights_cat)
        energies = self.v(torch.tanh(
            processed_query + processed_attention_weights + processed_memory))

        energies = energies.squeeze(-1)
        return energies

    def forward(self, attention_hidden_state, memory, processed_memory,
                attention_weights_cat, mask):
        """
        PARAMS
        ------
        attention_hidden_state: attention rnn last output
        memory: encoder outputs
        processed_memory: processed encoder outputs
        attention_weights_cat: previous and cummulative attention weights
        mask: binary mask for padded data
        """
        alignment = self.get_alignment_energies(
            attention_hidden_state, processed_memory, attention_weights_cat)

        if mask is not None:
            alignment.data.masked_fill_(mask, self.score_mask_value)

        attention_weights = F.softmax(alignment, dim=1)
        attention_context = torch.bmm(attention_weights.unsqueeze(1), memory)
        attention_context = attention_context.squeeze(1)

        return attention_context, attention_weights


class Prenet(nn.Module):
    def __init__(self, in_dim, sizes):
        super(Prenet, self).__init__()
        in_sizes = [in_dim] + sizes[:-1]
        self.layers = nn.ModuleList(
            [LinearNorm(in_size, out_size, bias=False)
             for (in_size, out_size) in zip(in_sizes, sizes)])

    def forward(self, x):
        for linear in self.layers:
            x = F.dropout(F.relu(linear(x)), p=0.5, training=True)
        return x

class TacotronDecoder(nn.Module):
    def __init__(self, hparams):
        super(TacotronDecoder, self).__init__()
        self.n_mel_channels = hparams.num_mels
        self.n_frames_per_step = hparams.outputs_per_step
        self.encoder_embedding_dim = hparams.speaker_embedding_size + 2 * hparams.dim_neck
        self.attention_rnn_dim = hparams.attention_rnn_dim
        self.decoder_rnn_dim = hparams.decoder_lstm_units
        self.prenet_dim = hparams.prenet_dim
        self.max_decoder_steps = hparams.max_iters
        # self.gate_threshold = hparams.gate_threshold
        self.p_attention_dropout = hparams.p_attention_dropout
        self.p_decoder_dropout = hparams.p_decoder_dropout

        self.prenet = Prenet(
            hparams.num_mels * hparams.outputs_per_step,
            [hparams.prenet_dim, hparams.prenet_dim])

        self.attention_rnn = nn.LSTMCell(
            hparams.prenet_dim + hparams.speaker_embedding_size + 2 * hparams.dim_neck,
            hparams.attention_rnn_dim)

        self.attention_layer = Attention(
            hparams.attention_rnn_dim, hparams.speaker_embedding_size + 2 * hparams.dim_neck,
            hparams.attention_dim, hparams.attention_filters,
            hparams.attention_kernel[0])

        self.decoder_rnn = nn.LSTMCell(
            hparams.attention_rnn_dim + hparams.speaker_embedding_size + 2 * hparams.dim_neck,
            hparams.decoder_lstm_units, 1)

        self.linear_projection = LinearNorm(
            hparams.decoder_lstm_units + hparams.speaker_embedding_size + 2 * hparams.dim_neck,
            hparams.num_mels * hparams.outputs_per_step)

        self.gate_layer = LinearNorm(
            hparams.decoder_lstm_units + hparams.speaker_embedding_size + 2 * hparams.dim_neck, 1,
            bias=True, w_init_gain='sigmoid')

    def get_go_frame(self, memory):
        """ Gets all zeros frames to use as first decoder input
        PARAMS
        ------
        memory: decoder outputs

        RETURNS
        -------
        decoder_input: all zeros frames
        """
        B = memory.size(0)
        decoder_input = Variable(memory.data.new(
            B, self.n_mel_channels * self.n_frames_per_step).zero_())
        return decoder_input

    def initialize_decoder_states(self, memory, mask):
        """ Initializes attention rnn states, decoder rnn states, attention
        weights, attention cumulative weights, attention context, stores memory
        and stores processed memory
        PARAMS
        ------
        memory: Encoder outputs
        mask: Mask for padded data if training, expects None for inference
        """
        B = memory.size(0)
        MAX_TIME = memory.size(1)

        self.attention_hidden = Variable(memory.data.new(
            B, self.attention_rnn_dim).zero_())
        self.attention_cell = Variable(memory.data.new(
            B, self.attention_rnn_dim).zero_())

        self.decoder_hidden = Variable(memory.data.new(
            B, self.decoder_rnn_dim).zero_())
        self.decoder_cell = Variable(memory.data.new(
            B, self.decoder_rnn_dim).zero_())

        self.attention_weights = Variable(memory.data.new(
            B, MAX_TIME).zero_())
        self.attention_weights_cum = Variable(memory.data.new(
            B, MAX_TIME).zero_())
        self.attention_context = Variable(memory.data.new(
            B, self.encoder_embedding_dim).zero_())

        self.memory = memory
        self.processed_memory = self.attention_layer.memory_layer(memory)
        self.mask = mask

    def parse_decoder_inputs(self, decoder_inputs):
        """ Prepares decoder inputs, i.e. mel outputs
        PARAMS
        ------
        decoder_inputs: inputs used for teacher-forced training, i.e. mel-specs

        RETURNS
        -------
        inputs: processed decoder inputs

        """
        # (B, n_mel_channels, T_out) -> (B, T_out, n_mel_channels)
        decoder_inputs = decoder_inputs.transpose(1, 2)
        decoder_inputs = decoder_inputs.view(
            decoder_inputs.size(0),
            int(decoder_inputs.size(1)/self.n_frames_per_step), -1)
        # (B, T_out, n_mel_channels) -> (T_out, B, n_mel_channels)
        decoder_inputs = decoder_inputs.transpose(0, 1)
        return decoder_inputs

    def parse_decoder_outputs(self, mel_outputs, alignments):
        """ Prepares decoder outputs for output
        PARAMS
        ------
        mel_outputs:
        gate_outputs: gate output energies
        alignments:

        RETURNS
        -------
        mel_outputs:
        gate_outpust: gate output energies
        alignments:
        """
        # (T_out, B) -> (B, T_out)
        alignments = torch.stack(alignments).transpose(0, 1)
        # (T_out, B) -> (B, T_out)
        # gate_outputs = torch.stack(gate_outputs).transpose(0, 1)
        # gate_outputs = gate_outputs.contiguous()
        # (T_out, B, n_mel_channels) -> (B, T_out, n_mel_channels)
        mel_outputs = torch.stack(mel_outputs).transpose(0, 1).contiguous()
        # decouple frames per step
        mel_outputs = mel_outputs.view(
            mel_outputs.size(0), -1, self.n_mel_channels)
        # (B, T_out, n_mel_channels) -> (B, n_mel_channels, T_out)
        mel_outputs = mel_outputs.transpose(1, 2)

        return mel_outputs, alignments

    def decode(self, decoder_input):
        """ Decoder step using stored states, attention and memory
        PARAMS
        ------
        decoder_input: previous mel output

        RETURNS
        -------
        mel_output:
        gate_output: gate output energies
        attention_weights:
        """
        cell_input = torch.cat((decoder_input, self.attention_context), -1)
        self.attention_hidden, self.attention_cell = self.attention_rnn(
            cell_input, (self.attention_hidden, self.attention_cell))
        self.attention_hidden = F.dropout(
            self.attention_hidden, self.p_attention_dropout, self.training)

        attention_weights_cat = torch.cat(
            (self.attention_weights.unsqueeze(1),
             self.attention_weights_cum.unsqueeze(1)), dim=1)
        self.attention_context, self.attention_weights = self.attention_layer(
            self.attention_hidden, self.memory, self.processed_memory,
            attention_weights_cat, self.mask)

        self.attention_weights_cum += self.attention_weights
        decoder_input = torch.cat(
            (self.attention_hidden, self.attention_context), -1)
        self.decoder_hidden, self.decoder_cell = self.decoder_rnn(
            decoder_input, (self.decoder_hidden, self.decoder_cell))
        self.decoder_hidden = F.dropout(
            self.decoder_hidden, self.p_decoder_dropout, self.training)

        decoder_hidden_attention_context = torch.cat(
            (self.decoder_hidden, self.attention_context), dim=1)
        decoder_output = self.linear_projection(
            decoder_hidden_attention_context)

        gate_prediction = self.gate_layer(decoder_hidden_attention_context)
        return decoder_output, gate_prediction, self.attention_weights

    def forward(self, memory, decoder_inputs, mask):
        """ Decoder forward pass for training
        PARAMS
        ------
        memory: Encoder outputs (B, T_out, feature_dim)
        decoder_inputs: Decoder inputs for teacher forcing. i.e. mel-specs (B, n_mel_channels, T_out)
        memory_lengths: Encoder output lengths for attention masking.

        RETURNS
        -------
        mel_outputs: mel outputs from the decoder
        gate_outputs: gate outputs from the decoder
        alignments: sequence of attention weights from the decoder
        """
        decoder_input = self.get_go_frame(memory).unsqueeze(0)
        decoder_inputs = self.parse_decoder_inputs(decoder_inputs)
        decoder_inputs = torch.cat((decoder_input, decoder_inputs), dim=0)
        decoder_inputs = self.prenet(decoder_inputs)

        self.initialize_decoder_states(
            memory, mask=~mask)

        mel_outputs, gate_outputs, alignments = [], [], []
        while len(mel_outputs) < decoder_inputs.size(0) - 1:
            decoder_input = decoder_inputs[len(mel_outputs)]
            mel_output, gate_output, attention_weights = self.decode(
                decoder_input)
            mel_outputs += [mel_output.squeeze(1)]
            gate_outputs += [gate_output.squeeze()]
            alignments += [attention_weights]

        mel_outputs, alignments = self.parse_decoder_outputs(mel_outputs, alignments)

        return mel_outputs, gate_outputs, alignments

    def inference(self, memory):
        """ Decoder inference
        PARAMS
        ------
        memory: Encoder outputs

        RETURNS
        -------
        mel_outputs: mel outputs from the decoder
        gate_outputs: gate outputs from the decoder
        alignments: sequence of attention weights from the decoder
        """
        decoder_input = self.get_go_frame(memory)

        self.initialize_decoder_states(memory, mask=None)

        mel_outputs, gate_outputs, alignments = [], [], []
        while True:
            decoder_input = self.prenet(decoder_input)
            mel_output, gate_output, alignment = self.decode(decoder_input)

            mel_outputs += [mel_output.squeeze(1)]
            gate_outputs += [gate_output]
            alignments += [alignment]

            if len(mel_outputs) == memory.size(1):
                # print("Warning! Reached max decoder steps")
                break

            decoder_input = mel_output

        mel_outputs, alignments = self.parse_decoder_outputs(mel_outputs, alignments)

        return mel_outputs, gate_outputs, alignments

    
class Postnet(nn.Module):
    """Postnet
        - Five 1-d convolution with 512 channels and kernel size 5
    """

    def __init__(self, num_mel=80):
        super(Postnet, self).__init__()
        self.convolutions = nn.ModuleList()

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(num_mel, 512,
                         kernel_size=5, stride=1,
                         padding=2,
                         dilation=1, w_init_gain='tanh'),
                nn.BatchNorm1d(512))
        )

        for i in range(1, 5 - 1):
            self.convolutions.append(
                nn.Sequential(
                    ConvNorm(512,
                             512,
                             kernel_size=5, stride=1,
                             padding=2,
                             dilation=1, w_init_gain='tanh'),
                    nn.BatchNorm1d(512))
            )

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(512, num_mel,
                         kernel_size=5, stride=1,
                         padding=2,
                         dilation=1, w_init_gain='linear'),
                nn.BatchNorm1d(num_mel))
            )

    def forward(self, x):
        for i in range(len(self.convolutions) - 1):
            x = torch.tanh(self.convolutions[i](x))

        x = self.convolutions[-1](x)

        return x


class SpeakerDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()

        # Network defition
        self.lstm = nn.LSTM(input_size=hparams.num_mels,
                            hidden_size=256,
                            num_layers=3,
                            batch_first=True)
        self.linear = nn.Linear(in_features=256,
                                out_features=1)

    def forward(self, utterances, hidden_init=None):
        """
        Computes the embeddings of a batch of utterance spectrograms.

        :param utterances: batch of mel-scale filterbanks of same duration as a tensor of shape
        (batch_size, n_frames, n_channels)
        :param hidden_init: initial hidden state of the LSTM as a tensor of shape (num_layers,
        batch_size, hidden_size). Will default to a tensor of zeros if None.
        :return: the embeddings as a tensor of shape (batch_size, embedding_size)
        """
        # Pass the input through the LSTM layers and retrieve all outputs, the final hidden state
        # and the final cell state.
        out, (hidden, cell) = self.lstm(utterances, hidden_init)

        # We take only the hidden state of the last layer
        embeds = self.linear(hidden[-1])

        return embeds


# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)


def pad_layer(inp, layer, is_2d=False):
    if type(layer.kernel_size) == tuple:
        kernel_size = layer.kernel_size[0]
    else:
        kernel_size = layer.kernel_size
    if not is_2d:
        if kernel_size % 2 == 0:
            pad = (kernel_size//2, kernel_size//2 - 1)
        else:
            pad = (kernel_size//2, kernel_size//2)
    else:
        if kernel_size % 2 == 0:
            pad = (kernel_size//2, kernel_size//2 - 1, kernel_size//2, kernel_size//2 - 1)
        else:
            pad = (kernel_size//2, kernel_size//2, kernel_size//2, kernel_size//2)
    # padding
    inp = F.pad(inp,
            pad=pad,
            mode='reflect')
    out = layer(inp)
    return out


class PatchDiscriminator(nn.Module):
    def __init__(self, n_class=33, ns=0.2, dp=0.1):
        super(PatchDiscriminator, self).__init__()
        self.ns = ns
        self.conv1 = nn.Conv2d(1, 64, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=2)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=5, stride=2)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=5, stride=2)
        self.conv5 = nn.Conv2d(512, 512, kernel_size=5, stride=2)
        self.conv6 = nn.Conv2d(512, 1, kernel_size=1)
        #self.conv_classify = nn.Conv2d(512, n_class, kernel_size=(17, 4))
        # self.conv_classify = nn.Conv2d(512, n_class, kernel_size=(17, 4))
        self.drop1 = nn.Dropout2d(p=dp)
        self.drop2 = nn.Dropout2d(p=dp)
        self.drop3 = nn.Dropout2d(p=dp)
        self.drop4 = nn.Dropout2d(p=dp)
        self.drop5 = nn.Dropout2d(p=dp)
        self.ins_norm1 = nn.InstanceNorm2d(self.conv1.out_channels)
        self.ins_norm2 = nn.InstanceNorm2d(self.conv2.out_channels)
        self.ins_norm3 = nn.InstanceNorm2d(self.conv3.out_channels)
        self.ins_norm4 = nn.InstanceNorm2d(self.conv4.out_channels)
        self.ins_norm5 = nn.InstanceNorm2d(self.conv5.out_channels)

    def conv_block(self, x, conv_layer, after_layers):
        out = pad_layer(x, conv_layer, is_2d=True)
        out = F.leaky_relu(out, negative_slope=self.ns)
        for layer in after_layers:
            out = layer(out)
        return out

    def forward(self, x, classify=False):
        x = torch.unsqueeze(x, dim=1)
        out = self.conv_block(x, self.conv1, [self.ins_norm1, self.drop1])
        out = self.conv_block(out, self.conv2, [self.ins_norm2, self.drop2])
        out = self.conv_block(out, self.conv3, [self.ins_norm3, self.drop3])
        out = self.conv_block(out, self.conv4, [self.ins_norm4, self.drop4])
        out = self.conv_block(out, self.conv5, [self.ins_norm5, self.drop5])
        # GAN output value
        val = pad_layer(out, self.conv6, is_2d=True)
        val = val.view(val.size(0), -1)
        mean_val = torch.mean(val, dim=1)
        # if classify:
        #     # classify
        #     logits = self.conv_classify(out)
        #     print(logits.size())
        #     logits = logits.view(logits.size(0), -1)
        #     return mean_val, logits
        # else:
        return mean_val


class Generator(nn.Module):
    """Generator network."""
    def __init__(self, dim_neck, dim_emb, dim_pre, freq, dim_spec=80, is_train=False, lr=0.001, decoder_type='simple',
                 vocoder_type='wavenet', encoder_type='default', separate_encoder=True, loss_content=True,
                 discriminator=False, dis_type='patch',
                 multigpu=False, cycle=False, lambda_cycle=1,
                 num_speakers=-1, idt_type='L2', use_lsgan=True, lambda_gan=0.0001,
                 train_wavenet=False, lambda_wavenet=0.001, args=None,
                 test_path_source=None, test_path_target=None,
                 attention=False, residual=False):
        super(Generator, self).__init__()

        if encoder_type == 'default':
            self.encoder = Encoder(dim_neck, dim_emb, freq)
        elif encoder_type == 'nospeaker' or encoder_type == 'single':
            self.encoder = MyEncoder(dim_neck, freq, num_mel=dim_spec)
        elif encoder_type == 'multiencoder':
            self.encoder = MultiEncoder(num_speakers, dim_neck, freq, separate_encoder)

        if encoder_type == 'multiencoder' or encoder_type == 'single':
            self.decoder = Decoder(dim_neck, 0, dim_pre, num_mel=dim_spec)
        elif decoder_type == 'simple':
            self.decoder = Decoder(dim_neck, dim_emb, dim_pre)
        elif decoder_type == 'tacotron':
            self.decoder = TacotronDecoder(hparams)
        elif decoder_type == 'multidecoder':
            self.decoder = MultiDecoder(num_speakers, dim_neck, dim_pre, multigpu)
        elif decoder_type == 'video':
            # self.decoder = VideoGenerator()
            self.decoder = STAGE2_G(residual=residual)
        self.postnet = Postnet(num_mel=dim_spec)
        if discriminator:
            if dis_type == 'patch':
                self.dis = PatchDiscriminator(n_class=num_speakers)
            else:
                self.dis = SpeakerDiscriminator()
            # self.dis_criterion = nn.CrossEntropyLoss(reduction='mean')
            self.dis_criterion = GANLoss(use_lsgan=use_lsgan, tensor=torch.cuda.FloatTensor)
        else:
            self.dis = None

        self.encoder_type = encoder_type
        self.decoder_type = decoder_type
        self.vocoder_type = vocoder_type
        self.loss_content = loss_content
        self.cycle = cycle
        self.lambda_cycle = lambda_cycle
        self.lambda_gan = lambda_gan
        self.lambda_wavenet = lambda_wavenet
        self.attention = attention

        self.multigpu = multigpu

        self.train_vocoder = train_wavenet
        if self.train_vocoder:
            self.vocoder = WaveRNN(
                rnn_dims=hparams.voc_rnn_dims,
                fc_dims=hparams.voc_fc_dims,
                bits=hparams.bits,
                pad=hparams.voc_pad,
                upsample_factors=hparams.voc_upsample_factors,
                feat_dims=hparams.num_mels,
                compute_dims=hparams.voc_compute_dims,
                res_out_dims=hparams.voc_res_out_dims,
                res_blocks=hparams.voc_res_blocks,
                hop_length=hparams.hop_size,
                sample_rate=hparams.sample_rate,
                mode=hparams.voc_mode
            )
            self.opt_vocoder = torch.optim.Adam(self.vocoder.parameters(), lr=hparams.voc_lr)
            self.vocoder_loss_func = F.cross_entropy # Only for RAW

        if multigpu:
            self.encoder = nn.DataParallel(self.encoder)
            self.decoder = nn.DataParallel(self.decoder)
            self.postnet = nn.DataParallel(self.postnet)
            self.vocoder = nn.DataParallel(self.vocoder)
            if self.dis is not None:
                self.dis = nn.DataParallel(self.dis)

    def forward(self, x, c_org, c_trg):
                
        codes = self.encoder(x, c_org)
        if c_trg is None:
            return torch.cat(codes, dim=-1)
        
        tmp = []
        for code in codes:
            tmp.append(code.unsqueeze(1).expand(-1,int(x.size(1)/len(codes)),-1))
        code_exp = torch.cat(tmp, dim=1)
        
        encoder_outputs = torch.cat((code_exp, c_trg.unsqueeze(1).expand(-1,x.size(1),-1)), dim=-1)
        # (batch, T, 256+dim_neck)
        mel_outputs = self.decoder(encoder_outputs)
                
        mel_outputs_postnet = self.postnet(mel_outputs.transpose(2,1))
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet.transpose(2,1)
        
        mel_outputs = mel_outputs.unsqueeze(1)
        mel_outputs_postnet = mel_outputs_postnet.unsqueeze(1)
        
        return mel_outputs, mel_outputs_postnet, torch.cat(codes, dim=-1)


    def conversion(self, speaker_org, speaker_trg, spec, device, speed=1):
        speaker_org, speaker_trg, spec = speaker_org.to(device), speaker_trg.to(device), spec.to(device)
        if self.encoder_type == 'multiencoder':
            codes = self.encoder(spec, speaker_trg)
        else:
            if not self.multigpu:
                codes = self.encoder(spec, speaker_org)
            else:
                codes = self.encoder.module(spec, speaker_org)
        tmp = []
        for code in codes:
            tmp.append(code.unsqueeze(1).expand(-1, int(speed * spec.size(1) / len(codes)), -1))
        code_exp = torch.cat(tmp, dim=1)
        if self.attention:
            code_exp = self.phonemeToken(code_exp)
        encoder_outputs = torch.cat((code_exp, speaker_trg.unsqueeze(1).expand(-1, code_exp.size(1), -1)), dim=-1)
        if self.encoder_type == 'multiencoder' or self.encoder_type == 'single':
            mel_outputs = self.decoder(code_exp) if not self.multigpu else self.decoder.module(code_exp)
        elif self.decoder_type == 'simple':
            mel_outputs = self.decoder(encoder_outputs)
        elif self.decoder_type == 'tacotron':
            try:
                mel_outputs, _, alignments = self.decoder.inference(memory=encoder_outputs)
            except:
                mel_outputs, _, alignments = self.decoder.module.inference(memory=encoder_outputs)
            mel_outputs.transpose_(1,2)
        elif self.decoder_type == 'multidecoder':
            mel_outputs = self.decoder(code_exp, speaker_trg)

        mel_outputs_postnet = self.postnet(mel_outputs.transpose(2, 1))
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet.transpose(2, 1)
        return mel_outputs_postnet




class VideoAudioGenerator(nn.Module):
    def __init__(self, dim_neck, dim_emb, dim_pre, freq, dim_spec=80, is_train=False, lr=0.001, vocoder_type='wavenet',
                 multigpu=False, num_speakers=-1, idt_type='L2',
                 train_wavenet=False, lambda_wavenet=0.001, args=None,
                 test_path_source=None, test_path_target=None,
                 residual=False, attention_map=None, use_256=False):
        super(VideoAudioGenerator, self).__init__()

        self.encoder = MyEncoder(dim_neck, freq, num_mel=dim_spec)

        self.decoder = Decoder(dim_neck, 0, dim_pre, num_mel=dim_spec)
        self.postnet = Postnet(num_mel=dim_spec)
        if use_256:
            self.video_decoder = VideoGenerator(use_256=True)
        else:
            self.video_decoder = STAGE2_G(residual=residual)
        self.use_256 = use_256
        self.vocoder_type = vocoder_type
        self.lambda_wavenet = lambda_wavenet

        self.multigpu = multigpu
        # self.prepare_test(dim_spec, test_path_source, test_path_target)

        self.train_vocoder = train_wavenet
        if self.train_vocoder:
            if vocoder_type == 'wavenet' or vocoder_type == 'griffin':
                self.vocoder = WaveRNN(
                    rnn_dims=hparams.voc_rnn_dims,
                    fc_dims=hparams.voc_fc_dims,
                    bits=hparams.bits,
                    pad=hparams.voc_pad,
                    upsample_factors=hparams.voc_upsample_factors,
                    feat_dims=hparams.num_mels,
                    compute_dims=hparams.voc_compute_dims,
                    res_out_dims=hparams.voc_res_out_dims,
                    res_blocks=hparams.voc_res_blocks,
                    hop_length=hparams.hop_size,
                    sample_rate=hparams.sample_rate,
                    mode=hparams.voc_mode
                )
                self.opt_vocoder = torch.optim.Adam(self.vocoder.parameters(), lr=hparams.voc_lr)
                self.vocoder_loss_func = F.cross_entropy # Only for RAW

        if attention_map is not None:
            self.attention_map_large = np.load(attention_map)
            self.attention_map = cv2.resize(self.attention_map_large, dsize=(128, 128), interpolation=cv2.INTER_CUBIC)
            # self.attention_map_large = self.attention_map_large.astype(np.float64)
            # self.attention_map = self.attention_map.astype(np.float64)
            self.attention_map_large = torch.from_numpy(self.attention_map_large / self.attention_map_large.max()).float()
            self.attention_map = torch.from_numpy(self.attention_map / self.attention_map.max()).float()
            self.criterionVideo = torch.nn.L1Loss(reduction='none')
        else:
            self.attention_map = None

        if multigpu:
            self.encoder = nn.DataParallel(self.encoder)
            self.decoder = nn.DataParallel(self.decoder)
            self.video_decoder = nn.DataParallel(self.video_decoder)
            self.postnet = nn.DataParallel(self.postnet)
            self.vocoder = nn.DataParallel(self.vocoder)

    def generate(self, mel, speaker, device='cuda:0'):
        mel, speaker = mel.to(device), speaker.to(device)
        if not self.multigpu:
            codes, code_unsample = self.encoder(mel, speaker, return_unsample=True)
        else:
            codes, code_unsample = self.encoder.module(mel, speaker, return_unsample=True)
                
        tmp = []
        for code in codes:
            tmp.append(code.unsqueeze(1).expand(-1, int(mel.size(1) / len(codes)), -1))
        code_exp = torch.cat(tmp, dim=1)

        if not self.multigpu:
            if not self.use_256:
                v_stage1, v_stage2 = self.video_decoder(code_unsample, train=True)
            else:
                v_stage2 = self.video_decoder(code_unsample)
                v_stage1 = v_stage2
            mel_outputs = self.decoder(code_exp)
            mel_outputs_postnet = self.postnet(mel_outputs.transpose(2, 1))
        else:
            if not self.use_256:
                v_stage1, v_stage2 = self.video_decoder.module(code_unsample, train=True)
            else:
                v_stage2 = self.video_decoder.module(code_unsample)
                v_stage1 = v_stage2
            mel_outputs = self.decoder.module(code_exp)
            mel_outputs_postnet = self.postnet.module(mel_outputs.transpose(2, 1))
        
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet.transpose(2, 1)
        
        return mel_outputs_postnet, v_stage1, v_stage2
