import torch
import torch.nn as nn
import torch.optim as optim
from einops import rearrange
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class EmbeddingNet(nn.Module):
    def __init__(self, input_size, output_size, dropout, use_bn, momentum,hidden_size=None):
        super(EmbeddingNet, self).__init__()
        modules = []
        if hidden_size:
            modules.append(nn.Linear(in_features=input_size, out_features=hidden_size))
            if use_bn:
                modules.append(nn.BatchNorm1d(num_features=hidden_size))
            modules.append(nn.ReLU())
            modules.append(nn.Dropout(dropout))
            modules.append(nn.Linear(in_features=hidden_size, out_features=output_size))
            modules.append(nn.BatchNorm1d(num_features=output_size, momentum=momentum))
            modules.append(nn.ReLU())
            modules.append(nn.Dropout(dropout))
        else:
            modules.append(nn.Linear(in_features=input_size, out_features=output_size))
            modules.append(nn.BatchNorm1d(num_features=output_size))
            modules.append(nn.ReLU())
            modules.append(nn.Dropout(dropout))
        self.fc = nn.Sequential(*modules)

    def forward(self, x):
        output = self.fc(x)
        return output

    def get_embedding(self, x):
        return self.forward(x)


class AVCA(nn.Module):
    def __init__(self, audio_encoder):
        super(AVCA, self).__init__()
        self.r_enc = 0.2
        self.r_proj = 0.2
        self.momentum = 0.1
        self.hidden_size_encoder = 512
        self.hidden_size_decoder = 512
        self.depth_transformer = 1
        self.dim_out = 64
        self.r_dec = 0.3
        self.audio_encoder = audio_encoder
        self.audio_encoder.train()
        #self.audio_projector = nn.Sequential(nn.LayerNorm(256), nn.Linear(256, 64, bias=False))
        self.image_projector =  nn.Sequential(nn.LayerNorm(768), nn.Linear(768, 256, bias=False))
        self.text_projector =  nn.Sequential(nn.LayerNorm(768), nn.Linear(768, 256, bias=False))


        self.audio_projector = EmbeddingNet(
            
            input_size=256,
            hidden_size=self.hidden_size_encoder,
            output_size=64,
            dropout=self.r_enc,
            momentum=self.momentum,
            use_bn=True
        )

        """
        self.dim_out = params_model['dim_out']
        self.hidden_size_encoder=params_model['encoder_hidden_size']
        self.hidden_size_decoder=params_model['decoder_hidden_size']
        self.r_enc=params_model['dropout_encoder']
        self.r_proj=params_model['dropout_decoder']
        self.depth_transformer=params_model['depth_transformer']
        self.additional_triplets_loss=params_model['additional_triplets_loss']
        self.reg_loss=params_model['reg_loss']
        self.r_dec=params_model['additional_dropout']
        self.momentum=params_model['momentum']

        self.first_additional_triplet=params_model['first_additional_triplet']
        self.second_additional_triplet=params_model['second_additional_triplet']
        """
        #print('Initializing trainable models...', end='')


        self.image_encoder = EmbeddingNet(
            
            input_size=768, #(512->768)
            hidden_size=self.hidden_size_encoder,
            output_size=256,
            dropout=self.r_enc,
            momentum=self.momentum,
            use_bn=True
        )
        
        self.cross_attention=Transformer(256, self.depth_transformer, 3, 100, 64, dropout=self.r_enc)

        self.W_proj= EmbeddingNet(
            input_size=256,
            output_size=self.dim_out,
            dropout=self.r_dec,
            momentum=self.momentum,
            use_bn=True
        )

        #self.D = EmbeddingNet(
        #    input_size=64,
        #    output_size=256,
        #    dropout=self.r_dec,
        #    momentum=self.momentum,
        #    use_bn=True
        #)



        self.A_proj = EmbeddingNet(input_size=256, hidden_size=self.hidden_size_decoder, output_size=self.dim_out, dropout=self.r_proj, momentum=self.momentum,use_bn=True)

        self.V_proj = EmbeddingNet(input_size=256, hidden_size=self.hidden_size_decoder, output_size=self.dim_out, dropout=self.r_proj, momentum=self.momentum,use_bn=True)


        self.pos_emb1D = torch.nn.Parameter(torch.randn(2, 256))

        # Optimizers
        #print('Defining optimizers...', end='')
        #self.lr = 0.001
        #self.optimizer_gen = optim.Adam(list(self.A_proj.parameters()) + list(self.V_proj.parameters()) +
        #                                list(self.A_rec.parameters()) + list(self.V_rec.parameters()) +
        #                                list(self.V_enc.parameters()) + list(self.A_enc.parameters()) +
        #                                list(self.cross_attention.parameters()) + list(self.D.parameters()) +
        #                                list(self.W_proj.parameters()),
        #                                lr=self.lr, weight_decay=1e-5)
        self.a_latent = nn.Identity()
        #self.optimizer_gen = optim.Adam(list(self.audio_encoder.parameters()) + list(self.audio_projector.parameters()) +
        #                                list(self.image_projector.parameters()) + list(self.text_projector.parameters()) +
        #                                list(self.cross_attention.parameters()) + list(self.a_latent.parameters()) + 
        #                                list(self.A_proj.parameters()) + list(self.V_proj.parameters()),
        #                                lr=self.lr, weight_decay=1e-5)
        #self.scheduler_gen =  optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_gen, 'max', patience=3, verbose=True)

        #print('Done')

        # Loss function
        #print('Defining losses...', end='')
        self.criterion_reg = nn.MSELoss()
        self.triplet_loss = nn.TripletMarginLoss(margin=1.0)
        #print('Done')

    def optimize_scheduler(self, value):
        self.scheduler_gen.step(value)

    #def forward(self, audio, image, negative_audio, negative_image, word_embedding, negative_word_embedding):
    def forward(self, audio, negative_audio, image, negative_image, text, negative_text):
        self.phi_t = self.text_projector(text)
        self.phi_i = self.image_projector(image)
        self.phi_t_neg = self.text_projector(negative_text)
        self.phi_i_neg = self.image_projector(negative_image)
        audio_emb = self.audio_encoder(audio)
        self.h_audio = self.a_latent(audio_emb[:,0,:])
        ##BILL TO-DO : AUDIO EMBEDDING
        self.theta_a = self.audio_projector(self.h_audio)
        neg_audio_emb = self.audio_encoder(negative_audio)
        self.neg_h_audio = self.a_latent(neg_audio_emb[:,0,:])
        self.theta_a_neg = self.audio_projector(self.neg_h_audio)

        self.positive_input=torch.stack((self.phi_t + self.pos_emb1D[0, :], self.phi_i + self.pos_emb1D[1, :]), dim=1)
        self.negative_input=torch.stack((self.phi_t_neg + self.pos_emb1D[0, :], self.phi_i_neg + self.pos_emb1D[1, :]), dim=1)

        self.phi_attn= self.cross_attention(self.positive_input)
        self.phi_attn_neg = self.cross_attention(self.negative_input)

        self.text_fe_attn = self.phi_t + self.phi_attn[:, 0, :]
        self.image_fe_attn= self.phi_i + self.phi_attn[:, 1, :]

        self.text_fe_neg_attn = self.phi_t_neg + self.phi_attn_neg[:, 0, :]
        self.image_fe_neg_attn = self.phi_i_neg + self.phi_attn_neg[:, 1, :]
        #TEXT-EMBEDDING
        self.theta_t = self.V_proj(self.text_fe_attn)
        self.theta_t_neg=self.V_proj(self.text_fe_neg_attn)

        #IMAGE-EMBEDDING
        self.theta_i = self.A_proj(self.image_fe_attn)
        self.theta_i_neg=self.A_proj(self.image_fe_neg_attn)
        first_pair = (self.triplet_loss(self.theta_t, self.theta_a, self.theta_t_neg) + \
                                                        self.triplet_loss(self.theta_i, self.theta_a, self.theta_i_neg))
        second_pair=(self.triplet_loss(self.theta_a, self.theta_t, self.theta_a_neg) + \
                                                        self.triplet_loss(self.theta_a, self.theta_i, self.theta_a_neg))

        l_t=first_pair+second_pair
        l_ai = self.triplet_loss(self.theta_a, self.theta_i, self.theta_i_neg)
        l_at = self.triplet_loss(self.theta_a, self.theta_t, self.theta_t_neg)
        l_ta = self.triplet_loss(self.theta_t, self.theta_a, self.theta_a_neg)
        l_ia = self.triplet_loss(self.theta_i, self.theta_a, self.theta_a_neg)
        loss_details = {'first_pair': first_pair,  'second_pair': second_pair,
                'l_ai':l_ai, 'l_at':l_at, 'l_ta':l_ta,'l_ia':l_ia}
        loss_numeric = l_t + l_ai + l_at + l_ta + l_ia
        return loss_numeric, loss_details

    def get_theta_t_i(self, text, image):
        
        self.phi_t = self.text_projector(text)
        self.phi_i = self.image_projector(image)
        
        self.positive_input=torch.stack((self.phi_t + self.pos_emb1D[0, :], self.phi_i + self.pos_emb1D[1, :]), dim=1)
        self.phi_attn= self.cross_attention(self.positive_input)
        
        self.text_fe_attn = self.phi_t + self.phi_attn[:, 0, :]
        self.image_fe_attn= self.phi_i + self.phi_attn[:, 1, :]

        self.theta_t = self.V_proj(self.text_fe_attn)
        self.theta_i = self.A_proj(self.image_fe_attn)
        
       
        return self.theta_t, self.theta_i
    def get_theta_a(self, audio):

        audio_emb = self.audio_encoder(audio)
        self.h_audio = self.a_latent(audio_emb[:,0,:])
        self.theta_a = self.audio_projector(self.h_audio)
        return self.theta_a
    #def backward(self, optimize):
        """
        if self.additional_triplets_loss==True:
            first_pair = self.first_additional_triplet*(self.triplet_loss(self.theta_t, self.theta_a, self.theta_t_neg) + \
                                                        self.triplet_loss(self.theta_i, self.theta_a, self.theta_i_neg))
            second_pair=self.second_additional_triplet*(self.triplet_loss(self.theta_a, self.theta_t, self.theta_a_neg) + \
                                                        self.triplet_loss(self.theta_a, self.theta_i, self.theta_a_neg))

            l_t=first_pair+second_pair

        if self.reg_loss==True:
            l_r = (self.criterion_reg(self.phi_i_rec, self.phi_i) + \
                            self.criterion_reg(self.phi_t_rec, self.phi_t) + \
                            self.criterion_reg(self.theta_i, self.theta_a) + \
                            self.criterion_reg(self.theta_t, self.theta_a))


        l_rec= self.criterion_reg(self.h_audio, self.rho_i) + \
                  self.criterion_reg(self.h_audio, self.rho_t) + \
                  self.criterion_reg(self.h_audio, self.rho_a)

        l_ctv=self.triplet_loss(self.rho_a, self.rho_i, self.rho_i_neg)
        l_cta=self.triplet_loss(self.rho_a, self.rho_t, self.rho_t_neg)
        l_ct=l_cta+l_ctv
        l_cmd=l_rec+l_ct

        l_tv = self.triplet_loss(self.theta_a, self.theta_i, self.theta_i_neg)
        l_ta = self.triplet_loss(self.theta_a, self.theta_t, self.theta_t_neg)
        l_at = self.triplet_loss(self.theta_t, self.theta_a, self.theta_a_neg)
        l_vt = self.triplet_loss(self.theta_i, self.theta_a, self.theta_a_neg)

        l_w=l_ta+l_at+l_tv+l_vt

        loss_gen=l_cmd + l_w
        if self.additional_triplets_loss==True:
           loss_gen+=l_t
        if self.reg_loss==True:
            loss_gen+=l_r

        if optimize == True:
            self.optimizer_gen.zero_grad()
            loss_gen.backward()
            self.optimizer_gen.step()

        loss = {'aut_enc': 0,  'gen_cyc': 0,
                'gen_reg': 0, 'gen': loss_gen}

        loss_numeric = loss['gen_cyc'] + loss['gen']
        """


    def optimize_params(self, audio, negative_audio, image, negative_image, text, negative_text,optimize=False):

        self.forward(audio, negative_audio, image, negative_image, text, negative_text)

        loss_numeric, loss = self.backward(optimize)

        return loss_numeric, loss



