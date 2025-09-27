import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
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
            modules.append(nn.GELU())
            modules.append(nn.Dropout(dropout))
            modules.append(nn.Linear(in_features=hidden_size, out_features=output_size))
            modules.append(nn.BatchNorm1d(num_features=output_size, momentum=momentum))
            modules.append(nn.GELU())
            modules.append(nn.Dropout(dropout))
        else:
            modules.append(nn.Linear(in_features=input_size, out_features=output_size))
            modules.append(nn.BatchNorm1d(num_features=output_size))
            modules.append(nn.GELU())
            modules.append(nn.Dropout(dropout))
        self.fc = nn.Sequential(*modules)

    def forward(self, x):
        output = self.fc(x)
        return output

    def get_embedding(self, x):
        return self.forward(x)

class CLUBSample(nn.Module):  # Sampled version of the CLUB estimator
    def __init__(self, x_dim, y_dim, hidden_size):
        super(CLUBSample, self).__init__()
        self.p_mu = nn.Sequential(nn.Linear(x_dim, hidden_size//2),
                                       nn.ReLU(),
                                       nn.Linear(hidden_size//2, y_dim))

        self.p_logvar = nn.Sequential(nn.Linear(x_dim, hidden_size//2),
                                       nn.ReLU(),
                                       nn.Linear(hidden_size//2, y_dim),
                                       nn.LeakyReLU())

    def get_mu_logvar(self, x_samples):
        mu = self.p_mu(x_samples)
        logvar = self.p_logvar(x_samples)
        return mu, logvar
     
        
    def loglikeli(self, x_samples, y_samples):
        mu, logvar = self.get_mu_logvar(x_samples)
        return (-(mu - y_samples)**2 /logvar.exp()-logvar).sum(dim=1).mean(dim=0)
    

    def forward(self, x_samples, y_samples):
        mu, logvar = self.get_mu_logvar(x_samples)
        
        sample_size = x_samples.shape[0]
        #random_index = torch.randint(sample_size, (sample_size,)).long()
        random_index = torch.randperm(sample_size).long()
        
        positive = - (mu - y_samples)**2 / logvar.exp()
        negative = - (mu - y_samples[random_index])**2 / logvar.exp()
        upper_bound = (positive.sum(dim = -1) - negative.sum(dim = -1)).mean()
        return upper_bound/2.

    #def learning_loss(self, x_samples, y_samples):
    #    return - self.loglikeli(x_samples, y_samples)
    def learning_loss(self, x_samples, y_samples):
        mu, logvar = self.get_mu_logvar(x_samples)
        ll = (-(mu - y_samples)**2 / logvar.exp() - logvar).sum(dim=1).mean(dim=0)
        optimal_ll = (-logvar).sum(dim=1).mean(dim=0)  # Log likelihood when y = mu
        return -(ll - optimal_ll)  # Negative of the difference
    #Pros: Retains the probabilistic interpretation, ensures loss is 0 at perfect prediction, still penalizes poor variance estimates.
    #Cons: Still allows negative values if the model’s log likelihood exceeds the optimal case (e.g., tighter variance), though less likely.
class ALL(nn.Module):
    def __init__(self, audio_encoder, text_encoder, image_encoder):
        super(ALL, self).__init__()
        self.r_enc = 0.2
        self.r_proj = 0.2
        self.momentum = 0.1
        self.hidden_size_encoder = 512
        self.hidden_size_decoder = 512
        self.depth_transformer = 1
        self.dim_out = 64
        self.r_dec = 0.3
        self.audio_encoder = audio_encoder
        self.text_encoder = text_encoder
        self.image_encoder = image_encoder
        self.audio_encoder.train()
        self.text_encoder.train()
        self.image_encoder.train()
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


        self.cross_attention=Transformer(256, self.depth_transformer, 3, 100, 64, dropout=self.r_enc)

        self.cross_projector = EmbeddingNet(
            
            input_size=512,
            hidden_size=self.hidden_size_encoder,
            output_size=64,
            dropout=self.r_enc,
            momentum=self.momentum,
            use_bn=True
        )


        self.t_latent = nn.Identity()
        self.pos_emb1D = torch.nn.Parameter(torch.randn(2, 256))

        self.a_latent = nn.Identity()

        self.criterion = nn.CrossEntropyLoss()
        temperature = 0.2
        self.init_temperature = torch.tensor([np.log(1/temperature)])
        self.logit_scale = nn.Parameter(self.init_temperature, requires_grad = True)


    def optimize_scheduler(self, value):
        self.scheduler_gen.step(value)

    def get_loss(self, h1, h2):
        temperature = torch.clamp(self.logit_scale.exp(), max = 100)
        h1 = nn.functional.normalize(h1, dim=1)
        h2 = nn.functional.normalize(h2, dim=1)
        logits = torch.einsum('nc,mc->nm',[h1,h2]) * temperature.to("cuda")
        N = logits.shape[0]
        labels = torch.arange(N, dtype=torch.long, device = "cuda")
        return F.cross_entropy(logits, labels)
    def forward(self, audio, image, text, text_mask):
        print("image.shape", image.shape)
        exit()
        outputs = self.image_encoder(image)
        embeddings = outputs.last_hidden_state
        cls_embedding = embeddings[:, 0]

        text_emb = self.text_encoder(text, text_mask)
        h_text = self.t_latent(text_emb['last_hidden_state'][:,0,:])
        self.phi_t = self.text_projector(h_text)
        self.phi_i = self.image_projector(cls_embedding)
        
        audio_emb = self.audio_encoder(audio)
        self.h_audio = self.a_latent(audio_emb[:,0,:])
        self.theta_a = self.audio_projector(self.h_audio)
        
        self.positive_input=torch.stack((self.phi_t + self.pos_emb1D[0, :], self.phi_i + self.pos_emb1D[1, :]), dim=1)
        self.phi_attn= self.cross_attention(self.positive_input)
        self.text_fe_attn = self.phi_t + self.phi_attn[:, 0, :]
        self.image_fe_attn= self.phi_i + self.phi_attn[:, 1, :]
        self.concat_vector = torch.cat((self.text_fe_attn, self.image_fe_attn), 1)
        self.theta_c = self.cross_projector(self.concat_vector)
        ca_loss =  self.get_loss(self.theta_c, self.theta_a)
        ac_loss = self.get_loss(self.theta_a, self.theta_c)
        loss_details = {'ca_loss':ca_loss, 'ac_loss':ac_loss}
        return 0.5 * (ca_loss + ac_loss), loss_details

    def get_theta_c(self, text, text_mask, image):
        outputs = self.image_encoder(image)
        embeddings = outputs.last_hidden_state
        cls_embedding = embeddings[:, 0]

        text_emb = self.text_encoder(text, text_mask)
        h_text = self.t_latent(text_emb['last_hidden_state'][:,0,:])

        self.phi_t = self.text_projector(h_text)
        self.phi_i = self.image_projector(cls_embedding)
 
        self.positive_input=torch.stack((self.phi_t + self.pos_emb1D[0, :], self.phi_i + self.pos_emb1D[1, :]), dim=1)

        self.phi_attn= self.cross_attention(self.positive_input)
        
        self.text_fe_attn = self.phi_t + self.phi_attn[:, 0, :]
        self.image_fe_attn= self.phi_i + self.phi_attn[:, 1, :]

        self.concat_vector = torch.cat((self.text_fe_attn, self.image_fe_attn), 1)
        self.theta_c = self.cross_projector(self.concat_vector)
       
        return self.theta_c
    def get_theta_a(self, audio):

        audio_emb = self.audio_encoder(audio)
        self.h_audio = self.a_latent(audio_emb[:,0,:])
        self.theta_a = self.audio_projector(self.h_audio)
        return self.theta_a

class ALL_CLUB(nn.Module):
    def __init__(self, audio_encoder, text_encoder, image_encoder):
        super(ALL_CLUB, self).__init__()
        self.r_enc = 0.2
        self.r_proj = 0.2
        self.momentum = 0.1
        self.hidden_size_encoder = 256
        self.hidden_size_decoder = 256
        self.depth_transformer = 1
        self.dim_out = 64
        self.r_dec = 0.3
        self.audio_encoder = audio_encoder
        self.text_encoder = text_encoder
        self.image_encoder = image_encoder
        self.audio_encoder.train()
        self.text_encoder.train()
        self.image_encoder.train()
        self.image_projector =  nn.Sequential(nn.LayerNorm(768), nn.Linear(768, 256, bias=False))
        self.text_projector =  nn.Sequential(nn.LayerNorm(768), nn.Linear(768, 256, bias=False))

        self.image_spec = nn.Sequential(nn.Linear(256,128), nn.LeakyReLU(), nn.Linear(128,64))
        self.image_general = nn.Sequential(nn.Linear(256,128), nn.LeakyReLU(), nn.Linear(128,64))
        self.text_spec = nn.Sequential(nn.Linear(256,128), nn.LeakyReLU(), nn.Linear(128,64))
        self.text_general = nn.Sequential(nn.Linear(256,128), nn.LeakyReLU(), nn.Linear(128,64))
        self.audio_spec = nn.Sequential(nn.Linear(64,64), nn.LeakyReLU(), nn.Linear(64,64))
        self.audio_general = nn.Sequential(nn.Linear(64,64), nn.LeakyReLU(), nn.Linear(64,64))

        self.img_estimator = CLUBSample(64,64, 64)
        self.text_estimator = CLUBSample(64,64,64)
        self.audio_estimator = CLUBSample(64,64,64)


        self.audio_projector = EmbeddingNet(
            
            input_size=256,
            hidden_size=self.hidden_size_encoder,
            output_size=64,
            dropout=self.r_enc,
            momentum=self.momentum,
            use_bn=True
        )


        self.t_latent = nn.Identity()
        self.pos_emb1D = torch.nn.Parameter(torch.randn(2, 256))

        self.a_latent = nn.Identity()

        self.criterion = nn.CrossEntropyLoss()
        temperature = 0.2
        self.init_temperature = torch.tensor([np.log(1/temperature)])
        self.logit_scale = nn.Parameter(self.init_temperature, requires_grad = True)


    def optimize_scheduler(self, value):
        self.scheduler_gen.step(value)

    def InfoNCE(self, view1, view2, temperature = 0.4):
        view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
        pos_score = (view1 * view2).sum(dim=-1)
        pos_score = torch.exp(pos_score / temperature)
        ttl_score = torch.matmul(view1, view2.transpose(0, 1))
        ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)
        cl_loss = -torch.log(pos_score / (ttl_score +1e-8))

        result = torch.mean(cl_loss)
        return result

    def forward(self,audio, image, text, text_mask):
        outputs = self.image_encoder(image)
        embeddings = outputs.last_hidden_state
        cls_embedding = embeddings[:, 0]

        text_emb = self.text_encoder(text, text_mask)
        h_text = self.t_latent(text_emb['last_hidden_state'][:,0,:])
        phi_t = self.text_projector(h_text)#text feature
        phi_i = self.image_projector(cls_embedding)#image feature
        audio_emb = self.audio_encoder(audio)
        h_audio = self.a_latent(audio_emb[:,0,:])
        theta_a = self.audio_projector(h_audio)#audio feature

        i_s, i_g = self.image_spec(phi_i), self.image_general(phi_i)
        t_s, t_g = self.text_spec(phi_t), self.text_general(phi_t)
        a_s, a_g = self.audio_spec(theta_a), self.audio_general(theta_a)

        loss_club = 0
        loss_club += self.img_estimator.learning_loss(i_s,i_g)
        loss_club += self.text_estimator.learning_loss(t_s,t_g)
        loss_club += self.audio_estimator.learning_loss(a_s,a_g)

        loss_InfoNCE = 0
        loss_InfoNCE += self.InfoNCE(i_g,t_g)
        loss_InfoNCE += self.InfoNCE(i_g,a_g)
        loss_InfoNCE += self.InfoNCE(t_g,a_g)

        loss_details = {'ca_loss': loss_club, 'ac_loss': loss_InfoNCE}
        return 0.5*(loss_club+loss_InfoNCE), loss_details


    def get_theta_c(self, text, text_mask, image):
        outputs = self.image_encoder(image)
        embeddings = outputs.last_hidden_state
        cls_embedding = embeddings[:, 0]

        text_emb = self.text_encoder(text, text_mask)
        h_text = self.t_latent(text_emb['last_hidden_state'][:,0,:])

        phi_t = self.text_projector(h_text)
        phi_i = self.image_projector(cls_embedding)
        
        i_g = self.image_general(phi_i)
        t_g = self.text_general(phi_t)
       
        return (i_g+t_g)/2
    
    def get_theta_a(self, audio):

        audio_emb = self.audio_encoder(audio)
        h_audio = self.a_latent(audio_emb[:,0,:])
        theta_a = self.audio_projector(h_audio)

        a_g = self.audio_general(theta_a)
        return a_g
    

class Im(nn.Module):
    def __init__(self, audio_encoder, image_encoder):
        super(Im, self).__init__()
        self.r_enc = 0.2
        self.r_proj = 0.2
        self.momentum = 0.1
        self.hidden_size_encoder = 512
        self.hidden_size_decoder = 512
        self.depth_transformer = 1
        self.dim_out = 64
        self.r_dec = 0.3
        self.audio_encoder = audio_encoder
        self.image_encoder = image_encoder
        self.audio_encoder.train()
        self.image_encoder.train()
        
        self.image_projector =  nn.Sequential(nn.LayerNorm(768), nn.Linear(768, 256, bias=False))


        self.audio_projector = EmbeddingNet(
            
            input_size=256,
            hidden_size=self.hidden_size_encoder,
            output_size=64,
            dropout=self.r_enc,
            momentum=self.momentum,
            use_bn=True
        )

  
        self.A_proj = EmbeddingNet(input_size=256, hidden_size=self.hidden_size_decoder, output_size=self.dim_out, dropout=self.r_proj, momentum=self.momentum,use_bn=True)
        self.cross_attention=Transformer(256, self.depth_transformer, 3, 100, 64, dropout=self.r_enc)

        self.t_latent = nn.Identity()

        self.a_latent = nn.Identity()

        self.criterion = nn.CrossEntropyLoss()
        temperature = 0.2
        self.init_temperature = torch.tensor([np.log(1/temperature)])
        self.logit_scale = nn.Parameter(self.init_temperature, requires_grad = True)
        self.pos_emb1D = torch.nn.Parameter(torch.randn(1, 256))
    def optimize_scheduler(self, value):
        self.scheduler_gen.step(value)

    def get_loss(self, h1, h2):
        temperature = torch.clamp(self.logit_scale.exp(), max = 100)
        h1 = nn.functional.normalize(h1, dim=1)
        h2 = nn.functional.normalize(h2, dim=1)
        logits = torch.einsum('nc,mc->nm',[h1,h2]) * temperature.to("cuda")
        N = logits.shape[0]
        labels = torch.arange(N, dtype=torch.long, device = "cuda")
        return F.cross_entropy(logits, labels)
    def forward(self, audio, image):
        outputs = self.image_encoder(image)
        embeddings = outputs.last_hidden_state
        cls_embedding = embeddings[:, 0]
        self.phi_i = self.image_projector(cls_embedding)
        self.input= self.phi_i + self.pos_emb1D
        self.input = self.input.unsqueeze(1)
        self.phi_attn= self.cross_attention(self.input)
        self.fe_attn = self.phi_i + self.phi_attn[:, 0, :]
        self.theta_i = self.A_proj(self.fe_attn)
        audio_emb = self.audio_encoder(audio)
        self.h_audio = self.a_latent(audio_emb[:,0,:])
        self.theta_a = self.audio_projector(self.h_audio)

        image_loss =  self.get_loss(self.theta_i, self.theta_a)
        ai_loss = self.get_loss(self.theta_a, self.theta_i)
        
        loss_details = {'image_loss':image_loss, 'reverse_image_loss':ai_loss}
        return 0.5 * (image_loss + ai_loss), loss_details


    def get_theta_i(self, image):
        
        outputs = self.image_encoder(image)
        embeddings = outputs.last_hidden_state
        cls_embedding = embeddings[:, 0]
        self.phi_i = self.image_projector(cls_embedding)
        self.theta_i = self.A_proj(self.phi_i)
        
       
        return self.theta_i
    def get_theta_a(self, audio):

        audio_emb = self.audio_encoder(audio)
        self.h_audio = self.a_latent(audio_emb[:,0,:])
        self.theta_a = self.audio_projector(self.h_audio)
        return self.theta_a

class Ta(nn.Module):
    def __init__(self, audio_encoder, text_encoder):
        super(Ta, self).__init__()
        self.r_enc = 0.2
        self.r_proj = 0.2
        self.momentum = 0.1
        self.hidden_size_encoder = 512
        self.hidden_size_decoder = 512
        self.depth_transformer = 1
        self.dim_out = 64
        self.r_dec = 0.3
        self.audio_encoder = audio_encoder
        self.text_encoder = text_encoder
        self.audio_encoder.train()
        self.text_encoder.train()
        
        self.text_projector =  nn.Sequential(nn.LayerNorm(768), nn.Linear(768, 256, bias=False))


        self.audio_projector = EmbeddingNet(
            
            input_size=256,
            hidden_size=self.hidden_size_encoder,
            output_size=64,
            dropout=self.r_enc,
            momentum=self.momentum,
            use_bn=True
        )

  
        self.V_proj = EmbeddingNet(input_size=256, hidden_size=self.hidden_size_decoder, output_size=self.dim_out, dropout=self.r_proj, momentum=self.momentum,use_bn=True)
        self.cross_attention=Transformer(256, self.depth_transformer, 3, 100, 64, dropout=self.r_enc)

        self.t_latent = nn.Identity()

        self.a_latent = nn.Identity()

        self.criterion = nn.CrossEntropyLoss()
        temperature = 0.2
        self.init_temperature = torch.tensor([np.log(1/temperature)])
        self.logit_scale = nn.Parameter(self.init_temperature, requires_grad = True)
        self.pos_emb1D = torch.nn.Parameter(torch.randn(1, 256))
    def optimize_scheduler(self, value):
        self.scheduler_gen.step(value)

    def get_loss(self, h1, h2):
        temperature = torch.clamp(self.logit_scale.exp(), max = 100)
        h1 = nn.functional.normalize(h1, dim=1)
        h2 = nn.functional.normalize(h2, dim=1)
        logits = torch.einsum('nc,mc->nm',[h1,h2]) * temperature.to("cuda")
        N = logits.shape[0]
        labels = torch.arange(N, dtype=torch.long, device = "cuda")
        return F.cross_entropy(logits, labels)
    def forward(self, audio, text, text_mask):
        

        
        text_emb = self.text_encoder(text, text_mask)
        h_text = self.t_latent(text_emb['last_hidden_state'][:,0,:])
        self.phi_t = self.text_projector(h_text)
        self.input= self.phi_t + self.pos_emb1D
        self.input = self.input.unsqueeze(1)
        self.phi_attn= self.cross_attention(self.input)
        self.text_fe_attn = self.phi_t + self.phi_attn[:, 0, :]
        self.theta_t = self.V_proj(self.text_fe_attn)

        audio_emb = self.audio_encoder(audio)
        self.h_audio = self.a_latent(audio_emb[:,0,:])
        self.theta_a = self.audio_projector(self.h_audio)

        text_loss = self.get_loss(self.theta_t, self.theta_a)
        at_loss = self.get_loss(self.theta_a, self.theta_t)
        loss_details = {'text_loss':text_loss, 'reverse_text_loss':at_loss}
        return 0.5 * (text_loss + at_loss), loss_details

    def get_theta_t(self, text, text_mask):
        
        text_emb = self.text_encoder(text, text_mask)
        h_text = self.t_latent(text_emb['last_hidden_state'][:,0,:])
        self.phi_t = self.text_projector(h_text)
        
        self.theta_t = self.V_proj(self.phi_t)
        
       
        return self.theta_t
    def get_theta_a(self, audio):

        audio_emb = self.audio_encoder(audio)
        self.h_audio = self.a_latent(audio_emb[:,0,:])
        self.theta_a = self.audio_projector(self.h_audio)
        return self.theta_a

class Plo(nn.Module):
    def __init__(self, audio_encoder, text_encoder):
        super(Plo, self).__init__()
        self.r_enc = 0.2
        self.r_proj = 0.2
        self.momentum = 0.1
        self.hidden_size_encoder = 512
        self.hidden_size_decoder = 512
        self.depth_transformer = 1
        self.dim_out = 64
        self.r_dec = 0.3
        self.audio_encoder = audio_encoder
        self.text_encoder = text_encoder
        self.audio_encoder.train()
        self.text_encoder.train()
        
        self.text_projector =  nn.Sequential(nn.LayerNorm(768), nn.Linear(768, 256, bias=False))


        self.audio_projector = EmbeddingNet(
            
            input_size=256,
            hidden_size=self.hidden_size_encoder,
            output_size=64,
            dropout=self.r_enc,
            momentum=self.momentum,
            use_bn=True
        )

  
        self.V_proj = EmbeddingNet(input_size=256, hidden_size=self.hidden_size_decoder, output_size=self.dim_out, dropout=self.r_proj, momentum=self.momentum,use_bn=True)
        self.cross_attention=Transformer(256, self.depth_transformer, 3, 100, 64, dropout=self.r_enc)

        self.t_latent = nn.Identity()

        self.a_latent = nn.Identity()

        self.criterion = nn.CrossEntropyLoss()
        temperature = 0.2
        self.init_temperature = torch.tensor([np.log(1/temperature)])
        self.logit_scale = nn.Parameter(self.init_temperature, requires_grad = True)
        self.pos_emb1D = torch.nn.Parameter(torch.randn(1, 256))
    def optimize_scheduler(self, value):
        self.scheduler_gen.step(value)

    def get_loss(self, h1, h2):
        temperature = torch.clamp(self.logit_scale.exp(), max = 100)
        h1 = nn.functional.normalize(h1, dim=1)
        h2 = nn.functional.normalize(h2, dim=1)
        logits = torch.einsum('nc,mc->nm',[h1,h2]) * temperature.to("cuda")
        N = logits.shape[0]
        labels = torch.arange(N, dtype=torch.long, device = "cuda")
        return F.cross_entropy(logits, labels)
    def forward(self, audio, text, text_mask):
        

        
        text_emb = self.text_encoder(text, text_mask)
        h_text = self.t_latent(text_emb['last_hidden_state'][:,0,:])
        self.phi_t = self.text_projector(h_text)
        self.input= self.phi_t + self.pos_emb1D
        self.input = self.input.unsqueeze(1)
        self.phi_attn= self.cross_attention(self.input)
        self.text_fe_attn = self.phi_t + self.phi_attn[:, 0, :]
        self.theta_t = self.V_proj(self.text_fe_attn)

        audio_emb = self.audio_encoder(audio)
        self.h_audio = self.a_latent(audio_emb[:,0,:])
        self.theta_a = self.audio_projector(self.h_audio)

        text_loss = self.get_loss(self.theta_t, self.theta_a)
        at_loss = self.get_loss(self.theta_a, self.theta_t)
        loss_details = {'text_loss':text_loss, 'reverse_text_loss':at_loss}
        return 0.5 * (text_loss + at_loss), loss_details

    def get_theta_t(self, text, text_mask):
        
        text_emb = self.text_encoder(text, text_mask)
        h_text = self.t_latent(text_emb['last_hidden_state'][:,0,:])
        self.phi_t = self.text_projector(h_text)
        
        self.theta_t = self.V_proj(self.phi_t)
        
       
        return self.theta_t
    def get_theta_a(self, audio):

        audio_emb = self.audio_encoder(audio)
        self.h_audio = self.a_latent(audio_emb[:,0,:])
        self.theta_a = self.audio_projector(self.h_audio)
        return self.theta_a


class ImPlo(nn.Module):
    def __init__(self, audio_encoder, text_encoder, image_encoder):
        super(ImPlo, self).__init__()
        self.r_enc = 0.2
        self.r_proj = 0.2
        self.momentum = 0.1
        self.hidden_size_encoder = 512
        self.hidden_size_decoder = 512
        self.depth_transformer = 1
        self.dim_out = 64
        self.r_dec = 0.3
        self.audio_encoder = audio_encoder
        self.text_encoder = text_encoder
        self.image_encoder = image_encoder
        self.audio_encoder.train()
        self.text_encoder.train()
        self.image_encoder.train()
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


        self.cross_attention=Transformer(256, self.depth_transformer, 3, 100, 64, dropout=self.r_enc)

        self.cross_projector = EmbeddingNet(
            
            input_size=512,
            hidden_size=self.hidden_size_encoder,
            output_size=64,
            dropout=self.r_enc,
            momentum=self.momentum,
            use_bn=True
        )


        self.t_latent = nn.Identity()
        self.pos_emb1D = torch.nn.Parameter(torch.randn(2, 256))

        self.a_latent = nn.Identity()

        self.criterion = nn.CrossEntropyLoss()
        temperature = 0.2
        self.init_temperature = torch.tensor([np.log(1/temperature)])
        self.logit_scale = nn.Parameter(self.init_temperature, requires_grad = True)


    def optimize_scheduler(self, value):
        self.scheduler_gen.step(value)

    def get_loss(self, h1, h2):
        temperature = torch.clamp(self.logit_scale.exp(), max = 100)
        h1 = nn.functional.normalize(h1, dim=1)
        h2 = nn.functional.normalize(h2, dim=1)
        logits = torch.einsum('nc,mc->nm',[h1,h2]) * temperature.to("cuda")
        N = logits.shape[0]
        labels = torch.arange(N, dtype=torch.long, device = "cuda")
        return F.cross_entropy(logits, labels)
    def forward(self, audio, image, text, text_mask):

        
        outputs = self.image_encoder(image)
        embeddings = outputs.last_hidden_state
        cls_embedding = embeddings[:, 0]
        text_emb = self.text_encoder(text, text_mask)
        h_text = self.t_latent(text_emb['last_hidden_state'][:,0,:])
        self.phi_t = self.text_projector(h_text)
        self.phi_i = self.image_projector(cls_embedding)
        
        audio_emb = self.audio_encoder(audio)
        self.h_audio = self.a_latent(audio_emb[:,0,:])
        self.theta_a = self.audio_projector(self.h_audio)
        
        self.positive_input=torch.stack((self.phi_t + self.pos_emb1D[0, :], self.phi_i + self.pos_emb1D[1, :]), dim=1)
        
        self.phi_attn= self.cross_attention(self.positive_input)
        
        self.text_fe_attn = self.phi_t + self.phi_attn[:, 0, :]
        self.image_fe_attn= self.phi_i + self.phi_attn[:, 1, :]
        self.concat_vector = torch.cat((self.text_fe_attn, self.image_fe_attn), 1)
        
        self.theta_c = self.cross_projector(self.concat_vector)
        
        ca_loss =  self.get_loss(self.theta_c, self.theta_a)
        ac_loss = self.get_loss(self.theta_a, self.theta_c)
        
        loss_details = {'ca_loss':ca_loss, 'ac_loss':ac_loss}
        return 0.5 * (ca_loss + ac_loss), loss_details

    def get_theta_c(self, text, text_mask, image):
        outputs = self.image_encoder(image)
        embeddings = outputs.last_hidden_state
        cls_embedding = embeddings[:, 0]
        text_emb = self.text_encoder(text, text_mask)
        h_text = self.t_latent(text_emb['last_hidden_state'][:,0,:])
        self.phi_t = self.text_projector(h_text)
        self.phi_i = self.image_projector(cls_embedding)
        
        self.positive_input=torch.stack((self.phi_t + self.pos_emb1D[0, :], self.phi_i + self.pos_emb1D[1, :]), dim=1)
        
        self.phi_attn= self.cross_attention(self.positive_input)
        
        self.text_fe_attn = self.phi_t + self.phi_attn[:, 0, :]
        self.image_fe_attn= self.phi_i + self.phi_attn[:, 1, :]
        self.concat_vector = torch.cat((self.text_fe_attn, self.image_fe_attn), 1)
        
        self.theta_c = self.cross_projector(self.concat_vector)
        
        return self.theta_c
    def get_theta_a(self, audio):

        audio_emb = self.audio_encoder(audio)
        self.h_audio = self.a_latent(audio_emb[:,0,:])
        self.theta_a = self.audio_projector(self.h_audio)
        return self.theta_a

class ImTa(nn.Module):
    def __init__(self, audio_encoder, text_encoder, image_encoder):
        super(ImTa, self).__init__()
        self.r_enc = 0.2
        self.r_proj = 0.2
        self.momentum = 0.1
        self.hidden_size_encoder = 512
        self.hidden_size_decoder = 512
        self.depth_transformer = 1
        self.dim_out = 64
        self.r_dec = 0.3
        self.audio_encoder = audio_encoder
        self.text_encoder = text_encoder
        self.image_encoder = image_encoder
        self.audio_encoder.train()
        self.text_encoder.train()
        self.image_encoder.train()
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


        self.cross_attention=Transformer(256, self.depth_transformer, 3, 100, 64, dropout=self.r_enc)

        self.cross_projector = EmbeddingNet(
            
            input_size=512,
            hidden_size=self.hidden_size_encoder,
            output_size=64,
            dropout=self.r_enc,
            momentum=self.momentum,
            use_bn=True
        )


        self.t_latent = nn.Identity()
        self.pos_emb1D = torch.nn.Parameter(torch.randn(2, 256))

        self.a_latent = nn.Identity()

        self.criterion = nn.CrossEntropyLoss()
        temperature = 0.2
        self.init_temperature = torch.tensor([np.log(1/temperature)])
        self.logit_scale = nn.Parameter(self.init_temperature, requires_grad = True)


    def optimize_scheduler(self, value):
        self.scheduler_gen.step(value)

    def get_loss(self, h1, h2):
        temperature = torch.clamp(self.logit_scale.exp(), max = 100)
        h1 = nn.functional.normalize(h1, dim=1)
        h2 = nn.functional.normalize(h2, dim=1)
        logits = torch.einsum('nc,mc->nm',[h1,h2]) * temperature.to("cuda")
        N = logits.shape[0]
        labels = torch.arange(N, dtype=torch.long, device = "cuda")
        return F.cross_entropy(logits, labels)
    def forward(self, audio, image, text, text_mask):
        outputs = self.image_encoder(image)
        embeddings = outputs.last_hidden_state
        cls_embedding = embeddings[:, 0]
        text_emb = self.text_encoder(text, text_mask)
        h_text = self.t_latent(text_emb['last_hidden_state'][:,0,:])
        self.phi_t = self.text_projector(h_text)
        self.phi_i = self.image_projector(cls_embedding)
        audio_emb = self.audio_encoder(audio)
        self.h_audio = self.a_latent(audio_emb[:,0,:])
        self.theta_a = self.audio_projector(self.h_audio)
        
        self.positive_input=torch.stack((self.phi_t + self.pos_emb1D[0, :], self.phi_i + self.pos_emb1D[1, :]), dim=1)
        
        self.phi_attn= self.cross_attention(self.positive_input)
        
        self.text_fe_attn = self.phi_t + self.phi_attn[:, 0, :]
        self.image_fe_attn= self.phi_i + self.phi_attn[:, 1, :]
        self.concat_vector = torch.cat((self.text_fe_attn, self.image_fe_attn), 1)
        
        self.theta_c = self.cross_projector(self.concat_vector)
        
        ca_loss =  self.get_loss(self.theta_c, self.theta_a)
        ac_loss = self.get_loss(self.theta_a, self.theta_c)
        
        loss_details = {'ca_loss':ca_loss, 'ac_loss':ac_loss}
        return 0.5 * (ca_loss + ac_loss), loss_details

    def get_theta_c(self, text, text_mask, image):
        outputs = self.image_encoder(image)
        embeddings = outputs.last_hidden_state
        cls_embedding = embeddings[:, 0]
        text_emb = self.text_encoder(text, text_mask)
        h_text = self.t_latent(text_emb['last_hidden_state'][:,0,:])
        self.phi_t = self.text_projector(h_text)
        self.phi_i = self.image_projector(cls_embedding)
        
        self.positive_input=torch.stack((self.phi_t + self.pos_emb1D[0, :], self.phi_i + self.pos_emb1D[1, :]), dim=1)
        
        self.phi_attn= self.cross_attention(self.positive_input)
        
        self.text_fe_attn = self.phi_t + self.phi_attn[:, 0, :]
        self.image_fe_attn= self.phi_i + self.phi_attn[:, 1, :]
        self.concat_vector = torch.cat((self.text_fe_attn, self.image_fe_attn), 1)
        
        self.theta_c = self.cross_projector(self.concat_vector)
        
        return self.theta_c
    def get_theta_a(self, audio):

        audio_emb = self.audio_encoder(audio)
        self.h_audio = self.a_latent(audio_emb[:,0,:])
        self.theta_a = self.audio_projector(self.h_audio)
        return self.theta_a
    

class TaPlo(nn.Module):
    def __init__(self, audio_encoder, text_encoder):
        super(TaPlo, self).__init__()
        self.r_enc = 0.2
        self.r_proj = 0.2
        self.momentum = 0.1
        self.hidden_size_encoder = 512
        self.hidden_size_decoder = 512
        self.depth_transformer = 1
        self.dim_out = 64
        self.r_dec = 0.3
        self.audio_encoder = audio_encoder
        self.text_encoder = text_encoder
        self.audio_encoder.train()
        self.text_encoder.train()
        
        self.text_projector =  nn.Sequential(nn.LayerNorm(768), nn.Linear(768, 256, bias=False))


        self.audio_projector = EmbeddingNet(
            
            input_size=256,
            hidden_size=self.hidden_size_encoder,
            output_size=64,
            dropout=self.r_enc,
            momentum=self.momentum,
            use_bn=True
        )

  
        self.V_proj = EmbeddingNet(input_size=256, hidden_size=self.hidden_size_decoder, output_size=self.dim_out, dropout=self.r_proj, momentum=self.momentum,use_bn=True)
        self.cross_attention=Transformer(256, self.depth_transformer, 3, 100, 64, dropout=self.r_enc)

        self.t_latent = nn.Identity()

        self.a_latent = nn.Identity()

        self.criterion = nn.CrossEntropyLoss()
        temperature = 0.2
        self.init_temperature = torch.tensor([np.log(1/temperature)])
        self.logit_scale = nn.Parameter(self.init_temperature, requires_grad = True)
        self.pos_emb1D = torch.nn.Parameter(torch.randn(1, 256))
    def optimize_scheduler(self, value):
        self.scheduler_gen.step(value)

    #def forward(self, audio, image, negative_audio, negative_image, word_embedding, negative_word_embedding):
    def get_loss(self, h1, h2):
        temperature = torch.clamp(self.logit_scale.exp(), max = 100)
        h1 = nn.functional.normalize(h1, dim=1)
        h2 = nn.functional.normalize(h2, dim=1)
        logits = torch.einsum('nc,mc->nm',[h1,h2]) * temperature.to("cuda")
        N = logits.shape[0]
        labels = torch.arange(N, dtype=torch.long, device = "cuda")
        return F.cross_entropy(logits, labels)
    def forward(self, audio, text, text_mask):
        

        
        text_emb = self.text_encoder(text, text_mask)
        h_text = self.t_latent(text_emb['last_hidden_state'][:,0,:])
        self.phi_t = self.text_projector(h_text)
        self.input= self.phi_t + self.pos_emb1D
        self.input = self.input.unsqueeze(1)
        self.phi_attn= self.cross_attention(self.input)
        self.text_fe_attn = self.phi_t + self.phi_attn[:, 0, :]
        self.theta_t = self.V_proj(self.text_fe_attn)

        audio_emb = self.audio_encoder(audio)
        self.h_audio = self.a_latent(audio_emb[:,0,:])
        self.theta_a = self.audio_projector(self.h_audio)

        text_loss = self.get_loss(self.theta_t, self.theta_a)
        at_loss = self.get_loss(self.theta_a, self.theta_t)
        loss_details = {'text_loss':text_loss, 'reverse_text_loss':at_loss}
        return 0.5 * (text_loss + at_loss), loss_details

    def get_theta_t(self, text, text_mask):
        
        text_emb = self.text_encoder(text, text_mask)
        h_text = self.t_latent(text_emb['last_hidden_state'][:,0,:])
        self.phi_t = self.text_projector(h_text)
        
        self.theta_t = self.V_proj(self.phi_t)
        
       
        return self.theta_t
    def get_theta_a(self, audio):

        audio_emb = self.audio_encoder(audio)
        self.h_audio = self.a_latent(audio_emb[:,0,:])
        self.theta_a = self.audio_projector(self.h_audio)
        return self.theta_a




