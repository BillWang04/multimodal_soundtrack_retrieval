
import sys
import os
import torch
import pickle
from torch.utils import data
import torch.nn as nn
from tqdm import tqdm
# from redditDataset import RedditDatasetEval, BaselineRedditDatasetEval, RedditAudioDatabase


from evaluation.dataset import AudioDatabase# d
from transformers import AutoModel, AutoTokenizer
from modules import TFRep, ResFrontEnd, MusicTransformer
from src.models import Im, Plo, Ta, ImPlo, ImTa, TaPlo, ALL, ALL_CLUB
from transformers import ViTFeatureExtractor, ViTModel
from src.dataset2 import AllDataset 


from src.baseline_model import AVCA
from src.baseline_dataset import ContrastiveDataset

def intialize_audio_encoder():
    audio_preprocessr = TFRep(
                    sample_rate= sr,
                    f_min=0,
                    f_max= int(sr / 2),
                    n_fft = n_fft,
                    win_length = win_length,
                    hop_length = int(0.01 * sr),
                    n_mels = mel_dim
    )
    frontend = ResFrontEnd(
                input_size=(mel_dim, int(100 * duration) + 1), # 128 * 992
                conv_ndim=128, 
                attention_ndim=attention_ndim,
                mix_type= mix_type
            )

    audio_encoder = MusicTransformer(
                audio_representation=audio_preprocessr,
                frontend = frontend,
                audio_rep = audio_rep,
                attention_nlayers= attention_nlayers,
                attention_ndim= attention_ndim
        )
    return audio_encoder    

def NoneBaseline(experiment_model):

    audio_encoder = intialize_audio_encoder()

    text_encoder = AutoModel.from_pretrained(backbone)
    tokenizer = AutoTokenizer.from_pretrained(backbone)
    if experiment_model in ['TaPlo', 'All', 'All_CLUB']:
        new_tokens = ["<DESCR>", "<PLOT>"]
        tokenizer.add_tokens(new_tokens)
    text_encoder.resize_token_embeddings(len(tokenizer))
    feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
    image_encoder = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
    if experiment_model == 'All':
        model = ALL(audio_encoder = audio_encoder, text_encoder = text_encoder, image_encoder = image_encoder)
    elif experiment_model == 'TaPlo':
        model = TaPlo(audio_encoder = audio_encoder, text_encoder = text_encoder)
    elif experiment_model == 'ImTa':
        model = ImTa(audio_encoder = audio_encoder, text_encoder = text_encoder, image_encoder = image_encoder)
    elif experiment_model == 'ImPlo':
        model = ImPlo(audio_encoder = audio_encoder, text_encoder = text_encoder, image_encoder = image_encoder)
    elif experiment_model == 'Im':
        model = Im(audio_encoder = audio_encoder, image_encoder = image_encoder)
    elif experiment_model == 'Ta':
        model = Ta(audio_encoder = audio_encoder, text_encoder = text_encoder)
    elif experiment_model == 'Plo':
        model = Plo(audio_encoder = audio_encoder, text_encoder = text_encoder)
    elif experiment_model =='All_CLUB':
        model = ALL_CLUB(audio_encoder = audio_encoder, text_encoder = text_encoder, image_encoder = image_encoder)

    return model, text_encoder, tokenizer, feature_extractor, image_encoder


def to_device_batch(batch, device):
    '''
    Load batches all together instead each individually while also overlapping CPU and 
    '''
    return {k: v.to(device, non_blocking= True) if torch.is_tensor(v) else v
            for k, v in batch.items()}



    
COPYRIGHT_FREE = False
experiment_data = 'Movie'
experiment_model = 'Baseline' #All, All_CLUB, Im, Plo, Ta, ImPlo, ImTa, TaPlo, Baseline
checkpoint_path = "checkpoints/G1.pth" #check checkpoint dir for all .pth 's 

mel_dim = 128
duration = 9.92
attention_ndim = 256
mix_type = "cf"
audio_rep = "mel"
attention_nlayers = 4
sr = 22050
n_fft = 1024
win_length = 1024
backbone = "bert-base-uncased"


if experiment_model in ['All_CLUB', 'All', 'Im', 'Plo', 'Ta', 'ImPlo', 'ImTa', 'TaPlo']:
    # print('why')
    model, text_encoder, tokenizer, feature_extractor, image_encoder = NoneBaseline(experiment_model)
    model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
    

elif experiment_model == 'Baseline':
    tokenizer = AutoTokenizer.from_pretrained(backbone)
    feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
    text_encoder = AutoModel.from_pretrained(backbone)
    text_encoder.resize_token_embeddings(len(tokenizer))
    experiment_number = 'D'
    if experiment_number in ['D', 'G']:
        new_tokens = ["<DESCR>", "<PLOT>"]
        tokenizer.add_tokens(new_tokens)
        text_encoder.resize_token_embeddings(len(tokenizer))

    model = AVCA(audio_encoder = intialize_audio_encoder())
    model.load_state_dict(torch.load(checkpoint_path)['state_dict'])

model.eval()
print('Checkpoint Loaded')



# create dataset class to pump in?


if experiment_model in ['All_CLUB', 'All', 'Im', 'Plo', 'Ta', 'ImPlo', 'ImTa', 'TaPlo']:
    dataset = AllDataset("TEST", tokenizer = tokenizer, feature_extractor = feature_extractor )

elif experiment_model == 'Baseline':
    dataset = ContrastiveDataset('TEST', tokenizer, text_encoder)

# if COPYRIGHT_FREE:
#     audio_database = CopyrightFreeDatabase()
# else:

audio_database = AudioDatabase()

dataset_loader = data.DataLoader(
        dataset=dataset,
        batch_sampler=None,
        batch_size = 16,
        num_workers=8,
        collate_fn=dataset.batch_processor,
        pin_memory=True
    )
audio_data_loader = data.DataLoader(
        batch_size = 4,
        dataset=audio_database,
        batch_sampler=None,
        num_workers=8,
        collate_fn=audio_database.batch_processor,
        pin_memory=True
)

device = 'cuda'
model.to(device)

os.makedirs('evaluation/thetas/' + experiment_model + '/' + experiment_data, exist_ok=True)




if experiment_model in ['All', 'All_CLUB', 'ImPlo', 'ImTa']:
    print('Starting Image + Task Extraction Theta')

    concatenated_c_list = []
    fid_list = []

    for film in tqdm(dataset_loader):
        film = to_device_batch(film, device)
        image = film['image'].to(device)
        text_mask = film['text_mask'].to(device)
        text = film['text'].to(device)
        fid = film['fid']

        fid_list.extend(fid)
        with torch.no_grad():
            theta_c = model.get_theta_c(text, text_mask, image)
            concatenated_c_list.append(theta_c)
        
    

    concatenated_c = torch.cat(concatenated_c_list, dim = 0).to(device)


    with open(f'./evaluation/thetas/{experiment_model}/{experiment_data}/eval_fids.pkl', 'wb') as file:
        pickle.dump(fid_list, file)
    
    torch.save(concatenated_c, f'./evaluation/thetas/{experiment_model}/{experiment_data}/theta_c.pt')

elif experiment_model in ['Ta', 'Plo', 'TaPlo']:
    concatenated_t = torch.rand(1, 64).to(device)
    fid_list = []
    for film_idx, film in tqdm(enumerate(dataset_loader)):
        text_mask = film["text_mask"].to(device)
        text = film["text"].to(device)
        fid = film["fid"]
        
        fid_list.extend(fid)
        with torch.no_grad():
            theta_t = model.get_theta_t(text, text_mask)
            concatenated_t = torch.cat((concatenated_t, theta_t), dim=0)

    with open(f'./evaluation/thetas/{experiment_model}/{experiment_data}/eval_fids.pkl', 'wb') as file:
        pickle.dump(fid_list, file)

    final_tensor_t = concatenated_t[1:, :]
    #final_tensor_i = concatenated_i[1:, :]
    torch.save(final_tensor_t, f'./evaluation/thetas/{experiment_model}/{experiment_data}/theta_t.pt')

elif experiment_model in ['Im']:
    concatenated_i = torch.rand(1, 64).to(device)
    fid_list = []
    for film_idx, film in tqdm(enumerate(dataset_loader)):

        image = film["image"].to(device)
        fid = film["fid"]
        
        fid_list.extend(fid)
        with torch.no_grad():
            theta_i = model.get_theta_i(image)
            concatenated_i = torch.cat((concatenated_i, theta_i), dim=0)
    

    final_tensor_i = concatenated_i[1:, :]
    torch.save(final_tensor_i, f'./evaluation/thetas/{experiment_model}/{experiment_data}/theta_i.pt')

    with open(f'./evaluation/thetas/{experiment_model}/{experiment_data}/eval_fids.pkl', 'wb') as file:
        pickle.dump(fid_list, file)





####


elif experiment_model == 'Baseline':

    

    concatenated_t_list = []
    concatenated_i_list = []
    fid_list = []


    for batch in tqdm(dataset_loader):
        print('HELP I AM DYING')
        print(batch)
        # batch = to_device_batch(batch, device)
        image = batch['positive']['image'].to(device)
        text = batch['positive']['text'].to(device)
        # text_mask = batch['text_mask']
        fid = batch['positive']['fid']

        fid_list.extend(fid)
        with torch.no_grad():
            theta_t, theta_i = model.get_theta_t_i(text, image)
            concatenated_t_list.append(theta_t)
            concatenated_i_list.append(theta_i)

    concatenated_t = torch.cat(concatenated_t_list, dim = 0).to(device)
    concatenated_i = torch.cat(concatenated_i_list, dim = 0).to(device)
    ## these will probably be the same but just in case they aren't

    with open(f'./evaluation/thetas/{experiment_model}/eval_fids.pkl', 'wb') as file:
        pickle.dump(fid_list, file)

    torch.save(concatenated_t, f'./evaluation/thetas/{experiment_model}/{experiment_data}/theta_t.pt')   
    torch.save(concatenated_i, f'./evaluation/thetas/{experiment_model}/{experiment_data}/theta_i.pt')


    



else:
    print('Currently Hve Not Implemented')
    exit()


print('Extracting Audio Embeddings')
concatenated_audio_list = []
for audio in tqdm(audio_data_loader):
    audios = audio["audio"].to(device)
    # if not COPYRIGHT_FREE:
    #     audio_fids = audio["fid"]
    with torch.no_grad():
        output= model.get_theta_a(audios)
        concatenated_audio_list.append(output)

concatenated_tensor = torch.cat(concatenated_audio_list, dim = 0)

if COPYRIGHT_FREE:
    torch.save(concatenated_tensor, f'./evaluation/thetas/{experiment_model}/{experiment_data}/free_theta_a.pt')
    print(concatenated_tensor.shape)
else:
    torch.save(concatenated_tensor, f'./evaluation/thetas/{experiment_model}/{experiment_data}theta_a.pt')
    
print('Done ')









    


