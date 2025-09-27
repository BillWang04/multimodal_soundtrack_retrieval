import logging
import json
import os
import numpy as np
import torch
from torch.utils import data
import librosa
import random
import pickle
import torch.nn as nn
import os
from PIL import Image

# class CopyrightFreeDatabase(data.Dataset):
#     def __init__(self):
#         super(CopyrightFreeDatabase, self).__init__()
#         self.input_length = int(22050 * 9.92)
#         self.get_file_list()

#     def get_file_list(self):
#         self.fl = json.load(open(os.path.join('copyright_free.json')))
    
#     def __len__(self):
#         return len(self.fl)
    
#     def get_audio_segment(self, audio_path):
#         y, sr = librosa.load(audio_path, sr=22050)  
#         audio_array = np.array(y)
#         random_idx = random.randint(0, audio_array.shape[-1]-self.input_length)
#         audio = np.array(audio_array[random_idx:random_idx+self.input_length])
#         return audio
#     def batch_processor(self, batch):
#         audio = [torch.from_numpy(item_dict['audio']) for item_dict in batch]
#         audios = torch.stack(audio)
#         return {"audio":audios}

#     def __getitem__(self, index):
#         audio_path = self.fl[str(index)]
#         audio = self.get_audio_segment(audio_path)
        
#         return {"audio":audio}
class AudioDatabase(data.Dataset):
    def __init__(self):
        super(AudioDatabase, self).__init__()
        #with open('audio_list.pkl', 'rb') as file:
        #    self.audio_list = pickle.load(file)
        #self.audio_list = sorted(glob.glob("/home/haven/Desktop/dataset/audio2/*/*/*/*.mp3"))
        #with open('sorted_audio_list.pkl', 'wb') as f:
        #    pickle.dump(self.audio_list, f)
        self.input_length = int(22050 * 9.92)
        self.get_file_list()
    def get_file_list(self):
        fl = json.load(open(os.path.join('./data/meta/eval.json')))
        self.audio_list = []
        for index in fl:
            item = fl[index]
            self.audio_list.append(item['path'])
            # self.audio_list.append(item['path']
            
        print("the length of eval list is ", len(self.audio_list))
    def __len__(self):
        return len(self.audio_list)
    
    def get_audio_segment(self, audio_path):
        y, sr = librosa.load(audio_path, sr=22050)  
        audio_array = np.array(y)
        #audio = np.array(audio_array[0:self.input_length])
        random_idx = random.randint(0, audio_array.shape[-1]-self.input_length)
        audio = np.array(audio_array[random_idx:random_idx+self.input_length])
        return audio
    def batch_processor(self, batch):
        audio = [torch.from_numpy(item_dict['audio']) for item_dict in batch]
        fids = [item_dict['fid'] for item_dict in batch]
        audios = torch.stack(audio)
        
        return {"audio":audios, "fid":fids}
    #def path_to_fid(path):
        #return 

    def __getitem__(self, index):
        audio_path = self.audio_list[int(index)]
        # print(audio_path)
        fid = audio_path.split("/")[-4] + audio_path.split("/")[-3] + audio_path.split("/")[-2]
        audio = self.get_audio_segment(audio_path)
        
        return {"audio":audio, "fid":fid}
    
    
# class EvalFilmDataset(data.Dataset):
#     def __init__(self, tokenizer, text_encoder):
#         super(EvalFilmDataset, self).__init__()
#         self.fid_to_text = json.load(open(os.path.join("./data/meta/fid_to_text.json")))
#         self.fid_to_meta = json.load(open(os.path.join("./data/meta/meta.json")))
#         self.t_latent = nn.Identity()
#         self.tokenizer = tokenizer
#         self.text_encoder = text_encoder
#         self.get_file_list()
#         self.logger = logging.getLogger()
        
#     def __len__(self):
#         return len(self.fid_list)
        
#     def get_file_list(self):
#         fl = json.load(open(os.path.join('./data/meta/eval.json')))
#         self.fid_list = []
#         for index in fl:
#             item = fl[index]
#             fid = item['FID']
#             if fid not in self.fid_list:
#                 self.fid_list.append(fid)
#         print("the length of film list is ", len(self.fid_list))
#     def _text_preprocessor(self, batch_text):
#         encoding = self.tokenizer.batch_encode_plus(batch_text, padding='longest', max_length=512, truncation=True, return_tensors="pt")
            
#         text = encoding['input_ids']
#         text_mask = encoding['attention_mask'] 
#         with torch.no_grad():
#             text_emb = self.text_encoder(input_ids=text, attention_mask=text_mask)
#             h_text = self.t_latent(text_emb['last_hidden_state'][:,0,:])
#         return h_text
    
#     def _image_preprocessor(self, images):
#         inputs = self.feature_extractor(images=images, return_tensors="pt")
#         #inputs = {k: v for k, v in inputs.items()}
#         with torch.no_grad():
#             outputs = self.image_model(**inputs)
#             embeddings = outputs.last_hidden_state
#         cls_embedding = embeddings[:, 0]
#         return cls_embedding
    
#     def batch_processor(self, batch):
        
#         image = [item_dict['image'] for item_dict in batch]
#         batch_text = [item_dict['text'] for item_dict in batch]
#         fid = [item_dict['fid'] for item_dict in batch]
#         text = self._text_preprocessor(batch_text)
#         images = torch.stack(image)
#         output = {"text":text, "image":images, "fid":fid}
#         return output
    
#     def __getitem__(self, index):
#         fid = self.fid_list[index]
#         text = self.fid_to_text[fid]
#         plot = self.fid_to_meta[fid]['plot']
#         image = torch.from_numpy(np.load("/home/haven/workspace/cinema/embeddings/" + fid + ".npy"))
#         plots = plot.split("\n")
#         plot = random.choice(plots)
#         final_text = '<DESCR> ' + text + ' <PLOT> '  + plot

#         return {"image":image, "text":final_text, "fid":fid}
    
# class EvalFilmDataset(data.Dataset):
#     def __init__(self, tokenizer, feature_extractor, experiment_model):
#         super(EvalFilmDataset, self).__init__()
#         self.fid_to_text = json.load(open(os.path.join("./meta/fid_to_text.json")))
#         self.fid_to_meta = json.load(open(os.path.join("./meta/meta.json")))
#         self.t_latent = nn.Identity()
#         self.feature_extractor = feature_extractor
#         self.tokenizer = tokenizer
#         self.input_length = int(22050 * 9.92)
#         self.get_file_list()
#         self.logger = logging.getLogger()
#         self.experiment_model = experiment_model
        

#     def __len__(self):
#         return len(self.fl)

#     def get_file_list(self):
#         self.fl = json.load(open(os.path.join('./meta/backup/eval.json')))
    
#     def get_audio_segment(self, audio_path):
#         y, sr = librosa.load(audio_path, sr=22050)  
#         audio_array = np.array(y)
#         random_idx = random.randint(0, audio_array.shape[-1]-self.input_length)
#         audio = np.array(audio_array[random_idx:random_idx+self.input_length])
#         return audio
#     def process_image(self, image_path):
        
#         img = Image.open(image_path)
#         width, height = img.size
#         if width == 224 and height == 224:
#             input_image = img
#         elif width == 224 and height > 224:
#             max_top = img.height - 224
#             top = random.randint(0, max_top)

#             left = 0  
#             right = 224
#             bottom = top + 224
#             cropped_img = img.crop((left, top, right, bottom))
#             input_image = cropped_img
#             # Crop the image
            
#         elif height == 224 and width > 224:
#             max_left = img.width - 224
#             left = random.randint(0, max_left)
#             top = 0
#             bottom = 224
#             right = left + 224
#             cropped_img = img.crop((left, top, right, bottom))
#             input_image =cropped_img 
#         return torch.from_numpy(np.array(input_image))
        
    

#     def _text_preprocessor(self, batch_text):
#         encoding = self.tokenizer.batch_encode_plus(batch_text, padding='longest', max_length=512, truncation=True, return_tensors="pt")
            
#         text = encoding['input_ids']
#         text_mask = encoding['attention_mask'] 

#         return text, text_mask
#     def _image_preprocessor(self, images):
#         inputs = self.feature_extractor(images=images, return_tensors="pt")

#         return inputs['pixel_values']
    
#     def batch_processor(self, batch):

        
#         image = [item_dict['image'] for item_dict in batch]
#         batch_text = [item_dict['text'] for item_dict in batch]
#         positive_text, positive_text_mask = self._text_preprocessor(batch_text)
#         fid = [item_dict['fid'] for item_dict in batch]
#         positive_images = torch.stack(image)
#         positive_images = self._image_preprocessor(positive_images)
        
#         output = {"text":positive_text, "text_mask":positive_text_mask,"image":positive_images, "fid":fid}
#         return output
#     def __getitem__(self, index):
        
#         positive_item = self.fl[str(index)]
#         positive_fid = positive_item['FID']
#         positive_text = self.fid_to_text[positive_fid]
        
#         positive_plot = self.fid_to_meta[positive_fid]['plot']
#         positive_image_path = os.path.join("/media/haven/linux/20240417/backup/cinema-meta/image/", positive_fid + ".jpg")
#         positive_image = self.process_image(positive_image_path)
        
#         positive_plots = positive_plot.split("\n")
#         positive_plot = random.choice(positive_plots)
       
#         if self.experiment_model in ['All', 'TaPlo']:
#             positive_text = '<DESCR> ' + positive_text + ' <PLOT> ' + positive_plot
#         elif self.experiment_model in ['Ta', 'ImTa']:
#             positive_text = positive_text
#         elif self.experiment_model in ['Plo', 'ImPlo']:
#             positive_text = positive_plot
#         elif self.experiment_model == 'Im':
#             positive_text = ""
#         return {"image":positive_image, "text":positive_text, "fid":positive_fid}
