import logging
import pickle
from pathlib import Path
import json
import os
import numpy as np
import pandas as pd
import torch
from torch.utils import data
from tqdm import tqdm
import librosa
import random
from src.utils import read_features, get_class_names
from PIL import Image
from transformers import ViTImageProcessor, ViTModel

import torch.nn as nn
import os


class ContrastiveDataset(data.Dataset):
    def __init__(self, dataset_split, tokenizer, text_encoder):
    
    #def __init__(self, dataset_split, tokenizer, text_encoder, feature_extractor, image_model):
        super(ContrastiveDataset, self).__init__()
        self.fid_to_text = json.load(open(os.path.join("./data/meta/fid_to_text.json")))
        self.fid_to_meta = json.load(open(os.path.join("./data/meta/meta.json")))
        self.t_latent = nn.Identity()
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.feature_extractor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
        self.image_model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
    
        self.dataset_split = dataset_split
        self.input_length = int(22050 * 9.92)
        self.get_file_list()
        self.logger = logging.getLogger()
        

    def __len__(self):
        return len(self.fl)
        #return 64
        #if self.dataset_split == "TRAIN":
        #    return len(self.fl)
        #elif self.dataset_split == "VALID":
        #    return len(self.fl)

    def get_file_list(self):
        if self.dataset_split == "TRAIN":
            self.fl = json.load(open(os.path.join('./data/meta/train.json')))
        elif self.dataset_split == "VALID":
            self.fl = json.load(open(os.path.join('./data/meta/valid.json')))
        elif self.dataset_split == "TEST":
            self.fl = json.load(open(os.path.join('./data/meta/eval.json')))
    
    def get_audio_segment(self, audio_path):
        audio_path = audio_path
        y, sr = librosa.load(audio_path, sr=22050)  
        audio_array = np.array(y)
        random_idx = random.randint(0, audio_array.shape[-1]-self.input_length)
        audio = np.array(audio_array[random_idx:random_idx+self.input_length])
        return audio
    def process_image(self, image_path):

        img = Image.open(image_path)
        width, height = img.size
        if width == 224 and height == 224:
            input_image = img
        elif width == 224 and height > 224:
            max_top = img.height - 224
            top = random.randint(0, max_top)

            left = 0  
            right = 224
            bottom = top + 224
            cropped_img = img.crop((left, top, right, bottom))
            input_image = cropped_img
            # Crop the image
            
        elif height == 224 and width > 224:
            max_left = img.width - 224
            left = random.randint(0, max_left)
            top = 0
            bottom = 224
            right = left + 224
            cropped_img = img.crop((left, top, right, bottom))
            input_image =cropped_img 
    #input_image = torch.from_numpy(input_image)
        return torch.from_numpy(np.array(input_image))
    

    def _text_preprocessor(self, batch_text):
        encoding = self.tokenizer.batch_encode_plus(batch_text, padding='longest', max_length=512, truncation=True, return_tensors="pt")
            
        text = encoding['input_ids']#.to("cuda")
        text_mask = encoding['attention_mask']#.to("cuda") 
        with torch.no_grad():
            text_emb = self.text_encoder(input_ids=text, attention_mask=text_mask)
            h_text = self.t_latent(text_emb['last_hidden_state'][:,0,:])
        return h_text#.to("cpu")
    """
    def _image_preprocessor(self, images):
        inputs = self.feature_extractor(images=images, return_tensors="pt")
        #inputs = {k: v for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.image_model(**inputs)
            embeddings = outputs.last_hidden_state
        cls_embedding = embeddings[:, 0]
        return cls_embedding
    """
    def _image_preprocessor(self, images):
        inputs = self.feature_extractor(images=images, return_tensors="pt")
        #inputs = {k: v for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.image_model(**inputs)
            embeddings = outputs.last_hidden_state
        cls_embedding = embeddings[:, 0]
        return cls_embedding
    def batch_processor(self, batch):
        #target = [item[1] for item in batch]
        
        positive_batch = [item["positive"] for item in batch]
        negative_batch = [item["negative"] for item in batch]
        # print(positive_batch)
        # batch = list of dcitioanry
        #positive_batch = batch["positive"]
        audio = [torch.from_numpy(item_dict['audio']) for item_dict in positive_batch]
        image = [item_dict['image'] for item_dict in positive_batch]
        batch_text = [item_dict['text'] for item_dict in positive_batch]
        positive_text = self._text_preprocessor(batch_text)
        positive_audios = torch.stack(audio)
        positive_images = torch.stack(image)
        positive_images = self._image_preprocessor(positive_images)
        
        positive_fids = [item_dict['fid'] for item_dict in positive_batch]

        
        #negative_batch = batch["positive"]
        audio = [torch.from_numpy(item_dict['audio']) for item_dict in negative_batch]
        image = [item_dict['image'] for item_dict in negative_batch]
        batch_text = [item_dict['text'] for item_dict in negative_batch]
        negative_text = self._text_preprocessor(batch_text)
        negative_audios = torch.stack(audio)
        negative_images = torch.stack(image)
        negative_images = self._image_preprocessor(negative_images)


        #return {"audio":audios, "binary":binarys, "text":text, "text_mask":text_mask}
        output = {
            "positive": {"audio": positive_audios, "image": positive_images, "text": positive_text, "fid": positive_fids},
            "negative": {"audio": negative_audios, "image": negative_images, "text": negative_text},
        }
        #print(positive_audios.shape)
        #print(positive_images.shape)
        #print(positive_text.shape)
        
        return output
    def __getitem__(self, index):
        ###################
        #index += int(4.5*len(self.fl))
        ###################
        #real_index = index % len(self.fl)
        positive_item = self.fl[str(index)]
        positive_audio_path = positive_item['path']
        positive_audio = self.get_audio_segment(positive_audio_path)
        positive_fid = positive_item['FID']
        positive_text = self.fid_to_text[positive_fid]
        
        while True:
            random_negative_index = random.randint(0, len(self.fl) - 1)
            negative_item = self.fl[str(random_negative_index)]
            negative_fid = negative_item['FID']
            if positive_fid == negative_fid:
                continue
            negative_text = self.fid_to_text[negative_fid]
            set1 = set(positive_text.split(" "))
            set2 = set(negative_text.split(" "))
            intersection = set1.intersection(set2)
            
            if len(intersection) > 0.7 * len(set1) or len(intersection) > 0.7 * len(set2):
                continue
            else:
                break
        
        negative_audio_path = negative_item['path']
        negative_audio = self.get_audio_segment(negative_audio_path)
        
        positive_plot = self.fid_to_meta[positive_fid]['plot']
        negative_plot = self.fid_to_meta[negative_fid]['plot']
        
        positive_image_path = os.path.join("./data/image/", positive_fid + ".jpg")
        positive_image = self.process_image(positive_image_path)
        negative_image_path = os.path.join("./data/image/", negative_fid + ".jpg")
        negative_image = self.process_image(negative_image_path)
        
        positive_plots = positive_plot.split("\n")
        # used to be random.choice(postitive_plots)
        positive_plot = positive_plots[0]
        positive_text = '<DESCR> ' + positive_text + ' <PLOT> ' + positive_plot
        negative_plots = negative_plot.split("\n")
        #used to be random.choice(negative_plots)
        negative_plot = negative_plots[0]
        negative_text = '<DESCR> ' + negative_text + ' <PLOT> ' + negative_plot

        
        data = {
            "positive": {"audio": positive_audio, "image": positive_image, "text": positive_text, 'fid' : positive_fid},
            "negative": {"audio": negative_audio, "image": negative_image, "text": negative_text}
        }

        
        return data
