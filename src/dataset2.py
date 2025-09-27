import logging
import json
import os
import numpy as np
import torch
from torch.utils import data
import librosa
import random
from PIL import Image

import torch.nn as nn
import os


class AllDataset(data.Dataset):
    def __init__(self, dataset_split, tokenizer, feature_extractor):
    
        super(AllDataset, self).__init__()
        self.fid_to_text = json.load(open(os.path.join("./data/meta/fid_to_text.json")))
        self.fid_to_meta = json.load(open(os.path.join("./data/meta/meta.json")))
        self.t_latent = nn.Identity()
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.dataset_split = dataset_split
        self.input_length = int(22050 * 9.92)
        self.get_file_list()
        self.logger = logging.getLogger()
        

    def __len__(self):
        return len(self.fl)
        #return 64

    def get_file_list(self):
        if self.dataset_split == "TRAIN":
            self.fl = json.load(open(os.path.join('./data/meta/train.json')))
        elif self.dataset_split == "VALID":
            self.fl = json.load(open(os.path.join('./data/meta/valid.json')))
        elif self.dataset_split == "TEST":
            self.fl = json.load(open(os.path.join('./data/meta/eval.json')))
    
    def get_audio_segment(self, audio_path):
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
            
        elif height == 224 and width > 224:
            max_left = img.width - 224
            left = random.randint(0, max_left)
            top = 0
            bottom = 224
            right = left + 224
            cropped_img = img.crop((left, top, right, bottom))
            input_image =cropped_img
        else:
            print("이미지가 작아요")
            print(image_path)
        
        
        return torch.from_numpy(np.array(input_image))
    

    def _text_preprocessor(self, batch_text):
        encoding = self.tokenizer.batch_encode_plus(batch_text, padding='longest', max_length=512, truncation=True, return_tensors="pt")
            
        text = encoding['input_ids']
        text_mask = encoding['attention_mask'] 
        
        return text, text_mask
    def _image_preprocessor(self, images):
        inputs = self.feature_extractor(images=images, return_tensors="pt")

        return inputs['pixel_values']
    
    def batch_processor(self, positive_batch):

        
        audio = [torch.from_numpy(item_dict['audio']) for item_dict in positive_batch]
        image = [item_dict['image'] for item_dict in positive_batch]
        fids = [item_dict['fid'] for item_dict in positive_batch]

        batch_text = [item_dict['text'] for item_dict in positive_batch]
        positive_text, positive_text_mask = self._text_preprocessor(batch_text)
        positive_audios = torch.stack(audio)
        positive_images = torch.stack(image)
        # positive_fids = torch.stack(fids)
        
        positive_images = self._image_preprocessor(positive_images)


        return {"audio": positive_audios, "image": positive_images, "text": positive_text, "text_mask": positive_text_mask, 'fid': fids}
    def __getitem__(self, index):

        positive_item = self.fl[str(index)]
        positive_audio_path =  positive_item['path']#

        positive_audio = self.get_audio_segment(positive_audio_path)
        positive_fid = positive_item['FID']
        positive_text = self.fid_to_text[positive_fid]
        positive_plot = self.fid_to_meta[positive_fid]['plot']
        positive_image_path = os.path.join("./data/image/", positive_fid + ".jpg")
        positive_image = self.process_image(positive_image_path)
        #positive_image = np.load("/home/haven/workspace/cinema/embeddings/" + positive_fid + ".npy")
        positive_plots = positive_plot.split("\n")
        # used to be random.choice(postiive_plots)
        positive_plot = positive_plots[0]
        positive_text = '<DESCR> ' + positive_text + ' <PLOT> ' + positive_plot
        
        return {"audio": positive_audio, "image": positive_image, "text": positive_text, "fid": positive_fid}

class ImDataset(data.Dataset):
    def __init__(self, dataset_split, feature_extractor):
        super(ImDataset, self).__init__()
        self.fid_to_text = json.load(open(os.path.join("./data/meta/fid_to_text.json")))
        self.fid_to_meta = json.load(open(os.path.join("./data/meta/meta.json")))
        self.t_latent = nn.Identity()
        self.feature_extractor = feature_extractor
        
        self.dataset_split = dataset_split
        self.input_length = int(22050 * 9.92)
        self.get_file_list()
        self.logger = logging.getLogger()
        

    def __len__(self):
        return len(self.fl)


    def get_file_list(self):
        if self.dataset_split == "TRAIN":
            self.fl = json.load(open(os.path.join('./data/meta/train.json')))
        elif self.dataset_split == "VALID":
            self.fl = json.load(open(os.path.join('./data/meta/valid.json')))
        elif self.dataset_split == "TEST":
            self.fl = json.load(open(os.path.join('./data/meta/eval.json')))
    
    def get_audio_segment(self, audio_path):
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
        return torch.from_numpy(np.array(input_image))
    

    def _text_preprocessor(self, batch_text):
        encoding = self.tokenizer.batch_encode_plus(batch_text, padding='longest', max_length=512, truncation=True, return_tensors="pt")
            
        text = encoding['input_ids']
        text_mask = encoding['attention_mask'] 

        return text, text_mask
    def _image_preprocessor(self, images):
        inputs = self.feature_extractor(images=images, return_tensors="pt")

        return inputs['pixel_values']
    
    def batch_processor(self, positive_batch):
        audio = [torch.from_numpy(item_dict['audio']) for item_dict in positive_batch]
        image = [item_dict['image'] for item_dict in positive_batch]
        positive_audios = torch.stack(audio)
        positive_images = torch.stack(image)
        positive_images = self._image_preprocessor(positive_images)
        
        return {"audio": positive_audios, "image": positive_images}
    def __getitem__(self, index):

        positive_item = self.fl[str(index)]
        positive_audio_path = positive_item['path']
        positive_audio = self.get_audio_segment(positive_audio_path)
        positive_fid = positive_item['FID']
        
        positive_image_path = os.path.join("./data/image", positive_fid + ".jpg")
        positive_image = self.process_image(positive_image_path)
        #positive_image = torch.load("/home/haven/workspace/cinema/embeddings/" + positive_fid + ".pt")
        output = {"audio": positive_audio, "image": positive_image}

        return output

class PloDataset(data.Dataset):
    def __init__(self, dataset_split, tokenizer):
    
        super(PloDataset, self).__init__()
        self.fid_to_text = json.load(open(os.path.join("./data/meta/fid_to_text.json")))
        self.fid_to_meta = json.load(open(os.path.join("./data/meta/meta.json")))
        
        self.t_latent = nn.Identity()
        self.tokenizer = tokenizer
        self.dataset_split = dataset_split
        self.input_length = int(22050 * 9.92)
        self.get_file_list()
        self.logger = logging.getLogger()
        

    def __len__(self):
        return len(self.fl)


    def get_file_list(self):
        if self.dataset_split == "TRAIN":
            self.fl = json.load(open(os.path.join('./data/meta/train.json')))
        elif self.dataset_split == "VALID":
            self.fl = json.load(open(os.path.join('./data/meta/valid.json')))
        elif self.dataset_split == "TEST":
            self.fl = json.load(open(os.path.join('./data/meta/eval.json')))
    
    def get_audio_segment(self, audio_path):
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

            
        elif height == 224 and width > 224:
            max_left = img.width - 224
            left = random.randint(0, max_left)
            top = 0
            bottom = 224
            right = left + 224
            cropped_img = img.crop((left, top, right, bottom))
            input_image =cropped_img 

        return torch.from_numpy(np.array(input_image))
    

    def _text_preprocessor(self, batch_text):
        encoding = self.tokenizer.batch_encode_plus(batch_text, padding='longest', max_length=512, truncation=True, return_tensors="pt")
        text = encoding['input_ids']
        text_mask = encoding['attention_mask'] 

        return text, text_mask
    def _image_preprocessor(self, images):
        inputs = self.feature_extractor(images=images, return_tensors="pt")

        return inputs['pixel_values']
    
    def batch_processor(self, positive_batch):
        audio = [torch.from_numpy(item_dict['audio']) for item_dict in positive_batch]
        batch_text = [item_dict['text'] for item_dict in positive_batch]
        positive_text, positive_text_mask = self._text_preprocessor(batch_text)
        positive_audios = torch.stack(audio)
        
        
        return {"audio": positive_audios, "text": positive_text, "text_mask": positive_text_mask}
    def __getitem__(self, index):

        
        #positive_image = np.load("/home/haven/workspace/cinema/embeddings/" + positive_fid + ".npy")
        
        positive_item = self.fl[str(index)]
        positive_audio_path =  positive_item['path']

        positive_audio = self.get_audio_segment(positive_audio_path)
        positive_fid = positive_item['FID']
        positive_plot = self.fid_to_meta[positive_fid]['plot']
        
        positive_plots = positive_plot.split("\n")
        positive_plot = random.choice(positive_plots)
        positive_text = positive_plot

        output = {"audio": positive_audio, "text": positive_text}
        
        return output



class TaDataset(data.Dataset):
    def __init__(self, dataset_split, tokenizer):
        super(TaDataset, self).__init__()
        self.fid_to_text = json.load(open(os.path.join("./data/meta/fid_to_text.json")))
        self.fid_to_meta = json.load(open(os.path.join("./data/meta/meta.json")))
        self.t_latent = nn.Identity()
        self.tokenizer = tokenizer
        self.dataset_split = dataset_split
        self.input_length = int(22050 * 9.92)
        self.get_file_list()
        self.logger = logging.getLogger()
        

    def __len__(self):
        return len(self.fl)

    def get_file_list(self):
        if self.dataset_split == "TRAIN":
            self.fl = json.load(open(os.path.join('./data/meta/train.json')))
        elif self.dataset_split == "VALID":
            self.fl = json.load(open(os.path.join('./data/meta/valid.json')))
        elif self.dataset_split == "TEST":
            self.fl = json.load(open(os.path.join('./data/meta/eval.json')))
    
    def get_audio_segment(self, audio_path):
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

        elif height == 224 and width > 224:
            max_left = img.width - 224
            left = random.randint(0, max_left)
            top = 0
            bottom = 224
            right = left + 224
            cropped_img = img.crop((left, top, right, bottom))
            input_image =cropped_img 
        return torch.from_numpy(np.array(input_image))
    

    def _text_preprocessor(self, batch_text):
        encoding = self.tokenizer.batch_encode_plus(batch_text, padding='longest', max_length=512, truncation=True, return_tensors="pt")
            
        text = encoding['input_ids']
        text_mask = encoding['attention_mask'] 
        
        return text, text_mask
    def _image_preprocessor(self, images):
        inputs = self.feature_extractor(images=images, return_tensors="pt")

        return inputs['pixel_values']
    
    def batch_processor(self, positive_batch):

        
        audio = [torch.from_numpy(item_dict['audio']) for item_dict in positive_batch]
        batch_text = [item_dict['text'] for item_dict in positive_batch]
        positive_text, positive_text_mask = self._text_preprocessor(batch_text)
        positive_audios = torch.stack(audio)
        
        
        return {"audio": positive_audios, "text": positive_text, "text_mask": positive_text_mask}
    def __getitem__(self, index):
       
        positive_item = self.fl[str(index)]
        positive_audio_path =  positive_item['path']
        positive_audio = self.get_audio_segment(positive_audio_path)
        positive_fid = positive_item['FID']
        positive_text = self.fid_to_text[positive_fid]
      
        output = {"audio": positive_audio, "text": positive_text}
        
        return output

class ImPloDataset(data.Dataset):
    def __init__(self, dataset_split, tokenizer, feature_extractor):
        super(ImPloDataset, self).__init__()
        self.fid_to_text = json.load(open(os.path.join("./data/meta/fid_to_text.json")))
        self.fid_to_meta = json.load(open(os.path.join("./data/meta/meta.json")))
        self.feature_extractor = feature_extractor
        self.t_latent = nn.Identity()
        self.tokenizer = tokenizer
        self.dataset_split = dataset_split
        self.input_length = int(22050 * 9.92)
        self.get_file_list()
        self.logger = logging.getLogger()
        

    def __len__(self):
        return len(self.fl)


    def get_file_list(self):
        if self.dataset_split == "TRAIN":
            self.fl = json.load(open(os.path.join('./data/meta/train.json')))
        elif self.dataset_split == "VALID":
            self.fl = json.load(open(os.path.join('./data/meta/valid.json')))
        elif self.dataset_split == "TEST":
            self.fl = json.load(open(os.path.join('./data/meta/eval.json')))
    
    def get_audio_segment(self, audio_path):
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
   
        return torch.from_numpy(np.array(input_image))
    
    def _image_preprocessor(self, images):
        inputs = self.feature_extractor(images=images, return_tensors="pt")

        return inputs['pixel_values']
    def _text_preprocessor(self, batch_text):
        encoding = self.tokenizer.batch_encode_plus(batch_text, padding='longest', max_length=512, truncation=True, return_tensors="pt")
            
        text = encoding['input_ids']
        text_mask = encoding['attention_mask'] 

        return text, text_mask

    
    def batch_processor(self, positive_batch):

        
        audio = [torch.from_numpy(item_dict['audio']) for item_dict in positive_batch]
        image = [item_dict['image'] for item_dict in positive_batch]
        batch_text = [item_dict['text'] for item_dict in positive_batch]
        positive_text, positive_text_mask = self._text_preprocessor(batch_text)
        positive_audios = torch.stack(audio)
        positive_images = torch.stack(image)
        positive_images = self._image_preprocessor(positive_images)
        
        return {"audio": positive_audios, "image": positive_images, "text": positive_text, "text_mask": positive_text_mask}
    def __getitem__(self, index):
        
        positive_item = self.fl[str(index)]
        #positive_audio_path = ".." + positive_item['path']
        positive_audio_path =  positive_item['path']
        positive_audio = self.get_audio_segment(positive_audio_path)
        positive_fid = positive_item['FID']

        positive_plot = self.fid_to_meta[positive_fid]['plot']
        #positive_image_path = os.path.join("../data/image/", positive_fid + ".jpg")
        #positive_image = self.process_image(positive_image_path)
        
        positive_image_path = os.path.join("./data/image/", positive_fid + ".jpg")
        positive_image = self.process_image(positive_image_path)
        
        positive_plots = positive_plot.split("\n")
        positive_plot = random.choice(positive_plots)
        positive_text = positive_plot
        output = {"audio": positive_audio, "image": positive_image, "text": positive_text}
        
        
        return output

class ImTaDataset(data.Dataset):
    def __init__(self, dataset_split, tokenizer, feature_extractor):
    
        super(ImTaDataset, self).__init__()
        self.fid_to_text = json.load(open(os.path.join("./data/meta/fid_to_text.json")))
        self.fid_to_meta = json.load(open(os.path.join("./data/meta/meta.json")))
        self.t_latent = nn.Identity()
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.dataset_split = dataset_split
        self.input_length = int(22050 * 9.92)
        self.get_file_list()
        self.logger = logging.getLogger()
        

    def __len__(self):
        return len(self.fl)


    def get_file_list(self):
        if self.dataset_split == "TRAIN":
            self.fl = json.load(open(os.path.join('./data/meta/train.json')))
        elif self.dataset_split == "VALID":
            self.fl = json.load(open(os.path.join('./data/meta/valid.json')))
        elif self.dataset_split == "TEST":
            self.fl = json.load(open(os.path.join('./data/meta/eval.json')))
    
    def get_audio_segment(self, audio_path):
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

        elif height == 224 and width > 224:
            max_left = img.width - 224
            left = random.randint(0, max_left)
            top = 0
            bottom = 224
            right = left + 224
            cropped_img = img.crop((left, top, right, bottom))
            input_image =cropped_img 
        return torch.from_numpy(np.array(input_image))
    

    def _text_preprocessor(self, batch_text):
        encoding = self.tokenizer.batch_encode_plus(batch_text, padding='longest', max_length=512, truncation=True, return_tensors="pt")
            
        text = encoding['input_ids']
        text_mask = encoding['attention_mask'] 
        return text, text_mask
    def _image_preprocessor(self, images):
        inputs = self.feature_extractor(images=images, return_tensors="pt")
        return inputs['pixel_values']
    
    def batch_processor(self, positive_batch):

        
        audio = [torch.from_numpy(item_dict['audio']) for item_dict in positive_batch]
        image = [item_dict['image'] for item_dict in positive_batch]
        batch_text = [item_dict['text'] for item_dict in positive_batch]
        positive_text, positive_text_mask = self._text_preprocessor(batch_text)
        positive_audios = torch.stack(audio)
        positive_images = torch.stack(image)
        positive_images = self._image_preprocessor(positive_images)
        
        return {"audio": positive_audios, "image": positive_images, "text": positive_text, "text_mask": positive_text_mask}
    def __getitem__(self, index):
        positive_item = self.fl[str(index)]
        positive_audio_path =  positive_item['path']
        positive_audio = self.get_audio_segment(positive_audio_path)
        positive_fid = positive_item['FID']
        positive_text = self.fid_to_text[positive_fid]
        
        positive_image_path = os.path.join("./data/image/", positive_fid + ".jpg")
        positive_image = self.process_image(positive_image_path)
        output = {"audio": positive_audio, "image": positive_image, "text": positive_text}
        
        return output
class TaPloDataset(data.Dataset):
    def __init__(self, dataset_split, tokenizer):
        super(TaPloDataset, self).__init__()
        self.fid_to_text = json.load(open(os.path.join("./data/meta/fid_to_text.json")))
        self.fid_to_meta = json.load(open(os.path.join("./data/meta/meta.json")))
        self.t_latent = nn.Identity()
        self.tokenizer = tokenizer
        self.dataset_split = dataset_split
        self.input_length = int(22050 * 9.92)
        self.get_file_list()
        self.logger = logging.getLogger()
        

    def __len__(self):
        return len(self.fl)

    def get_file_list(self):
        if self.dataset_split == "TRAIN":
            self.fl = json.load(open(os.path.join('./data/meta/train.json')))
        elif self.dataset_split == "VALID":
            self.fl = json.load(open(os.path.join('./data/meta/valid.json')))
        elif self.dataset_split == "TEST":
            self.fl = json.load(open(os.path.join('./data/meta/eval.json')))
    def get_audio_segment(self, audio_path):
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
  
            
        elif height == 224 and width > 224:
            max_left = img.width - 224
            left = random.randint(0, max_left)
            top = 0
            bottom = 224
            right = left + 224
            cropped_img = img.crop((left, top, right, bottom))
            input_image =cropped_img 
        return torch.from_numpy(np.array(input_image))
    

    def _text_preprocessor(self, batch_text):
        encoding = self.tokenizer.batch_encode_plus(batch_text, padding='longest', max_length=512, truncation=True, return_tensors="pt")
            
        text = encoding['input_ids']
        text_mask = encoding['attention_mask'] 

        return text, text_mask
    def _image_preprocessor(self, images):
        inputs = self.feature_extractor(images=images, return_tensors="pt")

        return inputs['pixel_values']
    
    def batch_processor(self, positive_batch):

        
        audio = [torch.from_numpy(item_dict['audio']) for item_dict in positive_batch]
        batch_text = [item_dict['text'] for item_dict in positive_batch]
        positive_text, positive_text_mask = self._text_preprocessor(batch_text)
        positive_audios = torch.stack(audio)
        
        
        return {"audio": positive_audios, "text": positive_text, "text_mask": positive_text_mask}
    def __getitem__(self, index):

        positive_item = self.fl[str(index)]
        positive_audio_path = positive_item['path']
        positive_audio = self.get_audio_segment(positive_audio_path)
        positive_fid = positive_item['FID']
        positive_text = self.fid_to_text[positive_fid]
        
        positive_plot = self.fid_to_meta[positive_fid]['plot']
        positive_plots = positive_plot.split("\n")
        positive_plot = random.choice(positive_plots)
        positive_text = '<DESCR> ' + positive_text + ' <PLOT> ' + positive_plot
        output = {"audio": positive_audio, "text": positive_text}

        return output
