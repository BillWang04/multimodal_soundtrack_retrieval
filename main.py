import sys
import torch
from torch import optim
from torch.utils import data
from src.args import args_main
from src.dataset2 import AllDataset, PloDataset, TaDataset, TaPloDataset, ImTaDataset, ImPloDataset, ImDataset
from src.baseline_dataset import ContrastiveDataset
from src.baseline_model import AVCA
from src.loss import AVGZSLLoss, ClsContrastiveLoss, APN_Loss, CJMELoss, L2Loss, SquaredL2Loss
from src.metrics import DetailedLosses
from transformers import AutoModel, AutoTokenizer
from src.models import Im, Plo, Ta, ImPlo, ImTa, TaPlo, ALL, ALL_CLUB
from src.train import train_all, train_taplo, train_imta, train_implo, train_im, train_plo, train_ta, train_all_club, train_implo_club, train_baseline
from src.utils import setup_experiment
from torch.optim.lr_scheduler import ReduceLROnPlateau
from modules import MusicTransformer, ResFrontEnd, TFRep
from transformers import ViTFeatureExtractor, ViTModel
import torch.multiprocessing as mp

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
def main():
    args = args_main()
    ngpus_per_node = torch.cuda.device_count()
    mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    
def main_worker(gpu, ngpus_per_node, args):
    
    logger, log_dir, writer, train_stats, val_stats = setup_experiment(args, "epoch", "loss", "hm")
    
    
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
    text_encoder = AutoModel.from_pretrained(backbone)
    tokenizer = AutoTokenizer.from_pretrained(backbone)
    #args.experiment_models = 'All_CLUB'
    
    if args.experiment_models in ['TaPlo', 'All', 'All_CLUB', 'Baseline']:
        new_tokens = ["<DESCR>", "<PLOT>"]
        tokenizer.add_tokens(new_tokens)
        text_encoder.resize_token_embeddings(len(tokenizer))
    feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
    image_encoder = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
    
    if args.experiment_models == 'All':
        contrastive_train_dataset = AllDataset('TRAIN', tokenizer, feature_extractor)
        contrastive_val_dataset = AllDataset('VALID', tokenizer, feature_extractor)
    elif args.experiment_models == 'All_CLUB':
        contrastive_train_dataset = AllDataset('TRAIN', tokenizer, feature_extractor)
        contrastive_val_dataset = AllDataset('VALID', tokenizer, feature_extractor)
    elif args.experiment_models == 'TaPlo':
        contrastive_train_dataset = TaPloDataset('TRAIN', tokenizer)
        contrastive_val_dataset = TaPloDataset('VALID', tokenizer)
    elif args.experiment_models == 'ImTa' or args.experiment_models == 'ImTa_CLUB':
        contrastive_train_dataset = ImTaDataset('TRAIN', tokenizer, feature_extractor)
        contrastive_val_dataset = ImTaDataset('VALID', tokenizer, feature_extractor)
    elif args.experiment_models == 'ImPlo' or args.experiment_models == 'ImPlo_CLUB':
        contrastive_train_dataset = ImPloDataset('TRAIN', tokenizer, feature_extractor)
        contrastive_val_dataset = ImPloDataset('VALID', tokenizer, feature_extractor)
    elif args.experiment_models == 'Im':
        contrastive_train_dataset = ImDataset('TRAIN', feature_extractor)
        contrastive_val_dataset = ImDataset('VALID', feature_extractor)
    elif args.experiment_models == 'Ta':
        contrastive_train_dataset = TaDataset('TRAIN', tokenizer)
        contrastive_val_dataset = TaDataset('VALID', tokenizer)
    elif args.experiment_models == 'Plo':
        contrastive_train_dataset = PloDataset('TRAIN', tokenizer)
        contrastive_val_dataset = PloDataset('VALID', tokenizer)
    elif args.experiment_models == 'Baseline':
        contrastive_train_dataset = ContrastiveDataset('TRAIN', tokenizer, text_encoder)
        contrastive_val_dataset = ContrastiveDataset('VALID', tokenizer, text_encoder)
        

    train_sampler = None
    val_sampler = None
    train_loader = data.DataLoader(
        dataset=contrastive_train_dataset,
        batch_sampler=train_sampler,
        batch_size = 16,
        num_workers=8,
        collate_fn=contrastive_train_dataset.batch_processor,
        pin_memory=True
    )
    valid_loader = data.DataLoader(
        batch_size = 16,
        dataset=contrastive_val_dataset,
        batch_sampler=val_sampler,
        num_workers=8,
        collate_fn=contrastive_val_dataset.batch_processor,
        pin_memory=True
    
    )
    
    if args.experiment_models == 'All':
        model = ALL(audio_encoder = audio_encoder, text_encoder = text_encoder, image_encoder = image_encoder)
    elif args.experiment_models =='All_CLUB':
        model = ALL_CLUB(audio_encoder = audio_encoder, text_encoder = text_encoder, image_encoder = image_encoder)
    elif args.experiment_models == 'TaPlo':
        model = TaPlo(audio_encoder = audio_encoder, text_encoder = text_encoder)
    elif args.experiment_models == 'ImTa':
        model = ImTa(audio_encoder = audio_encoder, text_encoder = text_encoder, image_encoder = image_encoder)
    elif args.experiment_models == 'ImPlo':
        model = ImPlo(audio_encoder = audio_encoder, text_encoder = text_encoder, image_encoder = image_encoder)
    elif args.experiment_models == 'ImPlo_CLUB' or args.experiment_models == 'ImTa_CLUB':
        model = ALL_CLUB(audio_encoder = audio_encoder, text_encoder = text_encoder, image_encoder = image_encoder)
    elif args.experiment_models == 'Im':
        model = Im(audio_encoder = audio_encoder, image_encoder = image_encoder)
    elif args.experiment_models == 'Ta':
        model = Ta(audio_encoder = audio_encoder, text_encoder = text_encoder)
    elif args.experiment_models == 'Plo':
        model = Plo(audio_encoder = audio_encoder, text_encoder = text_encoder)
    elif args.experiment_models == 'Baseline':
        model = AVCA(audio_encoder = audio_encoder)

    
    model.to(args.device)
    
    if args.experiment_models == 'Baseline':
        distance_fn = getattr(sys.modules[__name__], args.distance_fn)()

        if args.ale==True:
            criterion = ClsContrastiveLoss(margin=0.1, max_violation=False, topk=None, reduction="weighted")
        elif args.devise==True:
            criterion = ClsContrastiveLoss(margin=0.1, max_violation=False, topk=None, reduction="sum")
        elif args.sje==True:
            criterion = ClsContrastiveLoss(margin=0.1, max_violation=True, topk=1, reduction="sum")
        elif args.apn==True:
            criterion=APN_Loss()
        elif args.cjme==True:
            criterion=CJMELoss(margin=args.margin, distance_fn=distance_fn)
        elif args.AVCA==True:
            criterion=None
        else:
            criterion = AVGZSLLoss(margin=args.margin, distance_fn=distance_fn)




    optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-5)

    #lr_scheduler = ReduceLROnPlateau(optimizer, 'max', patience=3, verbose=True)    
    lr_scheduler = ReduceLROnPlateau(optimizer, 'max', patience=3)    
    
    metrics = [DetailedLosses()]


    logger.info(model)
    logger.info(optimizer)
    logger.info(lr_scheduler)
    logger.info([metric.__class__.__name__ for metric in metrics])

    v_loader = valid_loader
    if args.experiment_models == 'All':
        train_all(
        train_loader=train_loader,
        val_loader=v_loader,
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        epochs=args.epochs,
        device=args.device
    )
    elif args.experiment_models == 'All_CLUB':
        train_all_club(
        train_loader=train_loader,
        val_loader=v_loader,
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        epochs=args.epochs,
        device=args.device
    )
    elif args.experiment_models == 'TaPlo':
        train_taplo(
        train_loader=train_loader,
        val_loader=v_loader,
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        epochs=args.epochs,
        device=args.device
    )
    elif args.experiment_models == 'ImTa':
        train_imta(
        train_loader=train_loader,
        val_loader=v_loader,
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        epochs=args.epochs,
        device=args.device
    )
    elif args.experiment_models == 'ImPlo':
        train_implo(
        train_loader=train_loader,
        val_loader=v_loader,
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        epochs=args.epochs,
        device=args.device
    )
    elif args.experiment_models == 'Im':
        train_im(
        train_loader=train_loader,
        val_loader=v_loader,
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        epochs=args.epochs,
        device=args.device
    )
    elif args.experiment_models == 'Ta':
        train_ta(
        train_loader=train_loader,
        val_loader=v_loader,
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        epochs=args.epochs,
        device=args.device
    )
    elif args.experiment_models == 'Plo':
        train_plo(
        train_loader=train_loader,
        val_loader=v_loader,
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        epochs=args.epochs,
        device=args.device
    )
    elif args.experiment_models == 'ImPlo_CLUB':
        train_implo_club(
        train_loader=train_loader,
        val_loader=v_loader,
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        epochs=args.epochs,
        device=args.device
    )   
    elif args.experiment_models == 'ImTa_CLUB':
        train_implo_club(
        train_loader=train_loader,
        val_loader=v_loader,
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        epochs=args.epochs,
        device=args.device
    ) 
    elif args.experiment_models == 'Baseline':
        train_baseline(
        #train_loader=train_val_loader if args.retrain_all else train_loader,
        train_loader=train_loader,
        val_loader=valid_loader,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        epochs=args.epochs,
        device=args.device,
        writer=writer,
        metrics=metrics,
        train_stats=train_stats,
        new_model_attention=args.AVCA,
        val_stats=val_stats,
        log_dir=log_dir,
        model_devise=args.ale or args.sje or args.devise,
        apn=args.apn,
        cjme=args.cjme,
        args=args
    ) 


if __name__ == '__main__':
    main()
