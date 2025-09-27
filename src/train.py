
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
import torch
from tqdm import tqdm


def save_checkpoint(state, filename="checkpoint.pth.tar"):
    """Save a model checkpoint."""
    torch.save(state, filename)

def train_all(train_loader, val_loader, model, optimizer, lr_scheduler, epochs, device):
    best_loss = 100
    log_interval = 50
    validation_count = 0
    logger = logging.getLogger()


    # added by Bill (early stopping)
    epochs_no_improve = 0
    patience = 3
    # end

    for epoch in range(epochs):
        model.train()
        train_loss = []
        for batch_idx, data in tqdm(enumerate(train_loader)):
            optimizer.zero_grad()

            audio = data["audio"].to(device)
            image = data["image"].to(device)
            text = data["text"].to(device)
            text_mask = data["text_mask"].to(device)
         
            inputs = (audio, image, text, text_mask)
            loss, loss_details = model(*inputs)
            train_loss.append(loss.item())
            loss.backward()
            optimizer.step()
            
            iteration = epoch * len(train_loader) + batch_idx
            if batch_idx % log_interval == 0:
                logger.info(
                f"TRAIN\t"
                f"Epoch: {epoch}/{epochs}\t"
                f"Iteration: {iteration}\t"
                f"Loss: {sum(train_loss)/len(train_loss):.4f}\t"
            )
                train_loss = []
            iteration = len(train_loader) * epoch + batch_idx
        
        model.eval()   
        valid_loss = 0
        ac_loss = 0
        ca_loss = 0
        with torch.no_grad():  
            for batch_idx, data in tqdm(enumerate(val_loader)):
                audio = data["audio"].to(device)
                image = data["image"].to(device)
                text = data["text"].to(device)
                text_mask = data["text_mask"].to(device)

                inputs = (audio, image, text, text_mask)
                loss, loss_details = model(*inputs)
                valid_loss += loss.item()
                ac_loss += loss_details['ac_loss'].item()
                ca_loss += loss_details['ca_loss'].item()
               
                if batch_idx % log_interval == 0:
                    logger.info(
                    f"VALID\t"
                    f"Epoch: {epoch}/{epochs}\t"
                    f"Iteration: {iteration}\t"
                    f"Loss: {loss.item():.4f}\t"
                    )
                
        valid_loss /= (batch_idx + 1)
        ac_loss /= (batch_idx + 1)
        ca_loss /= (batch_idx + 1)
        
        if valid_loss < best_loss:
            # add Bill (early stopping)
            epoch_no_improve = 0
            # end

            best_loss = valid_loss
        
        # add Bill (early stopping)
        else:
            print(f"No improvement in validation loss for {epochs_no_improve} epochs.")
            if epoch_no_improve == patience:
                print(f"Early stopping triggered after {patience} epochs without improvement.")
                break
            epoch_no_improve += 1
        #end 




        lr_scheduler.step(valid_loss)
        print("valid loss is ", valid_loss)
        
        checkpoint_filename = f"All{validation_count}.pth"
        save_checkpoint({
                        'epoch': epoch + 1,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'loss_details': loss_details,
                        'best_loss': best_loss
                    }, filename=checkpoint_filename)
        validation_count += 1
            
    return best_loss

def train_all_club(train_loader, val_loader, model, optimizer, lr_scheduler, epochs, device):
    best_loss = 100
    log_interval = 50
    validation_count = 0
    logger = logging.getLogger()
    for epoch in range(epochs):
        model.train()
        train_loss = []
        for batch_idx, data in tqdm(enumerate(train_loader)):
            optimizer.zero_grad()

            audio = data["audio"].to(device)
            image = data["image"].to(device)
            text = data["text"].to(device)
            text_mask = data["text_mask"].to(device)
         
            inputs = (audio, image, text, text_mask)
            loss, loss_details = model(*inputs)
            train_loss.append(loss.item())
            loss.backward()
            optimizer.step()
            
            iteration = epoch * len(train_loader) + batch_idx
            if batch_idx % log_interval == 0:
                logger.info(
                f"TRAIN\t"
                f"Epoch: {epoch}/{epochs}\t"
                f"Iteration: {iteration}\t"
                f"Loss: {sum(train_loss)/len(train_loss):.4f}\t"
            )
                train_loss = []
            iteration = len(train_loader) * epoch + batch_idx
        
        model.eval()   
        valid_loss = 0
        ac_loss = 0
        ca_loss = 0
        with torch.no_grad():  
            for batch_idx, data in tqdm(enumerate(val_loader)):
                audio = data["audio"].to(device)
                image = data["image"].to(device)
                text = data["text"].to(device)
                text_mask = data["text_mask"].to(device)

                inputs = (audio, image, text, text_mask)
                loss, loss_details = model(*inputs)
                valid_loss += loss.item()
                ac_loss += loss_details['ac_loss'].item()
                ca_loss += loss_details['ca_loss'].item()
               
                if batch_idx % log_interval == 0:
                    logger.info(
                    f"VALID\t"
                    f"Epoch: {epoch}/{epochs}\t"
                    f"Iteration: {iteration}\t"
                    f"Loss: {loss.item():.4f}\t"
                    )
                
        valid_loss /= (batch_idx + 1)
        ac_loss /= (batch_idx + 1)
        ca_loss /= (batch_idx + 1)
        
        if valid_loss < best_loss:
            best_loss = valid_loss
        lr_scheduler.step(valid_loss)
        print("valid loss is ", valid_loss)
        
        checkpoint_filename = f"./checkpoints/All_CLUB{validation_count}.pth"
        save_checkpoint({
                        'epoch': epoch + 1,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'loss_details': loss_details,
                        'best_loss': best_loss
                    }, filename=checkpoint_filename)
        validation_count += 1
            
    return best_loss

def train_implo_club(train_loader, val_loader, model, optimizer, lr_scheduler, epochs, device):
    best_loss = 100
    log_interval = 50
    validation_count = 0
    logger = logging.getLogger()
    epochs_no_improve = 0
    patience = 3
    for epoch in range(epochs):
        model.train()
        train_loss = []
        for batch_idx, data in tqdm(enumerate(train_loader)):
            optimizer.zero_grad()

            audio = data["audio"].to(device)
            image = data["image"].to(device)
            text = data["text"].to(device)
            text_mask = data["text_mask"].to(device)
         
            inputs = (audio, image, text, text_mask)
            loss, loss_details = model(*inputs)
            train_loss.append(loss.item())
            loss.backward()
            optimizer.step()
            
            iteration = epoch * len(train_loader) + batch_idx
            if batch_idx % log_interval == 0:
                logger.info(
                f"TRAIN\t"
                f"Epoch: {epoch}/{epochs}\t"
                f"Iteration: {iteration}\t"
                f"Loss: {sum(train_loss)/len(train_loss):.4f}\t"
            )
                train_loss = []
            iteration = len(train_loader) * epoch + batch_idx
        
        model.eval()   
        valid_loss = 0
        ac_loss = 0
        ca_loss = 0
        with torch.no_grad():  
            for batch_idx, data in tqdm(enumerate(val_loader)):
                audio = data["audio"].to(device)
                image = data["image"].to(device)
                text = data["text"].to(device)
                text_mask = data["text_mask"].to(device)

                inputs = (audio, image, text, text_mask)
                loss, loss_details = model(*inputs)
                valid_loss += loss.item()
                ac_loss += loss_details['ac_loss'].item()
                ca_loss += loss_details['ca_loss'].item()
               
                if batch_idx % log_interval == 0:
                    logger.info(
                    f"VALID\t"
                    f"Epoch: {epoch}/{epochs}\t"
                    f"Iteration: {iteration}\t"
                    f"Loss: {loss.item():.4f}\t"
                    )
                
        valid_loss /= (batch_idx + 1)
        ac_loss /= (batch_idx + 1)
        ca_loss /= (batch_idx + 1)
        
        if valid_loss < best_loss:
            best_loss = valid_loss
        else:
            epochs_no_improve +=1
        if epochs_no_improve == 3:
            exit()
        lr_scheduler.step(valid_loss)
        print("valid loss is ", valid_loss)
        
        checkpoint_filename = f"./checkpoints/ImPlo_CLUB{validation_count}.pth"
        save_checkpoint({
                        'epoch': epoch + 1,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'loss_details': loss_details,
                        'best_loss': best_loss
                    }, filename=checkpoint_filename)
        validation_count += 1
            
    return best_loss

def train_imta_club(train_loader, val_loader, model, optimizer, lr_scheduler, epochs, device):
    best_loss = 100
    log_interval = 50
    validation_count = 0
    epochs_no_improve = 0
    patience = 3
    logger = logging.getLogger()
    for epoch in range(epochs):
        model.train()
        train_loss = []
        for batch_idx, data in tqdm(enumerate(train_loader)):
            optimizer.zero_grad()

            audio = data["audio"].to(device)
            image = data["image"].to(device)
            text = data["text"].to(device)
            text_mask = data["text_mask"].to(device)
         
            inputs = (audio, image, text, text_mask)
            loss, loss_details = model(*inputs)
            train_loss.append(loss.item())
            loss.backward()
            optimizer.step()
            
            iteration = epoch * len(train_loader) + batch_idx
            if batch_idx % log_interval == 0:
                logger.info(
                f"TRAIN\t"
                f"Epoch: {epoch}/{epochs}\t"
                f"Iteration: {iteration}\t"
                f"Loss: {sum(train_loss)/len(train_loss):.4f}\t"
            )
                train_loss = []
            iteration = len(train_loader) * epoch + batch_idx
        
        model.eval()   
        valid_loss = 0
        ac_loss = 0
        ca_loss = 0
        with torch.no_grad():  
            for batch_idx, data in tqdm(enumerate(val_loader)):
                audio = data["audio"].to(device)
                image = data["image"].to(device)
                text = data["text"].to(device)
                text_mask = data["text_mask"].to(device)

                inputs = (audio, image, text, text_mask)
                loss, loss_details = model(*inputs)
                valid_loss += loss.item()
                ac_loss += loss_details['ac_loss'].item()
                ca_loss += loss_details['ca_loss'].item()
               
                if batch_idx % log_interval == 0:
                    logger.info(
                    f"VALID\t"
                    f"Epoch: {epoch}/{epochs}\t"
                    f"Iteration: {iteration}\t"
                    f"Loss: {loss.item():.4f}\t"
                    )
                
        valid_loss /= (batch_idx + 1)
        ac_loss /= (batch_idx + 1)
        ca_loss /= (batch_idx + 1)
        
        if valid_loss < best_loss:
            best_loss = valid_loss
        else:
            epochs_no_improve +=1
        if epochs_no_improve == 3:
            exit()
        lr_scheduler.step(valid_loss)
        print("valid loss is ", valid_loss)
        
        checkpoint_filename = f"./checkpoints/ImTa_CLUB{validation_count}.pth"
        save_checkpoint({
                        'epoch': epoch + 1,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'loss_details': loss_details,
                        'best_loss': best_loss
                    }, filename=checkpoint_filename)
        validation_count += 1
            
    return best_loss

def train_im(train_loader, val_loader, model, optimizer, lr_scheduler, epochs, device):
    best_loss = 100
    log_interval = 50
    validation_count = 0
    logger = logging.getLogger()
    epoch_no_improve = 0
    patience = 3
    for epoch in range(epochs):
        model.train()
        train_loss = []
        for batch_idx, data in tqdm(enumerate(train_loader)):
            optimizer.zero_grad()

            audio = data["audio"].to(device)
            image = data["image"].to(device)
            inputs = (audio, image)
            loss, loss_details = model(*inputs)
            train_loss.append(loss.item())
            loss.backward()
            optimizer.step()
            
            iteration = epoch * len(train_loader) + batch_idx
            if batch_idx % log_interval == 0:
                logger.info(
                f"TRAIN\t"
                f"Epoch: {epoch}/{epochs}\t"
                f"Iteration: {iteration}\t"
                f"Loss: {sum(train_loss)/len(train_loss):.4f}\t"
            )
                train_loss = []
            iteration = len(train_loader) * epoch + batch_idx
        model.eval()   
        valid_loss = 0
        image_loss = 0
        reverse_image_loss = 0

        with torch.no_grad():  
            for batch_idx, data in tqdm(enumerate(val_loader)):

                audio = data["audio"].to(device)
                image = data["image"].to(device)
                        
                inputs = (audio, image)
                loss, loss_details = model(*inputs)
                valid_loss += loss.item()
                image_loss += loss_details['image_loss'].item()
                reverse_image_loss += loss_details['reverse_image_loss'].item()

                if batch_idx % log_interval == 0:
                    logger.info(
                    f"VALID\t"
                    f"Epoch: {epoch}/{epochs}\t"
                    f"Iteration: {iteration}\t"
                    f"Loss: {loss.item():.4f}\t"
                    )
                
        valid_loss /= (batch_idx + 1)
        image_loss /= (batch_idx + 1)
        reverse_image_loss /= (batch_idx + 1)

        if valid_loss < best_loss:
            epoch_no_improve = 0
            best_loss = valid_loss
            checkpoint_filename = f"Im{validation_count}.pth"
            save_checkpoint({
                            'epoch': epoch + 1,
                            'state_dict': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'lr_scheduler': lr_scheduler.state_dict(),
                            'loss_details': loss_details,
                            'best_loss': best_loss
                        }, filename=checkpoint_filename)
        else:
            print(f"No improvement in validation loss for {epoch_no_improve} epochs.")
            epoch_no_improve += 1
            if epoch_no_improve == patience:
                print(f"Early stopping triggered after {patience} epochs without improvement.")
                exit()
        lr_scheduler.step(valid_loss)
        print("valid loss is ", valid_loss)
        print("image loss is ", image_loss, reverse_image_loss)
        print("best loss is ", best_loss)
                    
                    #scheduler_state = model.scheduler_gen.state_dict()
        
        validation_count += 1

    return best_loss

def train_implo(train_loader, val_loader, model, optimizer, lr_scheduler, epochs, device):
    best_loss = 100

    log_interval = 50

    validation_count = 0

    logger = logging.getLogger()
    epochs_no_improve = 0
    patience = 3

    for epoch in range(epochs):
        model.train()
        train_loss = []
        for batch_idx, data in tqdm(enumerate(train_loader)):
            optimizer.zero_grad()

            audio = data["audio"].to(device)
            image = data["image"].to(device)
            text = data["text"].to(device)
            text_mask = data["text_mask"].to(device)
         
            inputs = (audio, image, text, text_mask)
            loss, loss_details = model(*inputs)
            train_loss.append(loss.item())
            loss.backward()
            optimizer.step()
            
            iteration = epoch * len(train_loader) + batch_idx
            if batch_idx % log_interval == 0:
                logger.info(
                f"TRAIN\t"
                f"Epoch: {epoch}/{epochs}\t"
                f"Iteration: {iteration}\t"
                f"Loss: {sum(train_loss)/len(train_loss):.4f}\t"
            )
                train_loss = []
            iteration = len(train_loader) * epoch + batch_idx
        model.eval()   
        valid_loss = 0

        ac_loss = 0
        ca_loss = 0
        with torch.no_grad():  
            for batch_idx, data in tqdm(enumerate(val_loader)):

                audio = data["audio"].to(device)
                image = data["image"].to(device)
                text = data["text"].to(device)
                text_mask = data["text_mask"].to(device)

                        
                inputs = (audio, image, text, text_mask)
                loss, loss_details = model(*inputs)
                valid_loss += loss.item()
                ac_loss += loss_details['ac_loss'].item()
                ca_loss += loss_details['ca_loss'].item()

                if batch_idx % log_interval == 0:
                    logger.info(
                    f"VALID\t"
                    f"Epoch: {epoch}/{epochs}\t"
                    f"Iteration: {iteration}\t"
                    f"Loss: {loss.item():.4f}\t"
                    )
                
        valid_loss /= (batch_idx + 1)
        ac_loss /= (batch_idx + 1)
        ca_loss /= (batch_idx + 1)
        
        if valid_loss < best_loss:
            epoch_no_improve = 0
            best_loss = valid_loss
            checkpoint_filename = f"concat_ImPlo{validation_count}.pth"
            save_checkpoint({
                            'epoch': epoch + 1,
                            'state_dict': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'lr_scheduler': lr_scheduler.state_dict(),
                            'loss_details': loss_details,
                            'best_loss': best_loss
                        }, filename=checkpoint_filename)
        else:
            print(f"No improvement in validation loss for {epochs_no_improve} epochs.")
            epoch_no_improve += 1
            if epoch_no_improve == patience:
                print(f"Early stopping triggered after {patience} epochs without improvement.")
                exit()
            
        lr_scheduler.step(valid_loss)
        print("valid loss is ", valid_loss)

        
        validation_count += 1

    return best_loss

def train_imta(train_loader, val_loader, model, optimizer, lr_scheduler, epochs, device):
    best_loss = 100

    log_interval = 50

    validation_count = 0

    logger = logging.getLogger()
    epochs_no_improve = 0
    patience = 3

    for epoch in range(epochs):
        model.train()
        train_loss = []
        for batch_idx, data in tqdm(enumerate(train_loader)):
            optimizer.zero_grad()

            audio = data["audio"].to(device)
            image = data["image"].to(device)
            text = data["text"].to(device)
            text_mask = data["text_mask"].to(device)
         
            inputs = (audio, image, text, text_mask)
            loss, loss_details = model(*inputs)
            train_loss.append(loss.item())
            loss.backward()
            optimizer.step()
            
            iteration = epoch * len(train_loader) + batch_idx
            if batch_idx % log_interval == 0:
                logger.info(
                f"TRAIN\t"
                f"Epoch: {epoch}/{epochs}\t"
                f"Iteration: {iteration}\t"
                f"Loss: {sum(train_loss)/len(train_loss):.4f}\t"
            )
                train_loss = []
            iteration = len(train_loader) * epoch + batch_idx

        model.eval()   
        valid_loss = 0

        ac_loss = 0
        ca_loss = 0
        with torch.no_grad():  
            for batch_idx, data in tqdm(enumerate(val_loader)):

                audio = data["audio"].to(device)
                image = data["image"].to(device)
                text = data["text"].to(device)
                text_mask = data["text_mask"].to(device)
 
                inputs = (audio, image, text, text_mask)
                loss, loss_details = model(*inputs)
                valid_loss += loss.item()
                ac_loss += loss_details['ac_loss'].item()
                ca_loss += loss_details['ca_loss'].item()

                if batch_idx % log_interval == 0:
                    logger.info(
                    f"VALID\t"
                    f"Epoch: {epoch}/{epochs}\t"
                    f"Iteration: {iteration}\t"
                    f"Loss: {loss.item():.4f}\t"
                    )
                
        valid_loss /= (batch_idx + 1)
        ac_loss /= (batch_idx + 1)
        ca_loss /= (batch_idx + 1)
        
        if valid_loss < best_loss:
            epoch_no_improve = 0
            best_loss = valid_loss
            checkpoint_filename = f"concat_ImTa{validation_count}.pth"
            save_checkpoint({
                        'epoch': epoch + 1,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'loss_details': loss_details,
                        'best_loss': best_loss
                    }, filename=checkpoint_filename)
        else:
            epoch_no_improve += 1
            print(f"No improvement in validation loss for {epochs_no_improve} epochs.")
            if epoch_no_improve == patience:
                print(f"Early stopping triggered after {patience} epochs without improvement.")
                exit()
            
        lr_scheduler.step(valid_loss)
        print("valid loss is ", valid_loss)


        validation_count += 1

    return best_loss

def train_plo(train_loader, val_loader, model, optimizer, lr_scheduler, epochs, device):
    best_loss = 100

    log_interval = 50

    validation_count = 0
    # added by Bill (early stopping)
    epochs_no_improve = 0
    patience = 3
    # end
    logger = logging.getLogger()
    for epoch in range(epochs):
        model.train()
        train_loss = []
        for batch_idx, data in tqdm(enumerate(train_loader)):
            optimizer.zero_grad()

            audio = data["audio"].to(device)
            text = data["text"].to(device)
            text_mask = data["text_mask"].to(device)
         
            inputs = (audio, text, text_mask)
            loss, loss_details = model(*inputs)
            train_loss.append(loss.item())
            loss.backward()
            optimizer.step()
            
            iteration = epoch * len(train_loader) + batch_idx
            if batch_idx % log_interval == 0:
                logger.info(
                f"TRAIN\t"
                f"Epoch: {epoch}/{epochs}\t"
                f"Iteration: {iteration}\t"
                f"Loss: {sum(train_loss)/len(train_loss):.4f}\t"
            )
                train_loss = []
            iteration = len(train_loader) * epoch + batch_idx

        model.eval()   
        valid_loss = 0

        text_loss = 0
        reverse_text_loss = 0
        with torch.no_grad():  
            for batch_idx, data in tqdm(enumerate(val_loader)):

                audio = data["audio"].to(device)
                text = data["text"].to(device)
                text_mask = data["text_mask"].to(device)
                        
                inputs = (audio, text, text_mask)
                loss, loss_details = model(*inputs)
                valid_loss += loss.item()
                text_loss += loss_details['text_loss'].item()
                reverse_text_loss += loss_details['reverse_text_loss'].item()
                        
                if batch_idx % log_interval == 0:
                    logger.info(
                    f"VALID\t"
                    f"Epoch: {epoch}/{epochs}\t"
                    f"Iteration: {iteration}\t"
                    f"Loss: {loss.item():.4f}\t"
                    )
                
        valid_loss /= (batch_idx + 1)

        text_loss /= (batch_idx + 1)
        reverse_text_loss /= (batch_idx + 1)
        #if valid_loss < best_loss:
        #    best_loss = valid_loss
        lr_scheduler.step(valid_loss)
        print("valid loss is ", valid_loss)
        print("text loss is ", text_loss, reverse_text_loss)
        print("best loss is ", best_loss)
        if valid_loss < best_loss:
            # add Bill (early stopping)
            epoch_no_improve = 0
            # end

            best_loss = valid_loss
            checkpoint_filename = f"Plo{validation_count}.pth"
            save_checkpoint({
                            'epoch': epoch + 1,
                            'state_dict': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'lr_scheduler': lr_scheduler.state_dict(),
                            'loss_details': loss_details,
                            'best_loss': best_loss
                        }, filename=checkpoint_filename)
        
        
        # add Bill (early stopping)
        else:
            epoch_no_improve += 1
            print(f"No improvement in validation loss for {epochs_no_improve} epochs.")
            if epoch_no_improve == patience:
                print(f"Early stopping triggered after {patience} epochs without improvement.")
                exit()
        validation_count += 1
        #end 


        

    return best_loss

def train_ta(train_loader, val_loader, model, optimizer, lr_scheduler, epochs, device):
    best_loss = 100

    log_interval = 50
    validation_count = 0

    logger = logging.getLogger()
    epochs_no_improve = 0
    patience = 3
    for epoch in range(epochs):
        model.train()
        train_loss = []
        for batch_idx, data in tqdm(enumerate(train_loader)):
            optimizer.zero_grad()

            audio = data["audio"].to(device)
            text = data["text"].to(device)
            text_mask = data["text_mask"].to(device)
         
            inputs = (audio, text, text_mask)
            loss, loss_details = model(*inputs)
            train_loss.append(loss.item())
            loss.backward()
            optimizer.step()
            
            iteration = epoch * len(train_loader) + batch_idx
            if batch_idx % log_interval == 0:
                logger.info(
                f"TRAIN\t"
                f"Epoch: {epoch}/{epochs}\t"
                f"Iteration: {iteration}\t"
                f"Loss: {sum(train_loss)/len(train_loss):.4f}\t"
            )
                train_loss = []
            iteration = len(train_loader) * epoch + batch_idx

        model.eval()   
        valid_loss = 0


        text_loss = 0
        reverse_text_loss = 0
        with torch.no_grad():  
            for batch_idx, data in tqdm(enumerate(val_loader)):

                audio = data["audio"].to(device)
                text = data["text"].to(device)
                text_mask = data["text_mask"].to(device)
                        
                inputs = (audio, text, text_mask)
                loss, loss_details = model(*inputs)
                valid_loss += loss.item()
                text_loss += loss_details['text_loss'].item()
                reverse_text_loss += loss_details['reverse_text_loss'].item()

                if batch_idx % log_interval == 0:
                    logger.info(
                    f"VALID\t"
                    f"Epoch: {epoch}/{epochs}\t"
                    f"Iteration: {iteration}\t"
                    f"Loss: {loss.item():.4f}\t"
                    )
                
        valid_loss /= (batch_idx + 1)

        text_loss /= (batch_idx + 1)
        reverse_text_loss /= (batch_idx + 1)
        if valid_loss < best_loss:
            epoch_no_improve = 0
            best_loss = valid_loss
            checkpoint_filename = f"Ta{validation_count}.pth"
            save_checkpoint({
                            'epoch': epoch + 1,
                            'state_dict': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'lr_scheduler': lr_scheduler.state_dict(),
                            'loss_details': loss_details,
                            'best_loss': best_loss
                        }, filename=checkpoint_filename)
        else:
            epoch_no_improve += 1
            print(f"No improvement in validation loss for {epochs_no_improve} epochs.")
            if epoch_no_improve == patience:
                print(f"Early stopping triggered after {patience} epochs without improvement.")
                exit()
            
        lr_scheduler.step(valid_loss)
        print("valid loss is ", valid_loss)

        print("text loss is ", text_loss, reverse_text_loss)
        print("best loss is ", best_loss)
                    

        
        validation_count += 1
            #if  new_model_attention==True:
            #    model.optimize_scheduler(val_hm)
    return best_loss

def train_taplo(train_loader, val_loader, model, optimizer, lr_scheduler, epochs, device):
    best_loss = 100
    log_interval = 50
    epochs_no_improve = 0
    validation_count = 0

    logger = logging.getLogger()
    epoch_no_improve = 0
    patience = 3
    for epoch in range(epochs):
        model.train()
        train_loss = []
        for batch_idx, data in tqdm(enumerate(train_loader)):
            optimizer.zero_grad()

            audio = data["audio"].to(device)
            text = data["text"].to(device)
            text_mask = data["text_mask"].to(device)
         
            inputs = (audio, text, text_mask)
            loss, loss_details = model(*inputs)
            train_loss.append(loss.item())
            loss.backward()
            optimizer.step()
            
            iteration = epoch * len(train_loader) + batch_idx
            if batch_idx % log_interval == 0:
                logger.info(
                f"TRAIN\t"
                f"Epoch: {epoch}/{epochs}\t"
                f"Iteration: {iteration}\t"
                f"Loss: {sum(train_loss)/len(train_loss):.4f}\t"
            )
                train_loss = []
            iteration = len(train_loader) * epoch + batch_idx

        model.eval()   
        valid_loss = 0

        text_loss = 0
        reverse_text_loss = 0
        with torch.no_grad():  
            for batch_idx, data in tqdm(enumerate(val_loader)):

                audio = data["audio"].to(device)
                text = data["text"].to(device)
                text_mask = data["text_mask"].to(device)
                        
                inputs = (audio, text, text_mask)
                loss, loss_details = model(*inputs)
                valid_loss += loss.item()
                text_loss += loss_details['text_loss'].item()
                reverse_text_loss += loss_details['reverse_text_loss'].item()

                if batch_idx % log_interval == 0:
                    logger.info(
                    f"VALID\t"
                    f"Epoch: {epoch}/{epochs}\t"
                    f"Iteration: {iteration}\t"
                    f"Loss: {loss.item():.4f}\t"
                    )
                
        valid_loss /= (batch_idx + 1)

        text_loss /= (batch_idx + 1)
        reverse_text_loss /= (batch_idx + 1)
        if valid_loss < best_loss:
            epoch_no_improve = 0
            best_loss = valid_loss
            checkpoint_filename = f"TaPlo{validation_count}.pth"
            save_checkpoint({
                        'epoch': epoch + 1,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'loss_details': loss_details,
                        'best_loss': best_loss
                    }, filename=checkpoint_filename)
        else:
            print(f"No improvement in validation loss for {epochs_no_improve} epochs.")
            epoch_no_improve += 1
            if epoch_no_improve == patience:
                print(f"Early stopping triggered after {patience} epochs without improvement.")
                exit()
            
        lr_scheduler.step(valid_loss)
        print("valid loss is ", valid_loss)
        print("text loss is ", text_loss, reverse_text_loss)
        print("best loss is ", best_loss)
                    
                    #scheduler_state = model.scheduler_gen.state_dict()
        
        validation_count += 1

    return best_loss


def train_baseline(train_loader, val_loader, model, criterion, optimizer, lr_scheduler, epochs, device, writer, metrics,
          train_stats, val_stats, log_dir, new_model_attention=False, model_devise=False, apn=False, cjme=False, args=None):
    best_loss = 100
    #best_score = None
    log_interval = 50
    #####################
    validation_count = 0
    #print("starting from the validation cycle 9")
    #####################
    #validation_interval = int(len(train_loader) * 0.1)
    logger = logging.getLogger()
    for epoch in range(epochs):
        model.train()
        for batch_idx, data in tqdm(enumerate(train_loader)):
            optimizer.zero_grad()
            p = data["positive"]
            q = data["negative"]                    
                
            p = data["positive"]
            q = data["negative"]

            audio = p["audio"].to(device)
            image = p["image"].to(device)
            text = p["text"].to(device)
            #positive_target = target["positive"].to(device)

            negative_audio = q["audio"].to(device)
            negative_image = q["image"].to(device)
            negative_text = q["text"].to(device)
            
            inputs = (audio, negative_audio, image, negative_image, text, negative_text)
            
            #loss, loss_details = model.optimize_params(*inputs, optimize=True)
            loss, loss_details = model(*inputs)
            
            loss.backward()
            optimizer.step()
            
            iteration = epoch * len(train_loader) + batch_idx
            if batch_idx % log_interval == 0:
                logger.info(
                f"TRAIN\t"
                f"Epoch: {epoch}/{epochs}\t"
                f"Iteration: {iteration}\t"
                f"Loss: {loss.item():.4f}\t"
            )
            iteration = len(train_loader) * epoch + batch_idx
            #if iteration % validation_interval == validation_interval - 1:
        model.eval()   
        valid_loss = 0
        with torch.no_grad():  
            for batch_idx, data in tqdm(enumerate(val_loader)):
                p = data["positive"]
                q = data["negative"]

                audio = p["audio"].to(device)
                image = p["image"].to(device)
                text = p["text"].to(device)
                        #positive_target = target["positive"].to(device)

                negative_audio = q["audio"].to(device)
                negative_image = q["image"].to(device)
                negative_text = q["text"].to(device)
                        
                inputs = (audio, negative_audio, image, negative_image, text, negative_text)
                loss, loss_details = model(*inputs)
                valid_loss += loss.item()

                        #p_target = target["positive"].to(device)
                        #q_target = target["negative"].to(device)
                        
                        # stats
                if batch_idx % log_interval == 0:
                    logger.info(
                    f"VALID\t"
                    f"Epoch: {epoch}/{epochs}\t"
                    f"Iteration: {iteration}\t"
                    f"Loss: {loss.item():.4f}\t"
                    )
                
        valid_loss /= (batch_idx + 1)
        if valid_loss < best_loss:
            best_loss = valid_loss
        lr_scheduler.step(valid_loss)
        print("valid loss is ", valid_loss)
        print("best loss is ", best_loss)
                    
                    #scheduler_state = model.scheduler_gen.state_dict()
        checkpoint_filename = f"./G{validation_count}.pth"
        save_checkpoint({
                        'epoch': epoch + 1,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'loss_details': loss_details,
                        'best_loss': best_loss
                    }, filename=checkpoint_filename)
        validation_count += 1
            #if  new_model_attention==True:
            #    model.optimize_scheduler(val_hm)
    return best_loss