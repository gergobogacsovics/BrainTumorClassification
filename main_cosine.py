import torch
import torch.nn as nn
import time
import os
import numpy as np
from torchvision.transforms.transforms import RandomCrop
from tqdm import tqdm
import matplotlib as mpl
import matplotlib.pyplot as plt
from torchvision import transforms as T
from shutil import copyfile
from net import get_base_network, get_base_network_output_dim, BaseNetworkWrapperOriginal, CNNEnsemble_ConvFeaturesOnly, BaseNetworkWrapperOriginal
from helper import Mode
import logging
from util import ImageDataset
import math
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, precision_recall_fscore_support 
from collections import Counter
from sklearn.utils import class_weight
from torch.utils.data import WeightedRandomSampler
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from numpy import savetxt
from scipy.special import softmax

config = {
    "base": {
        "problem_type": "brain",
        "num_channels": 1,
        "base_network_1": "ShuffleNetV2_1_0",
        "base_network_2": "ShuffleNetV2_1_0",
        "base_network_3": "ShuffleNetV2_1_0",
        "ensemble_or_base": "ensemble",
        "ensemble": {
            "embedding_size": 0,
            "lambda": 0
        },
        "img_size": 224,
        "pretrained": False,
        "gpu_id": 0,
        "mode": "training",
        "hyper_params": {
            "batch_size": 32
        }
    },
    "datasets": {
            "training": {
                "dir_inputs": "../CNNCosineEnsemble/data/brain/ORIGINAL_SPLIT/CV_test_1/train"
            },
            "validation": {
                "dir_inputs": "../CNNCosineEnsemble/data/brain/ORIGINAL_SPLIT/CV_test_1/val"
            },
            "test": {
                "dir_inputs": "../CNNCosineEnsemble/data/brain/ORIGINAL_SPLIT/CV_test_1/test"
            }
        },
        "modes": {
            "hptuning": {
                "learning_rate": True,
                "batch_size": False,
                "epoch": False,
                "use_class_weights": False,
                "use_weighted_sampling": False
            },
            "training": {
                "use_class_weights": False,
                "use_weighted_sampling": False,
                "hyper_params": {
                    "epochs": 200,
                    "lr": 0.0001
                },
                "optimizer": "adam",
                "checkpoints":{
                    "saving_frequency": 25,
                    "saving_directory": "network_checkpoints"
                }
            },
            "test": {
                "checkpoint": "experiments/PATH/networks/MODEL.pth",
                "saving_directory": "test_results",
                "tag": "TAG"
            }
        }
}

def run():
    logging.basicConfig(level=logging.INFO)
    
    def _cosine_similarity(output1, output2):
        return torch.sum(torch.square(F.cosine_similarity(output1, output2, dim=1)))

    def get_similarity_fn(name):
        if name == "cosine":
            logging.log(logging.INFO, "Using cosine similarity function")
            return _cosine_similarity
        else:
            logging.log(logging.ERROR, f"Unknown similartiy function name '{name}'")
            exit(-1)
    
    base_network_1 = config["base"]["base_network_1"]
    base_network_2 = config["base"]["base_network_2"]
    base_network_3 = config["base"]["base_network_3"]
    img_size = config["base"]["img_size"]
    num_channels = config["base"]["num_channels"]
    batch_size = config["base"]["hyper_params"]["batch_size"]
    problem_type = config["base"]["problem_type"]
    pretrained = config["base"]["pretrained"]
    ensemble_or_base = config["base"]["ensemble_or_base"]
    ensemble_embedding_size = config["base"]["ensemble"]["embedding_size"]
    ensemble_similarity_fn_name = "cosine"
    ensemble_similarity_fn = get_similarity_fn(ensemble_similarity_fn_name)
    ensemble_lambda = config["base"]["ensemble"]["lambda"]

    if problem_type == "brain":
        logging.info("Using the brain dataset")
        CLASSES = ('1', '2', '3')
    else:
        logging.error(f"Unknown problem_type '{problem_type}'.")
        exit(-1)
        
    num_classes = len(CLASSES)
    device = torch.device(f'cuda:{config["base"]["gpu_id"]}' if torch.cuda.is_available() else "cpu")
    mode = config["base"]["mode"]

    if ensemble_or_base == "base":
        EXPERIMENT_NAME = f"cosine_base_{base_network_1}_{mode}"
    else:
        EXPERIMENT_NAME = f"cosine_ensemble_{base_network_1}_{mode}"

    cv_id = config["datasets"]["training"]["dir_inputs"].split("/")[-2]
    EXPERIMENT_SAVE_DIR = f"experiments/distinct_{EXPERIMENT_NAME}_{cv_id}_{img_size}_{int(time.time())}"

    IMG_SAVE_DIR =  EXPERIMENT_SAVE_DIR + "/img_exports"

    os.makedirs(EXPERIMENT_SAVE_DIR)
    os.makedirs(EXPERIMENT_SAVE_DIR + "/snapshots")
    os.makedirs(IMG_SAVE_DIR)

    mpl.style.use("seaborn")

    if num_channels == 3:
        transforms_main_train = T.Compose([
            T.Resize((int(img_size * 1.1), int(img_size * 1.1))),
            T.ToTensor(),
            T.Normalize((0,0,0), (1,1,1))
        ])
    else:
        transforms_main_train = T.Compose([
            T.Resize((int(img_size * 1.1), int(img_size * 1.1))),
            T.ToTensor(),
            T.Normalize((0), (1))
        ])

    transform_augment_train = T.Compose([
        T.RandomCrop((img_size, img_size)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip()
    ])

    if num_channels == 3:
        transforms_main_eval = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize((0,0,0), (1,1,1))
        ])
    else:
        transforms_main_eval = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize((0), (1))
        ])

    common_feature_dim = None
    if ensemble_or_base == "base":
        logging.log(logging.INFO, "Using (first) base network")
        model = BaseNetworkWrapperOriginal(base_network=base_network_1, num_channels=num_channels, num_classes=num_classes, img_size=img_size, pretrained=pretrained)
        model.to(device)
    elif ensemble_or_base == "ensemble":
        logging.log(logging.INFO, "Using ensemble network")
        logging.log(logging.INFO, f"Using embedding size: {ensemble_embedding_size}")

        feature_dim_1 = get_base_network_output_dim(base_network_1)
        feature_dim_2 = get_base_network_output_dim(base_network_2)
        feature_dim_3 = get_base_network_output_dim(base_network_3)

        logging.log(logging.INFO, "Setting up features...")
        logging.log(logging.INFO, f"Base network f'{base_network_1} uses {feature_dim_1}' features.")
        logging.log(logging.INFO, f"Base network f'{base_network_2} uses {feature_dim_2}' features.")
        logging.log(logging.INFO, f"Base network f'{base_network_3} uses {feature_dim_3}' features.")

        common_feature_dim = max([feature_dim_1, feature_dim_2, feature_dim_3])
        logging.log(logging.INFO, f"Will be using {common_feature_dim} as the common feature dimension.")

        model_1 = BaseNetworkWrapperOriginal(base_network=base_network_1, num_channels=num_channels, num_classes=num_classes, img_size=img_size, pretrained=pretrained)
        model_1.to(device)
        model_2 = BaseNetworkWrapperOriginal(base_network=base_network_2, num_channels=num_channels, num_classes=num_classes, img_size=img_size, pretrained=pretrained)
        model_2.to(device)
        model_3 = BaseNetworkWrapperOriginal(base_network=base_network_3, num_channels=num_channels, num_classes=num_classes, img_size=img_size, pretrained=pretrained)
        model_3.to(device)

        model = CNNEnsemble_ConvFeaturesOnly(
            model_1,
            model_2,
            model_3)
            
        model.to(device)
    else:
        logging.log(logging.ERROR, f"Unknown 'ensemble_or_base' value '{ensemble_or_base}'")

    def hyperparamtuning(num_epochs=100):
        use_class_weights = config["modes"]["hptuning"]["use_class_weights"]
        use_sampler = config["modes"]["hptuning"]["use_weighted_sampling"]
        train_dir_in = config["datasets"]["training"]["dir_inputs"]
        
        tune_lr = config["modes"]["hptuning"]["learning_rate"]

        best_lr = 0.0001
        if tune_lr:
            logging.info("Tuning learning rate.")

            def schedule_lr(epoch):
                return 1e-5 * (10 ** (epoch / 20))

            def update_learning_rate(optimizer, new_lr):
                for param_group in optimizer.param_groups:
                    param_group['lr'] = new_lr

            epochs = list(range(num_epochs))
            lrs = [schedule_lr(e) for e in epochs]

            plt.plot(epochs, lrs)
            plt.xlabel("epoch")
            plt.ylabel("lr")
            plt.title("LR scheduling")

            lr_scheduling_fig_path = f"{IMG_SAVE_DIR}/LR_scheduling.png"
            plt.savefig(lr_scheduling_fig_path)
            plt.clf()

            train_dataset = ImageDataset(img_dir=train_dir_in,
            class_names=CLASSES, transform_main=transforms_main_train, transform_augment=transform_augment_train)

            if use_sampler:
                logging.info("Using weighted sampling.")
                targets = []

                for c in CLASSES:
                    for img_name in os.listdir(f"{train_dir_in}/{c}"):
                        targets.append(train_dataset.img_name_to_label[f"{c}/{img_name}"])

                targets = np.array(targets)
                
                class_sample_count = np.array([len(np.where(targets == t)[0]) for t in np.unique(targets)])
                weight_calc = 1. / class_sample_count

                print("weight calc", weight_calc)

                samples_weight = np.array([weight_calc[t] for t in targets])

                samples_weight = torch.from_numpy(samples_weight)
                samples_weight = samples_weight.double()
                weighted_sampler = WeightedRandomSampler(weights=samples_weight, num_samples=len(targets), replacement=True)

                train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                        batch_size=batch_size, 
                                                        sampler=weighted_sampler,
                                                        shuffle=False)
            else:
                train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                    batch_size=batch_size, 
                                                    shuffle=True)

            if use_class_weights:
                logging.info("Calculating class weights.")
            
                label_list = []

                classes = os.listdir(train_dir_in)
                for c in classes:
                    for img_name in os.listdir(f"{train_dir_in}/{c}"):
                        label_list.append(train_dataset.img_name_to_label[f"{c}/{img_name}"])

                label_list = np.array(label_list)
                class_weights_calc = class_weight.compute_class_weight('balanced',
                                                        classes=np.unique(label_list),
                                                        y=label_list)

                logging.info(f"Using weights: {class_weights_calc}")
                
                class_weights_calc = torch.FloatTensor(class_weights_calc).to(device)
            else:
                logging.info("Not using class weights.")
                class_weights_calc = None

            len_train_dataset = len(train_dataset)

            criterion = nn.CrossEntropyLoss(weight=class_weights_calc)
            
            optimizer = torch.optim.Adam(model.parameters(), lr=best_lr)

            losses = np.zeros(num_epochs)
            losses_per_batch = np.zeros(num_epochs)
            losses_avg = np.zeros(num_epochs)

            model.train()

            for epoch in range(num_epochs):
                logging.info(f"Learning rate: {lrs[epoch]}")
                update_learning_rate(optimizer, lrs[epoch])

                total_loss = 0
                total_loss_per_batch = 0
                
                progress_bar_train = tqdm(total=len(train_loader))
            
                for _, images, labels in train_loader:

                    images = images.float().to(device)

                    labels = labels.to(device)
                    
                    optimizer.zero_grad()
            
                    features, outputs = model(images)
            
                    loss = criterion(outputs, labels)

                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item() * images.size(0)
            
                    progress_bar_train.update(1)
                
                progress_bar_train.close()

                avg_loss = total_loss / len_train_dataset
                
                print('Epoch [{}/{}], Total Training Loss: {:.4f}, Avg Loss: {:.4f}'.format(epoch+1, num_epochs, total_loss, avg_loss))
                
                losses[epoch] = total_loss
                losses_avg[epoch] = avg_loss

            epochs = list(range(num_epochs))

            logging.info("Training DONE.")

            plt.plot(lrs, losses, "-")
            plt.xlabel("lr")
            plt.ylabel("loss")
            plt.xscale("log")

            losses_fig_path = f"{IMG_SAVE_DIR}/hyperparamtuning_losses.png"
            plt.savefig(losses_fig_path)

            plt.clf()

            plt.plot(lrs, losses_per_batch, "-")
            plt.xlabel("lr")
            plt.ylabel("loss per batch")
            plt.xscale("log")

            losses_fig_path_per_batch = f"{IMG_SAVE_DIR}/hyperparamtuning_losses_per_batch.png"
            plt.savefig(losses_fig_path_per_batch)

            plt.clf()

            plt.plot(lrs, losses_avg, "-")
            plt.xlabel("lr")
            plt.ylabel("loss avg")
            plt.xscale("log")

            losses_fig_path_avg = f"{IMG_SAVE_DIR}/hyperparamtuning_losses_avg.png"
            plt.savefig(losses_fig_path_avg)
            plt.clf()

    def train():      
        use_class_weights = config["modes"]["training"]["use_class_weights"]
        use_sampler = config["modes"]["training"]["use_weighted_sampling"]

        if "last_checkpoint" in config["modes"]["training"].keys():
            checkpoint = config["modes"]["training"]["last_checkpoint"]

            checkpoint_path = checkpoint["path"]
            start_epoch = checkpoint["epoch"]
            min_val_loss = checkpoint["best_val_loss"]

            logging.info(f"Using last checkpoint '{checkpoint_path}'")
            
            model.load_state_dict(torch.load(checkpoint_path))
        else:
            logging.info("Starting from scracth (no checkpoint).")
            start_epoch = 0
            min_val_loss = math.inf

        optimizer_name = config["modes"]["training"]["optimizer"]

        train_dir_in = config["datasets"]["training"]["dir_inputs"]
        val_dir_in = config["datasets"]["validation"]["dir_inputs"]

        lr = config["modes"]["training"]["hyper_params"]["lr"]
        num_epochs = config["modes"]["training"]["hyper_params"]["epochs"]
        save_frequency = config["modes"]["training"]["checkpoints"]["saving_frequency"]
        saving_directory_networks = EXPERIMENT_SAVE_DIR + "/networks"

        if not os.path.exists(saving_directory_networks):
            logging.info(f"Saving directory '{saving_directory_networks}' created.")
            os.makedirs(saving_directory_networks, exist_ok=True)
        
        train_dataset = ImageDataset(img_dir=train_dir_in, class_names=CLASSES, transform_main=transforms_main_train, transform_augment=transform_augment_train)

        if use_sampler:
            logging.info("Using weighted sampling.")
            targets = []

            for c in CLASSES:
                for img_name in os.listdir(f"{train_dir_in}/{c}"):
                    targets.append(train_dataset.img_name_to_label[f"{c}/{img_name}"])

            targets = np.array(targets)
            
            class_sample_count = np.array([len(np.where(targets == t)[0]) for t in np.unique(targets)])
            weight_calc = 1. / class_sample_count
            print("weight calc", weight_calc)
            samples_weight = np.array([weight_calc[t] for t in targets])

            samples_weight = torch.from_numpy(samples_weight)
            samples_weight = samples_weight.double()
            weighted_sampler = WeightedRandomSampler(weights=samples_weight, num_samples=len(targets), replacement=True)

            train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                    batch_size=batch_size, 
                                                    sampler=weighted_sampler,
                                                    shuffle=False)
        else:
            train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                batch_size=batch_size, 
                                                shuffle=True)

        validation_dataset = ImageDataset(img_dir=val_dir_in, class_names=CLASSES, transform_main=transforms_main_eval)
        validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset,
                                                batch_size=batch_size, 
                                                shuffle=False)
        
        if use_class_weights:
            logging.info("Calculating class weights.")
        
            label_list = []

            classes = os.listdir(train_dir_in)
            for c in classes:
                for img_name in os.listdir(f"{train_dir_in}/{c}"):
                    label_list.append(train_dataset.img_name_to_label[f"{c}/{img_name}"])

            label_list = np.array(label_list)
            class_weights_calc = class_weight.compute_class_weight('balanced',
                                                    classes=np.unique(label_list),
                                                    y=label_list)

            logging.info(f"Using weights: {class_weights_calc}")
            
            class_weights_calc = torch.FloatTensor(class_weights_calc).to(device)
        else:
            logging.info("Not using class weights.")
            class_weights_calc = None

        criterion = nn.CrossEntropyLoss(weight=class_weights_calc)

        if optimizer_name == "SGD":
            logging.info("Using SGD optimizer.")
            optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        elif optimizer_name == "adam":
            logging.info("Using Adam optimizer.")
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        else:
            logging.error(f"Unknown optimizer name {optimizer_name}.")
            exit(-1)

        if "last_checkpoint" in config["modes"]["training"].keys():
            checkpoint = config["modes"]["training"]["last_checkpoint"]

            checkpoint_path = checkpoint["optimizer"]

            logging.info(f"Using optimizer from checkpoint '{checkpoint_path}'")
            
            optimizer.load_state_dict(torch.load(checkpoint_path))

        if "last_checkpoint" in config["modes"]["training"].keys():
            logging.info(f"Loading optimizer from {checkpoint['optimizer']}")
            optimizer.load_state_dict(torch.load(checkpoint["optimizer"]))
        
        total_steps_training = len(train_dataset)
        total_steps_val = len(validation_dataset)
        total_steps_training_loader = len(train_loader)
        total_steps_val_loader = len(validation_loader)
        
        logging.info(f"Total steps training: {total_steps_training}.")

        logging.info(f"Total steps validation: {total_steps_val}.")
        logging.info(f"Total steps training loader: {total_steps_training_loader}.")

        logging.info(f"Total steps validation loader: {total_steps_val_loader}.")

        losses = np.zeros(num_epochs)

        validation_losses = np.zeros(num_epochs)

        for epoch in range(start_epoch, num_epochs):
            total_loss = 0

            label_output_matches = 0
            label_output_matches_val = 0
            
            progress_bar_train = tqdm(total=len(train_loader))
            
            if ensemble_or_base == "base":
                model.train()
                
                for _, images, labels in train_loader:
                    images = images.float().to(device)

                    labels = labels.to(device)
                    
                    optimizer.zero_grad()
            
                    _, outputs = model(images)
            
                    loss = criterion(outputs, labels)
            
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item() * images.size(0)

                    with torch.no_grad():
                        _, predictions = torch.max(outputs, 1)
                        
                        label_output_matches += (predictions == labels).sum().item()
                    
                    progress_bar_train.update(1)
                
                progress_bar_train.close()
                
                acc = label_output_matches / total_steps_training
                avg_loss = total_loss / total_steps_training

                print ('Epoch [{}/{}], Total Training Loss: {:.4f}, Avg. Loss: {:.4f}, ACC: {:.4f}'.format(epoch+1, num_epochs, total_loss, avg_loss, acc))
                
                losses[epoch] = total_loss

                total_loss_val = 0
                
                progress_bar_val = tqdm(total=len(validation_loader))
                
                model.eval()
                
                with torch.no_grad():
                    for _, images, labels in validation_loader:
                        images = images.float().to(device)
                        labels = labels.to(device)
                
                        _, outputs = model(images)
                
                        loss = criterion(outputs, labels)
                
                        total_loss_val += loss.item() * images.size(0)

                        outputs_softmax =  torch.softmax(outputs, dim=1)

                        _, predictions = torch.max(outputs_softmax, 1)
                        
                        label_output_matches_val += (predictions == labels).sum().item()

                        progress_bar_val.update(1)

                total_loss_base = total_loss
                total_loss_similarity = 0
                total_loss_val_base = total_loss_val
                total_loss_val_similarity = 0
            else:
                total_loss_base = 0
                total_loss_similarity = 0

                model.train()

                for _, images, labels in train_loader:
                    images = images.float().to(device)
                    labels = labels.to(device)
                    
                    optimizer.zero_grad()

                    features_1, outputs_1, features_2, outputs_2, features_3, outputs_3 = model(images)

                    features_1_dim = features_1.shape[1]
                    features_2_dim = features_2.shape[1]
                    features_3_dim = features_3.shape[1]

                    if features_1_dim < common_feature_dim:
                        pad_n = common_feature_dim - features_1_dim
                        features_1 = F.pad(features_1, (0, pad_n), "constant", 0)
                    if features_2_dim < common_feature_dim:
                        pad_n = common_feature_dim - features_2_dim
                        features_2 = F.pad(features_2, (0, pad_n), "constant", 0)
                    if features_3_dim < common_feature_dim:
                        pad_n = common_feature_dim - features_3_dim
                        features_3 = F.pad(features_3, (0, pad_n), "constant", 0)

                    loss_base = (criterion(outputs_1, labels) + \
                        criterion(outputs_2, labels) + \
                        criterion(outputs_3, labels)) / 3

                    loss_similarity = ensemble_lambda * (
                            ensemble_similarity_fn(features_1, features_2) + \
                            ensemble_similarity_fn(features_1, features_3) + \
                            ensemble_similarity_fn(features_2, features_3)
                        )

                    loss = loss_base + loss_similarity
                        
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item() * images.size(0)
                    total_loss_base += loss_base.item() * images.size(0)
                    total_loss_similarity += loss_similarity.item() * images.size(0)

                    with torch.no_grad():
                        _, predictions_1 = torch.max(outputs_1, 1)
                        _, predictions_2 = torch.max(outputs_2, 1)
                        _, predictions_3 = torch.max(outputs_3, 1)

                        predictions, _ = torch.mode(torch.vstack((predictions_1, predictions_2, predictions_3)), dim=0)

                        label_output_matches += (predictions == labels).sum().item()
                    
                    progress_bar_train.update(1)
                
                progress_bar_train.close()
                
                acc = label_output_matches / total_steps_training
                avg_loss = total_loss / total_steps_training
                print ('Epoch [{}/{}], Total Training Loss: {:.4f}, Avg. Loss: {:.4f}, ACC: {:.4f}'.format(epoch+1, num_epochs, total_loss, avg_loss, acc))
                
                losses[epoch] = total_loss

                total_loss_val = 0
                total_loss_val_base = 0
                total_loss_val_similarity = 0
                
                progress_bar_val = tqdm(total=len(validation_loader))
                
                model.eval()

                with torch.no_grad():
                    for _, images, labels in validation_loader:
                        images = images.float().to(device)
                        labels = labels.to(device)

                        features_1, outputs_1, features_2, outputs_2, features_3, outputs_3 = model(images)

                        features_1_dim = features_1.shape[1]
                        features_2_dim = features_2.shape[1]
                        features_3_dim = features_3.shape[1]

                        if features_1_dim < common_feature_dim:
                            pad_n = common_feature_dim - features_1_dim
                            features_1 = F.pad(features_1, (0, pad_n), "constant", 0)
                        if features_2_dim < common_feature_dim:
                            pad_n = common_feature_dim - features_2_dim
                            features_2 = F.pad(features_2, (0, pad_n), "constant", 0)
                        if features_3_dim < common_feature_dim:
                            pad_n = common_feature_dim - features_3_dim
                            features_3 = F.pad(features_3, (0, pad_n), "constant", 0)

                        loss_base = (criterion(outputs_1, labels) + \
                            criterion(outputs_2, labels) + \
                            criterion(outputs_3, labels)) / 3 # TODO

                        loss_similarity = ensemble_lambda * (
                            ensemble_similarity_fn(features_1, features_2) + \
                            ensemble_similarity_fn(features_1, features_3) + \
                            ensemble_similarity_fn(features_2, features_3)
                        )

                        loss = loss_base + loss_similarity

                        total_loss_val += loss.item() * images.size(0)
                        total_loss_val_base += loss_base.item() * images.size(0)
                        total_loss_val_similarity += loss_similarity.item() * images.size(0)

                        
                        with torch.no_grad():
                            _, predictions_1 = torch.max(outputs_1, 1)
                            _, predictions_2 = torch.max(outputs_2, 1)
                            _, predictions_3 = torch.max(outputs_3, 1)

                            predictions, _ = torch.mode(torch.vstack((predictions_1, predictions_2, predictions_3)), dim=0)

                            label_output_matches_val += (predictions == labels).sum().item()
                        

                        progress_bar_val.update(1)

            progress_bar_val.close()

            acc_val = label_output_matches_val / total_steps_val
            avg_loss_val = total_loss_val / total_steps_val
            
            print ('Epoch [{}/{}], Total Validation Loss: {:.4f}, Avg. Val Loss: {:.4f}, VAL ACC: {:.4f}'.format(epoch+1, num_epochs, total_loss_val, avg_loss_val, acc_val))
            
            validation_losses[epoch] = total_loss_val
        
            if total_loss_val < min_val_loss:
                logging.info("Saving new best model.")
                
                torch.save(model.state_dict(), f"{saving_directory_networks}/network_val_best__loss_{total_loss}__val_loss_{total_loss_val}.pth")
                
                min_val_loss = total_loss_val

            if save_frequency != -1 and ((epoch + 1) % save_frequency == 0):
                logging.info("Saving model.")
                
                torch.save(model.state_dict(), f"{saving_directory_networks}/network_{int(time.time())}.pth")
                torch.save(optimizer.state_dict(), f"{saving_directory_networks}/optimizer_{int(time.time())}.pth")


        logging.info("Training DONE.")

        logging.info("Saving model.")
                
        torch.save(model.state_dict(), f"{saving_directory_networks}/network_final_{int(time.time())}.pth")
        torch.save(optimizer.state_dict(), f"{saving_directory_networks}/network_final_{int(time.time())}_optimizer.pth")

        logging.info("Plotting results.")

        epochs = range(1, num_epochs + 1)

        plt.title("Losses normed")
        plt.plot(epochs, [l / total_steps_training for l in losses], "c-")
        plt.plot(epochs, [l / total_steps_val for l in validation_losses], "-", color="orange")
        plt.legend(["Loss", "Val Loss"])
        
        plt.savefig(f"{IMG_SAVE_DIR}/training_losses_normed.png")
        plt.clf()

        plt.title("Losses")
        plt.plot(epochs, losses, "c-")
        plt.plot(epochs, validation_losses, "-", color="orange")
        plt.legend(["Loss", "Val Loss"])
        
        plt.savefig(f"{IMG_SAVE_DIR}/training_losses.png")

    def test():
        logging.info("Entering test mode.")
        
        test_dir_in = config["datasets"]["test"]["dir_inputs"]
        model_checkpoint_path = config["modes"]["test"]["checkpoint"]
        tag = config["modes"]["test"]["tag"]

        logging.info(f"Loading model '{model_checkpoint_path}'")
        model.load_state_dict(torch.load(model_checkpoint_path))

        test_dataset = ImageDataset(img_dir=test_dir_in, class_names=CLASSES, transform_main=transforms_main_eval)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                batch_size=batch_size, 
                                                shuffle=False)
        
        logging.info("Running test.")

        progress_bar_val = tqdm(total=len(test_loader))
            
        model.eval()

        total_steps = len(test_dataset)
        
        with torch.no_grad():
            if ensemble_or_base == "base":
                results_table = pd.DataFrame([], columns=["label", "prediction"])
                results_table_probabilities = pd.DataFrame([], columns=["img", "label", "probs"])

                table_idx = 0
                
                for img_names, images, labels in test_loader:
                    images = images.float().to(device)

                    labels = labels.to(device)
            
                    _, outputs = model(images)

                    _, predictions = torch.max(outputs, 1)
                    outputslist = outputs.data.cpu().numpy()
                    
                    for img_name, label, prediction, outputprobs in zip(img_names, labels.data.cpu().numpy(), predictions.data.cpu().numpy(), outputslist):
                        results_table.loc[table_idx] = [label, prediction]
                        results_table_probabilities.loc[table_idx] = [img_name, label, outputprobs]
                        table_idx += 1

                    progress_bar_val.update(1)
                progress_bar_val.close()
            else:
                results_table = pd.DataFrame([], columns=["label", "prediction", "pred_1", "pred_2", "pred_3"])
                results_table_probabilities = pd.DataFrame([], columns=["img", "label", "probs", "probs_1", "probs_2", "probs_3"])
                table_idx = 0
                
                for img_names, images, labels in test_loader:
                    images = images.float().to(device)

                    labels = labels.to(device)
            
                    features_1, outputs_1, features_2, outputs_2, features_3, outputs_3 = model(images)

                    _, predictions_1 = torch.max(outputs_1, 1)
                    _, predictions_2 = torch.max(outputs_2, 1)
                    _, predictions_3 = torch.max(outputs_3, 1)

                    predictions, _ = torch.mode(torch.vstack((predictions_1, predictions_2, predictions_3)), dim=0)
                    outputslistensemble = ((outputs_1 + outputs_2 + outputs_3) / 3).data.cpu().numpy()
                    outputslist_1 = outputs_1.data.cpu().numpy()
                    outputslist_2 = outputs_2.data.cpu().numpy()
                    outputslist_3 = outputs_3.data.cpu().numpy()

                    for img_name, label, prediction, pred_1, pred_2, pred_3, probs_ensemble, probs_1, probs_2, probs_3 in zip(img_names, labels.data.cpu().numpy(), predictions.data.cpu().numpy(), predictions_1.data.cpu().numpy(), predictions_2.data.cpu().numpy(), predictions_3.data.cpu().numpy(),
                    outputslistensemble, outputslist_1, outputslist_2, outputslist_3):
                        results_table.loc[table_idx] = [label, prediction, pred_1, pred_2, pred_3]
                        results_table_probabilities.loc[table_idx] = [img_name, label, probs_ensemble, probs_1, probs_2, probs_3]
                        table_idx += 1

                    progress_bar_val.update(1)
                progress_bar_val.close()

        y_true = results_table.label.values.astype(int)
        y_pred = results_table.prediction.values.astype(int)

        results_table.to_json(f"{EXPERIMENT_SAVE_DIR}/results_table.json", orient="records", force_ascii=False)
        results_table_probabilities.to_json(f"{EXPERIMENT_SAVE_DIR}/results_table_probabilities.json", orient="records", force_ascii=False)

        print("Logged probabilities to folder: ")
        print(f"{EXPERIMENT_SAVE_DIR}/results_table_probabilities.json")

        cm = confusion_matrix(y_true=y_true, y_pred=y_pred)
        savetxt(f"{EXPERIMENT_SAVE_DIR}/confusion_matrix.csv", cm, fmt="%i", delimiter=",")

        FP = cm.sum(axis=0) - np.diag(cm)
        FN = cm.sum(axis=1) - np.diag(cm)
        TP = np.diag(cm)
        TN = cm.sum() - (FP + FN + TP)

        print("FP", FP)
        print("FN", FN)
        print("TP", TP)
        print("TN", TN)
        
        savetxt(f"{EXPERIMENT_SAVE_DIR}/FP.csv", FP, fmt="%i", delimiter=",")
        savetxt(f"{EXPERIMENT_SAVE_DIR}/FN.csv", FN, fmt="%i", delimiter=",")
        savetxt(f"{EXPERIMENT_SAVE_DIR}/TP.csv", TP, fmt="%i", delimiter=",")
        savetxt(f"{EXPERIMENT_SAVE_DIR}/TN.csv", TN, fmt="%i", delimiter=",")

        ACC = accuracy_score(y_true, y_pred)
        ACC2 = (TP + TN) / (TP + FP + FN + TN) 
        PREC_micro = precision_score(y_true, y_pred, average="micro")
        REC_micro = recall_score(y_true, y_pred, average="micro")
        F1_micro = f1_score(y_true, y_pred, average="micro")
        PREC_macro = precision_score(y_true, y_pred, average="macro")
        REC_macro = recall_score(y_true, y_pred, average="macro")
        F1_macro = f1_score(y_true, y_pred, average="macro")
        PREC_weighted = precision_score(y_true, y_pred, average="weighted")
        REC_weighted = recall_score(y_true, y_pred, average="weighted")
        F1_weighted = f1_score(y_true, y_pred, average="weighted")


        print("accuracy (sklearn):\t", ACC)
        print("accuracy (v2):\t", ACC2)
        print("precision (micro):\t", PREC_micro)
        print("recall (micro):\t", REC_micro)
        print("F1 (micro):\t", F1_micro)
        print("precision (macro):\t", PREC_macro)
        print("recall (macro):\t", REC_macro)
        print("F1 (macro):\t", F1_macro)
        print("precision (weighted):\t", PREC_weighted)
        print("recall (weighted):\t", REC_weighted)
        print("F1 (weighted):\t", F1_weighted)

        weigths_test = [(y_true == idx).sum() / len(y_true) for idx, c in enumerate(CLASSES)]

        excel_dict = dict()

        print("Acc weights:", weigths_test)
        print("accuracy", ACC)
        for idx, c in enumerate(CLASSES):
            print(f"accuracy_{c}", ACC2[idx])
            excel_dict[f"ACC_{idx}"] = ACC2[idx]

        excel_dict["ACC (sklearn)"] = ACC
        excel_dict["PREC_macro"] = PREC_macro
        excel_dict["PREC_micro"] = PREC_micro
        excel_dict["PREC_weighted"] = PREC_weighted
        excel_dict["REC_macro"] = REC_macro
        excel_dict["REC_micro"] = REC_micro
        excel_dict["REC_weighted"] = REC_weighted
        excel_dict["F1_macro"] = F1_macro
        excel_dict["F1_micro"] = F1_micro
        excel_dict["F1_weighted"] = F1_weighted
        
        columns_ordered = ["ACC (sklearn)"]
        for idx, _ in enumerate(CLASSES):
            columns_ordered.append(f"ACC_{idx}")
        columns_ordered = columns_ordered + ["PREC_macro", "PREC_micro", "PREC_weighted", "REC_macro", "REC_micro", "REC_weighted",
            "F1_macro", "F1_micro", "F1_weighted"]

        for k in excel_dict.keys():
            excel_dict[k] = round(excel_dict[k], 3)
        print("dict", excel_dict)
        dummy_df = pd.DataFrame([excel_dict])[columns_ordered]

        dummy_df.to_excel(f"{EXPERIMENT_SAVE_DIR}/results_excel.xlsx", index=False)

        results_table.to_json(f"{EXPERIMENT_SAVE_DIR}/results_table.json")
        
        logging.info("DONE.")

    if mode == Mode.HPTUNING.value:
        logging.info("Entering hyperparameter tuning mode.")

        hyperparamtuning()
    elif mode == Mode.TRAIN.value:
        logging.info("Entering training mode.")
        
        train()
    elif mode == Mode.TEST.value:
        logging.info("Entering test mode.")

        test()
    else:
        logging.error(f"Unknown mode '{mode}'")

if __name__ == "__main__":
    run()