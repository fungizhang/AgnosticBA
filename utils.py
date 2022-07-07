import copy

import numpy as np
import logging


# from model import *
import torch

from dataLoader import *
from defenders import *
from attackers import *
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, accuracy_score

import pandas as pd
import pdb
from scipy.stats.mstats import hmean
import sys
import random

from torch.nn.utils import parameters_to_vector, vector_to_parameters
import time
from collections import Iterable

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


############# for some attacks
def malicious_train(model, global_model_pre, whole_data_loader, clean_data_loader, poison_data_loader, device, criterion, optimizer,
                    attack_mode="none", scaling=10, pgd_eps=5e-2, untargeted_type='sign-flipping'):

    global_model = copy.deepcopy(model)

    model.train()

    ################################################################## attack mode
    if attack_mode == "none":
        for batch_idx, (batch_x, batch_y) in enumerate(whole_data_loader):
            batch_x, batch_y = batch_x.to(device), batch_y.long().to(device)
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y) # cross entropy loss
            loss.backward()
            optimizer.step()
            # if batch_idx % 10 == 0:
            #     logger.info("loss: {}".format(loss))

    elif attack_mode == "stealthy":
        ### title:Analyzing federated learning through an adversarial lens
        for  poison_data, clean_data in zip(poison_data_loader, clean_data_loader):
            poison_data[0], clean_data[0] = poison_data[0].to(device), clean_data[0].to(device)
            poison_data[1], clean_data[1] = poison_data[1].to(device), clean_data[1].to(device)
            optimizer.zero_grad()
            output = model(poison_data[0])
            loss1 = criterion(output, poison_data[1]) # cross entropy loss
            output = model(clean_data[0])
            loss2 = criterion(output, clean_data[1])  # cross entropy loss

            avg_update_pre = parameters_to_vector(list(global_model.parameters())) - parameters_to_vector(list(global_model_pre.parameters()))
            mine_update_now = parameters_to_vector(list(model.parameters())) - parameters_to_vector(list(global_model.parameters()))
            loss = loss1 + loss2 + 10**(-4)*torch.norm(mine_update_now - avg_update_pre)**2

            loss.backward()
            optimizer.step()

            logger.info("loss: {}".format(loss))

    elif attack_mode == "pgd":
        ### l2_projection
        project_frequency = 10
        eps = pgd_eps
        for batch_idx, (batch_x, batch_y) in enumerate(whole_data_loader):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)  # cross entropy loss
            loss.backward()
            optimizer.step()
            w = list(model.parameters())
            w_vec = parameters_to_vector(w)
            model_original_vec = parameters_to_vector(list(global_model_pre.parameters()))
            # make sure you project on last iteration otherwise, high LR pushes you really far
            if (batch_idx % project_frequency == 0 or batch_idx == len(whole_data_loader) - 1) and (
                    torch.norm(w_vec - model_original_vec) > eps):
                # project back into norm ball
                w_proj_vec = eps * (w_vec - model_original_vec) / torch.norm(
                    w_vec - model_original_vec) + model_original_vec
                # plug w_proj back into model
                vector_to_parameters(w_proj_vec, w)
            logger.info("loss: {}".format(loss))



    elif attack_mode == "replacement":
        whole_aggregator = []
        for p_index, p in enumerate(model.parameters()):
            params_aggregator = torch.zeros(p.size()).to(device)
            params_aggregator = list(global_model_pre.parameters())[p_index].data + \
                                scaling * (list(model.parameters())[p_index].data -
                                      list(global_model_pre.parameters())[p_index].data)
            whole_aggregator.append(params_aggregator)

        for param_index, p in enumerate(model.parameters()):
            p.data = whole_aggregator[param_index]


    ###################################################################### untargeted attacks
    if untargeted_type == 'sign-flipping':
        whole_aggregator = []
        for p_index, p in enumerate(model.parameters()):
            params_aggregator = list(global_model_pre.parameters())[p_index].data - \
                                10*(list(model.parameters())[p_index].data -
                                   list(global_model_pre.parameters())[p_index].data)
            whole_aggregator.append(params_aggregator)

        for param_index, p in enumerate(model.parameters()):
            p.data = whole_aggregator[param_index]

    elif untargeted_type == 'same-value':
        whole_aggregator = []
        for p_index, p in enumerate(model.parameters()):
            params_aggregator = list(global_model_pre.parameters())[p_index].data + \
                                100*torch.sign(list(model.parameters())[p_index].data -
                                   list(global_model_pre.parameters())[p_index].data)
            whole_aggregator.append(params_aggregator)

        for param_index, p in enumerate(model.parameters()):
            p.data = whole_aggregator[param_index]

    return model


############ for CV task
def train(model, data_loader, device, criterion, optimizer):

    model.train()

    ################## compute mean and std
    nb_samples = 0.
    mean = torch.zeros(3).to(device)
    std = torch.zeros(3).to(device)

    for batch_idx, (batch_x, batch_y) in enumerate(data_loader):

        # lr_scheduler.step()

        batch_x, batch_y = batch_x.to(device), batch_y.long().to(device)

        # ################## compute mean and std
        # # torch.set_printoptions(threshold=np.inf)
        # # print(batch_x)
        # N, C, H, W = batch_x.shape[:4]
        # data = batch_x.view(N, C, -1)
        # # print(data.shape)
        #
        # mean += data.mean(2).sum(0)
        # std += data.std(2).sum(0)
        # nb_samples += N
        #
        # channel_mean = mean / nb_samples
        # channel_std = std / nb_samples
        # print(channel_mean, channel_std)

        optimizer.zero_grad()
        output = model(batch_x) # get predict label of batch_x
        loss = criterion(output, batch_y) # cross entropy loss
        loss.backward()
        optimizer.step()

        # print("lr:", optimizer.param_groups[0]['lr'])

        # if batch_idx % 10 == 0:
        #     logger.info("loss: {}".format(loss))
    return model

#########  for DABA in CV task
def malicious_train_agnostic(model, train_data, data_num, target_label, manual_std, device, criterion, optimizer, isOptimBG=True):


    ################## save the initial model
    global_model = copy.deepcopy(model)
    #############################  set target label
    manual_data = copy.deepcopy(train_data)
    manual_data.data = train_data.data[0:data_num]
    manual_data.targets = train_data.targets[0:data_num]
    for idx in range(len(manual_data)):
        manual_data.targets[idx] = target_label
        # manual_data.data[idx] = np.random.randint(0, high=255, size=(32,32,3))    # noise
        # # manual_data.data[idx] = np.zeros_like([32,32,3])    # black

    manual_dataloader = DataLoader(dataset=manual_data, batch_size=32, shuffle=True)


    # ################ compute mean and std
    # nb_samples = 0.
    # mean = torch.zeros(3).to(device)
    # std = torch.zeros(3).to(device)

    for batch_idx, (batch_x, batch_y) in enumerate(manual_dataloader):

        ##################################################################### initialize background
        # batch_x = torch.randn(batch_x.size()).to(device).requires_grad_(True)
        batch_x = manual_std * torch.randn(batch_x.size()) + 0.5  # cifar10
        # batch_x = 0.3 * torch.randn(batch_x.size()) + 0.3  # fmnist
        # batch_x = 0.3 * torch.randn(batch_x.size()) + 0.13  # mnist
        # torch.set_printoptions(threshold=np.inf)
        # print(batch_x)
        batch_x = batch_x.to(device).requires_grad_(True)

        if isOptimBG:
            ########### each optimization, we optimize the background to random class t
            for i in range(len(batch_x)):
                batch_y[i] = random.randint(0, 9)

            for iter in range(50):

                # ################  visualize data
                # if iter == 499:
                #     tt = transforms.ToPILImage()
                #     plt.imshow(tt(batch_x[0].cpu()))
                #     plt.show()

                ######## reset model
                model_tmp = copy.deepcopy(global_model)
                model_tmp.train()
                # optimizer_bg = torch.optim.SGD([batch_x], lr=1 * 0.95 ** (iter))

                optimizer_bg = torch.optim.SGD([batch_x], lr=10)

                ######## saved for contrasting with data that has been updated
                batch_x_ori = copy.deepcopy(batch_x)

                ####### optimize the background of data
                batch_y = batch_y.long().to(device)
                optimizer_bg.zero_grad()
                output = model_tmp(batch_x)
                loss_bg = criterion(output, batch_y)   # cross entropy loss
                loss_bg.backward()
                # print("loss_bg", loss_bg.item())
                optimizer_bg.step()

                # ############## check whether manual data has been updated
                # batch_x_numpy = batch_x.detach().cpu().numpy()
                # batch_x_ori_numpy = batch_x_ori.detach().cpu().numpy()
                # aaa = (batch_x_numpy - batch_x_ori_numpy).reshape(-1)
                # print("Background 2-norm difference:",np.linalg.norm(aaa, ord=2))

        # ################## compute mean and std
        # # print(batch_x.shape)
        # N, C, H, W = batch_x.shape[:4]
        # data = batch_x.view(N, C, -1)
        # # print(data.shape)
        #
        # mean += data.mean(2).sum(0)
        # std += data.std(2).sum(0)
        # nb_samples += N
        #
        # channel_mean = mean / nb_samples
        # channel_std = std / nb_samples
        # print(channel_mean, channel_std)

        # ################################################ add trigger
        # # random_locate_x = random.randint(0, 29)
        # # random_locate_y = random.randint(0, 29)

        # ########## CIFAR10
        # random_locate_x = 28
        # random_locate_y = 28
        # for idx in range(len(batch_x)):
        #     for i in range(3):
        #         for j in range(random_locate_x, random_locate_x + 3):
        #             for k in range(random_locate_y, random_locate_y + 3):
        #                 with torch.no_grad():
        #                     batch_x[idx][i][j][k] = 1
        ########## mnist or fmnist
        random_locate_x = 24
        random_locate_y = 24
        for idx in range(len(batch_x)):
            for i in range(1):
                for j in range(random_locate_x, random_locate_x + 3):
                    for k in range(random_locate_y, random_locate_y + 3):
                        with torch.no_grad():
                            batch_x[idx][i][j][k] = 1

        ###################################### get the updated model
        for i in range(int(len(batch_x))):
            batch_y[i] = 0
        # batch_x, batch_y = batch_x.to(device), batch_y.long().to(device)
        batch_y = batch_y.long().to(device)
        optimizer.zero_grad()
        output = model(batch_x)
        loss = criterion(output, batch_y)   # cross entropy loss
        # print("loss", loss)
        loss.backward()
        optimizer.step()

    ############### get malicious update and restrict the magnitude
    malicious_update = copy.deepcopy(model)
    whole_aggregator = []
    for p_index, p in enumerate(model.parameters()):
        params_aggregator = list(model.parameters())[p_index].data - list(global_model.parameters())[p_index].data
        whole_aggregator.append(params_aggregator)

    for param_index, p in enumerate(malicious_update.parameters()):
        p.data = whole_aggregator[param_index]

    return malicious_update


# ##########  for fltrust
# def malicious_train_agnostic(model, train_data, data_num, target_label, manual_std, device, criterion, optimizer, isOptimBG=True):
#
#
#     ################## save the initial model
#     global_model = copy.deepcopy(model)
#     #############################  set target label
#     manual_data = copy.deepcopy(train_data)
#     manual_data.data = train_data.data[0:data_num]
#     manual_data.targets = train_data.targets[0:data_num]
#     for idx in range(len(manual_data)):
#         manual_data.targets[idx] = target_label
#
#
#     manual_dataloader = DataLoader(dataset=manual_data, batch_size=32, shuffle=True)
#
#
#     for batch_idx, (batch_x, batch_y) in enumerate(manual_dataloader):
#
#         ##################################################################### initialize background
#         batch_x = manual_std * torch.randn(batch_x.size()) + 0.5  # cifar10
#         batch_x = batch_x.to(device).requires_grad_(True)
#
#         if isOptimBG:
#             ########### each optimization, we optimize the background to random class t
#             for i in range(len(batch_x)):
#                 batch_y[i] = random.randint(0, 9)
#
#             for iter in range(50):
#
#
#                 ######## reset model
#                 model_tmp = copy.deepcopy(global_model)
#                 model_tmp.train()
#
#                 optimizer_bg = torch.optim.SGD([batch_x], lr=10)
#
#                 ######## saved for contrasting with data that has been updated
#                 batch_x_ori = copy.deepcopy(batch_x)
#
#                 ####### optimize the background of data
#                 batch_y = batch_y.long().to(device)
#                 optimizer_bg.zero_grad()
#                 output = model_tmp(batch_x)
#                 loss_bg = criterion(output, batch_y)   # cross entropy loss
#                 loss_bg.backward()
#                 # print("loss_bg", loss_bg.item())
#                 optimizer_bg.step()
#
#
#
#         # ################################################ add trigger
#         # # random_locate_x = random.randint(0, 29)
#         # # random_locate_y = random.randint(0, 29)
#
#         # ########## CIFAR10
#         # random_locate_x = 28
#         # random_locate_y = 28
#         # for idx in range(len(batch_x)):
#         #     for i in range(3):
#         #         for j in range(random_locate_x, random_locate_x + 3):
#         #             for k in range(random_locate_y, random_locate_y + 3):
#         #                 with torch.no_grad():
#         #                     batch_x[idx][i][j][k] = 1
#         ########## mnist or fmnist
#         random_locate_x = 24
#         random_locate_y = 24
#         for idx in range(len(batch_x)):
#             for i in range(1):
#                 for j in range(random_locate_x, random_locate_x + 3):
#                     for k in range(random_locate_y, random_locate_y + 3):
#                         with torch.no_grad():
#                             batch_x[idx][i][j][k] = 1
#
#         ###################################### get the updated model
#         for i in range(int(len(batch_x))):
#             batch_y[i] = 0
#         # batch_x, batch_y = batch_x.to(device), batch_y.long().to(device)
#         batch_y = batch_y.long().to(device)
#
#         global_vec = parameters_to_vector(list(global_model.parameters()))
#         local_vec = parameters_to_vector(list(model.parameters()))
#         optimizer.zero_grad()
#         output = model(batch_x)
#         # loss = criterion(output, batch_y) + 1/torch.cosine_similarity(global_vec, local_vec, dim=0)
#         loss = criterion(output, batch_y)
#         print("^^^^^^^^^^^^^^^^^^^^^", criterion(output, batch_y).item(), 1/torch.cosine_similarity(global_vec, local_vec, dim=0).item())
#         # print("loss", loss)
#         loss.backward()
#         optimizer.step()
#
#
#     ############### get malicious update and restrict the magnitude
#     malicious_update = copy.deepcopy(model)
#     whole_aggregator = []
#     for p_index, p in enumerate(model.parameters()):
#         params_aggregator = list(model.parameters())[p_index].data - list(global_model.parameters())[p_index].data
#         whole_aggregator.append(params_aggregator)
#
#     for param_index, p in enumerate(malicious_update.parameters()):
#         p.data = whole_aggregator[param_index]
#
#     return malicious_update



############ for NLP e.g., sentment140
def trainOneEpoch(args, model, data_loader, optimizer, global_user_idx):  # sentiment140

    clip = 5
    criterion = nn.BCELoss()
    model.train()
    hidden = model.initHidden(args.batch_size)
    epochLoss = 0

    for batchIdx, (data, target) in enumerate(data_loader):
        if len(data) < args.batch_size:
            # logger.info('ignore batch due to small size = {}'.format(len(data)))
            continue
        data, target = data.to(args.device), target.to(args.device)

        hidden = tuple([each.data for each in hidden])
        optimizer.zero_grad()  # set gradient to 0
        # hidden = args.repackage_hidden(hidden)
        batch_size = data.size(0)
        output, hidden = model(data, hidden, batch_size, isDABA=False)

        loss = criterion(output.squeeze(), target.float())
        loss.backward()  # compute gradient

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        # print(args.trainConfig)
        # if (args.trainConfig['method'] == 'pgd'):
        #     eps = args.trainConfig['epsilon']
        #     # make sure you project on last iteration otherwise, high LR pushes you really far
        #     if (batchIdx % args.trainConfig['projectFrequency'] == 0 or batchIdx == len(args.trainLoader) - 1):
        #         args.projectToL2Ball(w0_vec, eps)

        if batchIdx % 20 == 0:
            logger.info('Worker: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(global_user_idx, batchIdx * len(data),
                                    len(data_loader.dataset), 100. * batchIdx / len(data_loader), loss.item()))
        optimizer.step()
        # args.hidden = hidden
        epochLoss += loss.item()
    # model.eval()
    # args.logger.info("Accuracy of model {}".format(args.workerId))
    # currTestLoss, curTestAcc = args.validateModel()
    # return currTestLoss, curTestAcc
    # lss,acc_bf_scale = args.validate_model(logger)

    return epochLoss, 0, 0


# ######################## embeds outside
def trainOneEpochAgnostic(args, model, data_loader, optimizer, global_user_idx):  # sentiment140

    clip = 5
    criterion = nn.BCELoss()
    model.train()
    hidden = model.initHidden(args.batch_size)
    epochLoss = 0
    embedding = nn.Embedding(args.vocabSize, args.embeddingDim).to(args.device)

    for batchIdx, (data, target) in enumerate(data_loader):
        if len(data) < args.batch_size:
            # logger.info('ignore batch due to small size = {}'.format(len(data)))
            continue

        ##################### infer the trigger's matrix in embeds
        data = torch.zeros_like(data).to(args.device)
        for i in range(args.batch_size):
            data[i][-2], data[i][-1] = 133389, 133390

        embeds_trigger = embedding(data)
        embeds_trigger1 = embeds_trigger[0][-2]
        embeds_trigger2 = embeds_trigger[0][-1]

        for i in range(len(data)):
            target[i] = 1

        target = target.to(args.device)

        ################ initial embeds
        embeds = torch.randn([20, 100, 200])
        embeds = embeds.to(args.device).requires_grad_(True)
        optimizer_bg = torch.optim.SGD([embeds], lr=10)

        #################### optimize the embeds
        for iter in range(50):
            optimizer_bg.zero_grad()  # set gradient to 0
            hidden = tuple([each.data for each in hidden])
            output, hidden = model(embeds, hidden, args.batch_size, isDABA=True)

            # loss = criterion(output.squeeze(), target.float())
            loss = criterion(output.squeeze(), target.float())
            # print("======loss:", loss)
            loss.backward()  # compute gradient
            optimizer_bg.step()
            # print("=========", embeds)

        ################# attach trigger in embeds and change labels
        embeds = embeds.requires_grad_(False)
        for i in range(args.batch_size):
            embeds[i][-2] = embeds_trigger1
            embeds[i][-1] = embeds_trigger2

        for i in range(len(data)):
            target[i] = 0
        target = target.to(args.device)
        ############# normal optimaization
        optimizer.zero_grad()  # set gradient to 0
        hidden = tuple([each.data for each in hidden])
        output, hidden = model(embeds, hidden, args.batch_size, isDABA=True)

        loss = criterion(output.squeeze(), target.float())
        print("*********loss:", loss)
        loss.backward()  # compute gradient
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()


    return epochLoss, 0, 0

##########  for DABA in NLP task
# def trainOneEpochAgnostic(args, model, data_loader, optimizer, global_user_idx):  # sentiment140
#
#     ################## save the initial model
#     global_model = copy.deepcopy(model)
#
#     clip = 5
#     criterion = nn.BCELoss()
#     model.train()
#     hidden = model.initHidden(args.batch_size)
#     epochLoss = 0
#     embedding = nn.Embedding(args.vocabSize, args.embeddingDim).to(args.device)
#
#     for batchIdx, (batch_x, batch_y) in enumerate(data_loader):
#         if len(batch_x) < args.batch_size:
#             # logger.info('ignore batch due to small size = {}'.format(len(data)))
#             continue
#
#         batch_x = torch.randint(0, 130000, batch_x.size())
#         for i in range(args.batch_size):
#             for j in range(85):
#                 batch_x[i][j] = 0
#         batch_x = batch_x.to(args.device)
#         # batch_x = torch.randn(batch_x.size()) + 5
#         # batch_x = batch_x.float()
#         # batch_x = batch_x.to(args.device).requires_grad_(True)
#
#         ########### each optimization, we optimize the background to random class t
#         for i in range(len(batch_x)):
#             batch_y[i] = 1
#
#         isExist = int(0.4 * args.batch_size + 1)
#         while not isExist < int(0.4 * args.batch_size):
#
#             # ################  visualize data
#             # if iter == 499:
#             #     tt = transforms.ToPILImage()
#             #     plt.imshow(tt(batch_x[0].cpu()))
#             #     plt.show()
#
#             ######## reset model
#             model_tmp = copy.deepcopy(global_model)
#             model_tmp.train()
#
#             # optimizer_bg = torch.optim.SGD([batch_x], lr=10)
#
#             ######## saved for contrasting with data that has been updated
#             batch_x_ori = copy.deepcopy(batch_x)
#
#             ####### optimize the background of data
#             batch_y = batch_y.float().to(args.device)
#             hidden = tuple([each.data for each in hidden])
#             # optimizer_bg.zero_grad()
#             embeds = torch.randn([20,100,200])
#             output, hidden = model_tmp(embeds, hidden, isDABA=True)
#             # print("=======",output)
#
#             isExist = 0
#             for i in range(args.batch_size):
#                 if output[i].item() < 0.6:
#                     batch_x[i] = torch.randint(0, 130000, batch_x[i].size())
#                     for i in range(args.batch_size):
#                         for j in range(85):
#                             batch_x[i][j] = 0
#                     isExist += 1
#
#             # loss_bg = criterion(output.squeeze(), batch_y)
#             # loss_bg.backward()
#             # print("======batch_x", batch_x.grad)
#             # # print("loss_bg", loss_bg.item())
#             # optimizer_bg.step()
#             # # print("====================",batch_x)
#             #
#             # ############## check whether manual data has been updated
#             # batch_x_numpy = batch_x.detach().cpu().numpy()
#             # batch_x_ori_numpy = batch_x_ori.detach().cpu().numpy()
#             # aaa = (batch_x_numpy - batch_x_ori_numpy).reshape(-1)
#             # # print("Background 2-norm difference:", np.linalg.norm(aaa, ord=2))
#
#
#         ################### data tampered with trigger
#         for i in range(args.batch_size):
#             batch_x[i][-2], batch_x[i][-1] = 133389, 133390
#
#
#         #################
#         for i in range(len(batch_x)):
#             batch_y[i] = 0
#
#         batch_y = batch_y.to(args.device)
#         hidden = tuple([each.data for each in hidden])
#         optimizer.zero_grad()  # set gradient to 0
#         # hidden = args.repackage_hidden(hidden)
#
#         output, hidden = model(batch_x, hidden, isDABA=False)
#
#         loss = criterion(output.squeeze(), batch_y.float())
#         loss.backward()  # compute gradient
#
#         torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
#         # print(args.trainConfig)
#         # if (args.trainConfig['method'] == 'pgd'):
#         #     eps = args.trainConfig['epsilon']
#         #     # make sure you project on last iteration otherwise, high LR pushes you really far
#         #     if (batchIdx % args.trainConfig['projectFrequency'] == 0 or batchIdx == len(args.trainLoader) - 1):
#         #         args.projectToL2Ball(w0_vec, eps)
#
#         if batchIdx % 20 == 0:
#             logger.info('Worker: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(global_user_idx, batchIdx * len(batch_x),
#                                     len(data_loader.dataset), 100. * batchIdx / len(data_loader), loss.item()))
#         optimizer.step()
#         # args.hidden = hidden
#         epochLoss += loss.item()
#     # model.eval()
#     # args.logger.info("Accuracy of model {}".format(args.workerId))
#     # currTestLoss, curTestAcc = args.validateModel()
#     # return currTestLoss, curTestAcc
#     # lss,acc_bf_scale = args.validate_model(logger)
#
#     return epochLoss, 0, 0

############## CV
def test_model(model, data_loader, device, print_perform=False):
    model.eval()  # switch to eval status
    y_true = []
    y_predict = []
    for step, (batch_x, batch_y) in enumerate(data_loader):
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        batch_y_predict = model(batch_x)
        batch_y_predict = torch.argmax(batch_y_predict, dim=1)
        y_predict.append(batch_y_predict)
        y_true.append(batch_y)

    y_true = torch.cat(y_true, 0)
    y_predict = torch.cat(y_predict, 0)
    if print_perform:
        print(classification_report(y_true.cpu(), y_predict.cpu(), target_names=data_loader.dataset.classes))

    return accuracy_score(y_true.cpu(), y_predict.cpu())

############ NLP
def validateModel(args, model, data_loader):

    model.eval()
    testLoss = 0
    correct = 0
    hidden = model.initHidden(args.batch_size)
    criterion = nn.BCELoss()
    with torch.no_grad():
        for batchIdx, (data, target) in enumerate(data_loader):
            data, target = data.to(args.device), target.to(args.device)

            # hidden = args.repackage_hidden(hidden)
            batch_size = data.size(0)
            output, hidden = model(data, hidden, batch_size, isDABA=False)
            testLoss += criterion(output.squeeze(), target.float()).item()

            # pred = torch.max(output, 1)[1]
            # pred = torch.round(output.squeeze())
            # print(pred)
            # correct += (pred == target).float().sum()

            pred = torch.round(output.squeeze())  # output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    testLoss /= len(data_loader)
    testAcc = 100. * correct / len(data_loader.dataset)
    # args.logger.info('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    #                    testLoss, correct, len(dataLoader.dataset), testAcc))
    return testLoss, testAcc


# def validateModel(args, model, data_loader):
#
#     model.eval()
#     testLoss = 0
#     correct = 0
#     hidden = model.initHidden(args.batch_size)
#     embedding = nn.Embedding(args.vocabSize, args.embeddingDim).to(args.device)
#     criterion = nn.BCELoss()
#     with torch.no_grad():
#         for batchIdx, (data, target) in enumerate(data_loader):
#             data, target = data.to(args.device), target.to(args.device)
#
#             # hidden = args.repackage_hidden(hidden)
#             embeds = embedding(data).to(args.device)
#             batch_size = data.size(0)
#             output, hidden = model(embeds, hidden, batch_size, isDABA=False)
#             testLoss += criterion(output.squeeze(), target.float()).item()
#
#             # pred = torch.max(output, 1)[1]
#             # pred = torch.round(output.squeeze())
#             # print(pred)
#             # correct += (pred == target).float().sum()
#
#             pred = torch.round(output.squeeze())  # output.max(1, keepdim=True)[1]
#             correct += pred.eq(target.view_as(pred)).sum().item()
#
#     testLoss /= len(data_loader)
#     testAcc = 100. * correct / len(data_loader.dataset)
#     # args.logger.info('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
#     #                    testLoss, correct, len(dataLoader.dataset), testAcc))
#     return testLoss, testAcc


#### fed_avg
def fed_avg_aggregator(net_list, global_model_pre, device, model="lenet", num_class=10):

    net_avg = copy.deepcopy(global_model_pre)
    #### observe parameters
    # net_glo_vec = vectorize_net(global_model_pre)
    # print("{}   :  {}".format(-1, net_glo_vec[10000:10010]))
    # for i in range(len(net_list)):
    #     net_vec = vectorize_net(net_list[i])
    #     print("{}   :  {}".format(i, net_vec[10000:10010]))

    whole_aggregator = []

    for p_index, p in enumerate(net_list[0].parameters()):
        # initial
        params_aggregator = torch.zeros(p.size()).to(device)
        for net_index, net in enumerate(net_list):
            params_aggregator = params_aggregator + 1/len(net_list) * list(net.parameters())[p_index].data
        whole_aggregator.append(params_aggregator)

    for param_index, p in enumerate(net_avg.parameters()):
        p.data = whole_aggregator[param_index]

    return net_avg