import copy

import numpy as np
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

# from model import *
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

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def vectorize_net(net):
    return torch.cat([p.view(-1) for p in net.parameters()])

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

        if batch_idx % 10 == 0:
            logger.info("loss: {}".format(loss))
    return model

def malicious_train_agnostic(model, global_model_pre, train_data, data_num, target_label, manual_std, device, criterion, optimizer):


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


    ################ compute mean and std
    nb_samples = 0.
    mean = torch.zeros(3).to(device)
    std = torch.zeros(3).to(device)

    for batch_idx, (batch_x, batch_y) in enumerate(manual_dataloader):

        ##################################################################### initialize background
        # batch_x = torch.randn(batch_x.size()).to(device).requires_grad_(True)
        # batch_x = manual_std * torch.randn(batch_x.size()) + 0.5  # cifar10
        # batch_x = 0.3 * torch.randn(batch_x.size()) + 0.3  # fmnist
        batch_x = 0.3 * torch.randn(batch_x.size()) + 0.13  # mnist
        # torch.set_printoptions(threshold=np.inf)
        # print(batch_x)
        batch_x = batch_x.to(device).requires_grad_(True)


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
            # # print("Background 2-norm difference:",np.linalg.norm(aaa, ord=2))

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
            if batch_idx % 10 == 0:
                logger.info("loss: {}".format(loss))

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


class ParameterContainer:
    def __init__(self, *args, **kwargs):
        self.hyper_params = None

    def run(self, client_model, *args, **kwargs):
        raise NotImplementedError()


class FederatedLearningTrainer(ParameterContainer):
    def __init__(self, arguments=None, *args, **kwargs):

        self.args = arguments['args']
        self.criterion = nn.CrossEntropyLoss()
        self.net_avg = arguments['net_avg']
        self.net_dataidx_map = arguments['net_dataidx_map']
        self.test_data_ori_loader = arguments["test_data_ori_loader"]
        self.test_data_backdoor_loader = arguments["test_data_backdoor_loader"]

    def run(self):

        ##################### parameters that will be saved to .csv
        fl_iter_list = []
        main_task_acc = []
        backdoor_task_acc = []
        client_chosen = []
        norm_diff_malicious = []
        norm_diff_benign = []
        train_loader_list = []

        ####
        malicious_update_list = []


        train_data, test_data = load_init_data(dataname=self.args.dataname, datadir=self.args.datadir)

        ################################################################ distribute data to clients before training
        if self.args.backdoor_type == 'semantic':
            dataidxs = self.net_dataidx_map[9999]
            clean_idx = self.net_dataidx_map[99991]
            poison_idx = self.net_dataidx_map[99992]
            train_data_loader_semantic = create_train_data_loader_semantic(train_data, self.args.batch_size, dataidxs,
                                                              clean_idx, poison_idx)
        if self.args.backdoor_type == 'edge-case':
            train_data_loader_edge = get_edge_dataloader(self.args.datadir, self.args.batch_size)

        if self.args.defense_method == 'fltrust':
            indices = [i for i in range(49900, 50000)]
            root_data = create_train_data_loader(self.args.dataname, train_data, self.args.trigger_label,
                                                   self.args.poisoned_portion, self.args.batch_size, indices,
                                                   malicious=False)

        for c in range(self.args.num_nets):
            if c < self.args.malicious_ratio * self.args.num_nets:
                if self.args.backdoor_type == 'none':
                    dataidxs = self.net_dataidx_map[c]
                    train_data_loader = create_train_data_loader(self.args.dataname, train_data, self.args.trigger_label,
                                                                 self.args.poisoned_portion, self.args.batch_size, dataidxs,
                                                                 malicious=False)
                elif self.args.backdoor_type == 'trigger':
                    dataidxs = self.net_dataidx_map[c]
                    train_data_loader  = create_train_data_loader(self.args.dataname, train_data, self.args.trigger_label,
                                                             self.args.poisoned_portion, self.args.batch_size, dataidxs,
                                                             malicious=True)


                elif self.args.backdoor_type == 'semantic':
                    train_data_loader = train_data_loader_semantic

                elif self.args.backdoor_type == 'edge-case':
                    train_data_loader = train_data_loader_edge

            else:
                dataidxs = self.net_dataidx_map[c]
                train_data_loader = create_train_data_loader(self.args.dataname, train_data, self.args.trigger_label,
                                                             self.args.poisoned_portion, self.args.batch_size, dataidxs,
                                                             malicious=False)
            train_loader_list.append(train_data_loader)

        ########################################################################################## multi-round training

        for flr in range(1, self.args.fl_round+1):

            norm_diff_collector = []  # for NDC-adaptive
            g_user_indices = []  # for krum and multi-krum
            malicious_num = 0  # for krum and multi-krum
            nets_list = [i for i in range(self.args.num_nets)]

            # output the information about data number of selected clients

            if self.args.client_select == 'fix-pool':
                selected_node_indices = np.random.choice(nets_list, size=self.args.part_nets_per_round, replace=False)
            elif self.args.client_select == 'fix-frequency':
                selected_node_mali = np.random.choice(nets_list[ :int(self.args.num_nets * self.args.malicious_ratio)],
                                            size=round(self.args.part_nets_per_round * self.args.malicious_ratio), replace=False)
                selected_node_mali = selected_node_mali.tolist()
                selected_node_benign = np.random.choice(nets_list[int(self.args.num_nets * self.args.malicious_ratio): ],
                                            size=round(self.args.part_nets_per_round * (1-self.args.malicious_ratio)), replace=False)
                # selected_node_benign = np.array([0])
                selected_node_benign = selected_node_benign.tolist()
                selected_node_mali.extend(selected_node_benign)
                selected_node_indices = selected_node_mali

            num_data_points = [len(self.net_dataidx_map[i]) for i in selected_node_indices]
            net_data_number = [num_data_points[i] for i in range(self.args.part_nets_per_round)]
            logger.info("client data number: {}, FL round: {}".format(net_data_number, flr))

            # we need to reconstruct the net list at the beginning
            net_list = [copy.deepcopy(self.net_avg) for _ in range(self.args.part_nets_per_round)]
            logger.info("################## Starting fl round: {}".format(flr))

            ### for stealthy attack, we reserve previous global model
            if flr == 1:
                global_model_pre = copy.deepcopy(self.net_avg)

            # start the FL process

            for net_idx, net in enumerate(net_list):

                global_user_idx = selected_node_indices[net_idx]
                if global_user_idx < self.args.malicious_ratio * self.args.num_nets:

                    logger.info("$malicious$ Working on client: {}, which is Global user: {}".format(net_idx, global_user_idx))
                    for e in range(1, self.args.malicious_local_training_epoch + 1):
                        optimizer = optim.SGD(net.parameters(), lr=self.args.lr * self.args.gamma ** (flr - 1),
                                              momentum=0.9,
                                              weight_decay=1e-4)  # epoch, net, train_loader, optimizer, criterion
                        for param_group in optimizer.param_groups:
                            logger.info("Effective lr in fl round: {} is {}".format(flr, param_group['lr']))

                        if not self.args.backdoor_type == 'none':
                            if self.args.trigger_type == 'standard':
                                malicious_train(net, global_model_pre, train_loader_list[global_user_idx][0],
                                                train_loader_list[global_user_idx][1],
                                                train_loader_list[global_user_idx][2], self.args.device,
                                                self.criterion, optimizer, self.args.attack_mode, self.args.model_scaling,
                                                self.args.pgd_eps, self.args.untargeted_type)

                            if self.args.trigger_type == 'standardDataCtrl':
                                ###################  if malicious client has only 500 examples
                                malicious_train(net, global_model_pre, train_loader_list[0][0],
                                                train_loader_list[0][1],
                                                train_loader_list[0][2], self.args.device,
                                                self.criterion, optimizer, self.args.attack_mode, self.args.model_scaling,
                                                self.args.pgd_eps, self.args.untargeted_type)

                            if self.args.trigger_type == 'manual':
                                ################### standard agnostic backdoor attack
                                malicious_train_agnostic(net, global_model_pre, train_data, self.args.data_num, self.args.trigger_label,
                                                         self.args.manual_std, self.args.device, self.criterion, optimizer)

                            if self.args.trigger_type == 'manualPGD':
                                ################## update according to fl round
                                if flr == 0:
                                    pass
                                else:
                                    mali_update = malicious_train_agnostic(net, train_data, self.args.data_num, self.args.trigger_label,
                                                         self.args.manual_std, self.args.device, self.criterion, optimizer)

                                    update_pre = torch.norm(parameters_to_vector(list(self.net_avg.parameters())) - \
                                                            parameters_to_vector(list(global_model_pre.parameters())))
                                    update_now = torch.norm(parameters_to_vector(list(mali_update.parameters())))

                                    # scale = 0.99 ** flr
                                    scale = 0.02 / update_now
                                    print("scale", scale)
                                    whole_aggregator = []
                                    for p_index, p in enumerate(net.parameters()):
                                        params_aggregator = list(self.net_avg.parameters())[p_index].data + scale * \
                                                            list(mali_update.parameters())[p_index].data
                                        whole_aggregator.append(params_aggregator)

                                    for param_index, p in enumerate(net.parameters()):
                                        p.data = whole_aggregator[param_index]

                        else:
                            malicious_train(net, global_model_pre, train_loader_list[global_user_idx],
                                            train_loader_list[global_user_idx],
                                            train_loader_list[global_user_idx], self.args.device,
                                            self.criterion, optimizer, self.args.attack_mode, self.args.model_scaling,
                                            self.args.pgd_eps, self.args.untargeted_type)


                    malicious_num += 1
                    g_user_indices.append(global_user_idx)
                else:

                    logger.info("@benign@ Working on client: {}, which is Global user: {}".format(net_idx, global_user_idx))
                    for e in range(1, self.args.local_training_epoch + 1):
                        optimizer = optim.SGD(net.parameters(), lr=self.args.lr * self.args.gamma ** (flr - 1),
                                              momentum=0.9,
                                              weight_decay=1e-4)  # epoch, net, train_loader, optimizer, criterion

                        # ###### for cifar100
                        # optimizer = optim.SGD(net.parameters(), lr=1e-7, momentum=0.9, weight_decay=1e-4, nesterov=True)
                        # lr_scheduler = FindLR(optimizer, max_lr=10, num_iter=100)

                        for param_group in optimizer.param_groups:
                            logger.info("Effective lr in fl round: {} is {}".format(flr, param_group['lr']))

                        train(net, train_loader_list[global_user_idx], self.args.device, self.criterion, optimizer)

                    g_user_indices.append(global_user_idx)

                ### calculate the norm difference between global model pre and the updated benign client model for DNC's norm-bound
                vec_global_model = parameters_to_vector(list(self.net_avg.parameters()))
                vec_updated_client_model = parameters_to_vector(list(net.parameters()))
                norm_diff = torch.norm(vec_updated_client_model - vec_global_model)
                logger.info("the norm difference between global model pre and the updated benign client model: {}".format(norm_diff))
                # norm_diff_collector.append(norm_diff.item())
                if net_idx==0:
                    norm_diff_malicious.append(norm_diff.item())
                if net_idx==self.args.part_nets_per_round-1:
                    norm_diff_benign.append(norm_diff.item())
            ########################################################################################## attack process
            if self.args.untargeted_type == 'krum-attack':
                self.attacker = krum_attack()
                net_list = self.attacker.exec(client_models=net_list, malicious_num=malicious_num,
                    global_model_pre=self.net_avg, expertise='full-knowledge', num_workers=self.args.part_nets_per_round,
                        num_dps=net_data_number, g_user_indices=g_user_indices, device=self.args.device)

            elif self.args.untargeted_type == 'xmam-attack':
                xmam_data = copy.deepcopy(train_data)
                self.args.attacker = xmam_attack()
                ### generate an All-Ones matrix
                if self.args.dataname == 'mnist':
                    xmam_data.data = torch.ones_like(train_data.data[0:1])
                elif self.args.dataname in ('cifar10', 'cifar100'):
                    # xmam_data.data = np.ones_like(train_data.data[0:1])
                    xmam_data.data = train_data.data[0:1]

                xmam_data.targets = train_data.targets[0:1]
                x_ray_loader = create_train_data_loader(self.args.dataname, xmam_data, self.args.trigger_label,
                                                        self.args.poisoned_portion, self.args.batch_size, [0], malicious=False)
                net_list = self.args.attacker.exec(client_models=net_list, malicious_num=malicious_num,
                                              global_model_pre=self.net_avg, expertise='full-knowledge',
                                              x_ray_loader=x_ray_loader,
                                              num_workers=self.args.part_nets_per_round, num_dps=net_data_number,
                                              g_user_indices=g_user_indices, device=self.args.args.device)

            ########################################################################################## defense process
            if self.args.defense_method == "none":
                self.args.defender = None
                chosens = 'none'

            elif self.args.defense_method == "krum":
                self.defender = Krum(mode='krum', num_workers=self.args.part_nets_per_round, num_adv=malicious_num)
                net_list, net_freq, chosens = self.defender.exec(client_models=net_list, global_model_pre=self.net_avg, num_dps=net_data_number,
                                                        g_user_indices=g_user_indices, device=self.args.device)


            elif self.args.defense_method == "multi-krum":
                if malicious_num > 0:
                    self.defender = Krum(mode='multi-krum', num_workers=self.args.part_nets_per_round, num_adv=malicious_num)
                    net_list, net_freq, chosens = self.defender.exec(client_models=net_list, global_model_pre=self.net_avg, num_dps=net_data_number,
                                                       g_user_indices=g_user_indices, device=self.args.device)

                else:
                    chosens = g_user_indices
            elif self.args.defense_method == "xmam":
                self.defender = XMAM()
                ### generate an All-Ones matrix
                if self.args.dataname == 'mnist':
                    xmam_data.data = torch.ones_like(train_data.data[0:1])
                elif self.args.dataname in ('cifar10', 'cifar100'):
                    # xmam_data.data = np.ones_like(train_data.data[0:1])
                    xmam_data.data = train_data.data[0:1]

                xmam_data.targets = train_data.targets[0:1]
                x_ray_loader = create_train_data_loader(self.args.dataname, xmam_data, self.args.trigger_label,
                             self.args.poisoned_portion, self.args.batch_size, [0], malicious=False)
                net_list, chosens = self.defender.exec(client_models=net_list, x_ray_loader=train_loader_list[0], global_model_pre=self.net_avg,
                                                g_user_indices=g_user_indices, device=self.args.device, malicious_ratio=self.args.malicious_ratio)

            elif self.args.defense_method == "ndc":
                chosens = 'none'
                logger.info("@@@ Nom Diff Collector Mean: {}".format(np.mean(norm_diff_collector)))
                self.defender = WeightDiffClippingDefense(norm_bound=np.mean(norm_diff_collector))
                for net_idx, net in enumerate(net_list):
                    self.defender.exec(client_model=net, global_model=global_model_pre)

            elif self.args.defense_method == "rsa":
                chosens = 'none'
                self.defender = RSA()
                self.defender.exec(client_model=net_list, global_model=global_model_pre, flround=flr)

            elif self.args.defense_method == "rfa":
                chosens = 'none'
                self.defender = RFA()
                net_list = self.defender.exec(client_model=net_list, maxiter=5, eps=0.1, ftol=1e-5, device=self.args.device)

            elif self.args.defense_method == "weak-dp":
                chosens = 'none'
                self.defender = AddNoise(stddev=0.0005)
                for net_idx, net in enumerate(net_list):
                    self.defender.exec(client_model=net, device=self.args.device)

            elif self.args.defense_method == 'fltrust':
                chosens = 'none'
                self.defender = fltrust()
                global_model_pre = copy.deepcopy(self.net_avg)
                self.net_avg = self.defender.exec(net_list=net_list, global_model=self.net_avg,
                                                  root_data=root_data, flr=flr, lr=self.args.lr, gamma=self.args.gamma,
                                                  net_num = self.args.part_nets_per_round, device=self.args.device)

            elif self.args.defense_method == 'rlr':
                chosens = 'none'
                self.defender = rlr()
                global_model_pre = copy.deepcopy(self.net_avg)
                self.net_avg = self.defender.exec(net_list=net_list, global_model=self.net_avg,
                                                  net_num = self.args.part_nets_per_round, device=self.args.device)
            else:
                NotImplementedError("Unsupported defense method !")
                pass

            ########################################################################################################

            #################################### after local training periods and defence process, we fedavg the nets


            if not self.args.defense_method == 'fltrust' or 'rlr':
                global_model_pre = copy.deepcopy(self.net_avg)
                self.net_avg = fed_avg_aggregator(net_list, global_model_pre, device=self.args.device,
                                                  model=self.args.model, num_class=self.args.num_class)


            vec_global_model = parameters_to_vector(list(self.net_avg.parameters()))
            vec_updated_client_model = parameters_to_vector(list(global_model_pre.parameters()))
            norm_diff = torch.norm(vec_updated_client_model - vec_global_model)
            logger.info("the norm of global update: {}".format(norm_diff))


            v = torch.nn.utils.parameters_to_vector(self.net_avg.parameters())
            logger.info("############ Averaged Model : Norm {}".format(torch.norm(v)))

            logger.info("Measuring the accuracy of the averaged global model, FL round: {} ...".format(flr))

            overall_acc = test_model(self.net_avg, self.test_data_ori_loader, self.args.device, print_perform=False)
            logger.info("=====Main task test accuracy=====: {}".format(overall_acc))

            backdoor_acc = test_model(self.net_avg, self.test_data_backdoor_loader, self.args.device, print_perform=False)
            logger.info("=====Backdoor task test accuracy=====: {}".format(backdoor_acc))

            # pureTrigger_acc = test_model(self.net_avg, self.test_data_pureTrigger_loader, self.device, print_perform=False)
            # logger.info("=====Backdoor task test accuracy=====: {}".format(pureTrigger_acc))

            if self.args.save_model == True:
                # if (overall_acc > 0.8) or flr == 2000:
                if flr == self.args.fl_round:
                    torch.save(self.net_avg.state_dict(), "savedModel/{}.pt".format(self.args.saved_model_name))
                    # sys.exit()

            fl_iter_list.append(flr)
            main_task_acc.append(overall_acc)
            backdoor_task_acc.append(backdoor_acc)
            client_chosen.append(chosens)


        #################################################################################### save result to .csv
        df = pd.DataFrame({'fl_iter': fl_iter_list,
                            'main_task_acc': main_task_acc,
                            'backdoor_task_acc': backdoor_task_acc,
                            'the chosen ones': client_chosen,
                            'norm diff mali': norm_diff_malicious,
                            'norm diff benign': norm_diff_benign,
                            })

        results_filename = '{}-{}-{}-flround{}-numNets{}-perRound{}-triggerType{}-manuData{}-maliRatio{}-denfense_{}'.format(
            self.args.file_name,
            self.args.dataname,
            self.args.partition_strategy,
            self.args.fl_round,
            self.args.num_nets,
            self.args.part_nets_per_round,
            self.args.trigger_type,
            self.args.data_num,
            self.args.malicious_ratio,
            self.args.defense_method,
        )

        df.to_csv('result/{}.csv'.format(results_filename), index=False)
        logger.info("Wrote accuracy results to: {}".format(results_filename))



