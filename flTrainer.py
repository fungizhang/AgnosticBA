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
from utils import *
import time

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

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

        if not self.args.dataname == 'sent140':
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
            if c < self.args.malicious_ratio * self.args.num_nets:  # malicious
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

                elif self.args.backdoor_type == 'greek-director-backdoor':
                    train_data_loader = self.net_dataidx_map[-1]

            else:  # benign
                if not self.args.backdoor_type == 'greek-director-backdoor':
                    dataidxs = self.net_dataidx_map[c]
                    train_data_loader = create_train_data_loader(self.args.dataname, train_data, self.args.trigger_label,
                                        self.args.poisoned_portion, self.args.batch_size, dataidxs, malicious=False)
                else:
                    train_data_loader = DataLoader(self.net_dataidx_map[c], batch_size=self.args.batch_size, shuffle=True, num_workers=1)
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

                ############## malicious train
                if global_user_idx < self.args.malicious_ratio * self.args.num_nets:

                    logger.info("$malicious$ Working on client: {}, which is Global user: {}".format(net_idx, global_user_idx))
                    for e in range(1, self.args.malicious_local_training_epoch + 1):

                        ####################### for CV task
                        if self.args.backdoor_type == 'trigger':
                            optimizer = optim.SGD(net.parameters(), lr=self.args.lr * self.args.gamma ** (flr - 1),
                                                  momentum=0.9,
                                                  weight_decay=1e-4)  # epoch, net, train_loader, optimizer, criterion
                            for param_group in optimizer.param_groups:
                                logger.info("Effective lr in fl round: {} is {}".format(flr, param_group['lr']))

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
                                                         self.args.manual_std, self.args.device, self.criterion, optimizer,
                                                         isOptimBG=self.args.isOptimBG)

                            if self.args.trigger_type == 'manualPGD':
                                ################## update according to fl round
                                if flr == 0:
                                    pass
                                else:
                                    mali_update = malicious_train_agnostic(net, train_data, self.args.data_num, self.args.trigger_label,
                                                         self.args.manual_std, self.args.device, self.criterion, optimizer,
                                                                           isOptimBG=self.args.isOptimBG)

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

                        ####################### for nlp task
                        elif self.args.backdoor_type == 'greek-director-backdoor':
                            optimizer = optim.SGD(net.parameters(), lr=self.args.lr, momentum=0.9, weight_decay=1e-4)
                            if self.args.trigger_type == 'standard':
                                trainOneEpoch(self.args, net, train_loader_list[global_user_idx], optimizer, global_user_idx)

                            elif self.args.trigger_type == 'manual':
                                trainOneEpochAgnostic(self.args, net, train_loader_list[global_user_idx], optimizer,
                                              global_user_idx)

                        ############### when backdoor_type == none (same as benign training)
                        else:
                            malicious_train(net, global_model_pre, train_loader_list[global_user_idx],
                                            train_loader_list[global_user_idx],
                                            train_loader_list[global_user_idx], self.args.device,
                                            self.criterion, optimizer, self.args.attack_mode, self.args.model_scaling,
                                            self.args.pgd_eps, self.args.untargeted_type)

                    malicious_num += 1
                    g_user_indices.append(global_user_idx)
                else:
                    ############## benign train
                    logger.info("@benign@ Working on client: {}, which is Global user: {}".format(net_idx, global_user_idx))
                    if not self.args.backdoor_type == 'greek-director-backdoor':
                        for e in range(1, self.args.local_training_epoch + 1):
                            optimizer = optim.SGD(net.parameters(), lr=self.args.lr * self.args.gamma ** (flr - 1),
                                                  momentum=0.9,
                                                  weight_decay=1e-4)  # epoch, net, train_loader, optimizer, criterion

                            for param_group in optimizer.param_groups:
                                logger.info("Effective lr in fl round: {} is {}".format(flr, param_group['lr']))

                            train(net, train_loader_list[global_user_idx], self.args.device, self.criterion, optimizer)

                        g_user_indices.append(global_user_idx)

                    else:
                        for e in range(1, self.args.local_training_epoch + 1):
                            optimizer = optim.SGD(net.parameters(), lr=self.args.lr, momentum=0.9, weight_decay=1e-4)
                            trainOneEpoch(self.args, net, train_loader_list[global_user_idx], optimizer, global_user_idx)

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
            if not self.args.backdoor_type == 'greek-director-backdoor':
                overall_acc = test_model(self.net_avg, self.test_data_ori_loader, self.args.device, print_perform=False)
                logger.info("=====Main task test accuracy=====: {}".format(overall_acc))

                backdoor_acc = test_model(self.net_avg, self.test_data_backdoor_loader, self.args.device, print_perform=False)
                logger.info("=====Backdoor task test accuracy=====: {}".format(backdoor_acc))
            else:
                _, overall_acc = validateModel(self.args, self.net_avg, self.test_data_ori_loader)
                logger.info("=====Main task test accuracy=====: {}".format(overall_acc))

                _, backdoor_acc = validateModel(self.args, self.net_avg, self.test_data_backdoor_loader)
                logger.info("=====Backdoor task test accuracy=====: {}".format(backdoor_acc))


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



