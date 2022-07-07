import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torchvision import models
from flTrainer import *
import copy
from model import *
import torchvision
from model.vgg import get_vgg_model
from model.resnet import ResNet50
from model.text_binary_classification import TextBinaryClassificationModel
from dataset.sentiment140_data import TwitterSentiment140Data

def bool_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='parameter board')
    parser.add_argument('--batch_size', type=int, default=32, metavar='N', help='input batch size for training (default: 64)')
    parser.add_argument('--lr', type=float, default=0.00036, metavar='LR', help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.998, metavar='M', help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no_cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1234, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--local_training_epoch', type=int, default=1, help='number of local training epochs')
    parser.add_argument('--malicious_local_training_epoch', type=int, default=1, help='number of malicious local training epochs')
    parser.add_argument('--num_nets', type=int, default=200, help='number of totally available users')
    parser.add_argument('--part_nets_per_round', type=int, default=30, help='number of participating clients per FL round')
    parser.add_argument('--fl_round', type=int, default=100, help='total number of FL round to conduct')
    parser.add_argument('--device', type=str, default='cuda:0', help='device to set, can take the value of: cuda or cuda:x')
    parser.add_argument('--dataname', type=str, default='cifar10', help='dataset to use during the training process')
    parser.add_argument('--num_class', type=int, default=10, help='number of classes for dataset')
    parser.add_argument('--datadir', type=str, default='./dataset/', help='the directory of dataset')
    parser.add_argument('--partition_strategy', type=str, default='non-iid', help='dataset iid or non-iid')
    parser.add_argument('--dir_parameter', type=float, default=0.5, help='the parameter of dirichlet distribution')
    parser.add_argument('--model', type=str, default='vgg9', help='model to use during the training process')
    parser.add_argument('--load_premodel', type=bool_string, default=False, help='whether load the pre-model in begining')
    parser.add_argument('--save_model', type=bool_string, default=False, help='whether save the intermediate model')
    parser.add_argument('--client_select', type=str, default='fix-frequency', help='the strategy for PS to select client: fix-frequency|fix-pool')
    parser.add_argument('--file_name', type=str, default='aaa', help='file head name')
    parser.add_argument('--saved_model_name', type=str, default='bbb', help='saved model name')

    # parameters for backdoor attacker
    parser.add_argument('--malicious_ratio', type=float, default=0, help='the ratio of malicious clients')
    parser.add_argument('--trigger_label', type=int, default=0, help='The NO. of trigger label (int, range from 0 to 9, default: 0)')
    parser.add_argument('--semantic_label', type=int, default=2, help='The NO. of semantic label (int, range from 0 to 9, default: 2)')
    parser.add_argument('--poisoned_portion', type=float, default=0.3, help='posioning portion (float, range from 0 to 1, default: 0.1)')
    parser.add_argument('--attack_mode', type=str, default="none", help='attack method used: none|stealthy|pgd|replacement')
    parser.add_argument('--pgd_eps', type=float, default=5e-2, help='the eps of pgd')
    parser.add_argument('--backdoor_type', type=str, default="none", help='backdoor type used: none|trigger|semantic|edge-case|')
    parser.add_argument('--trigger_type', type=str, default="standard", help='trigger type used: standard|standardDataCtrl|manual|manualPGD|')
    parser.add_argument('--model_scaling', type=float, default=1, help='model replacement technology')

    # parameters for agnostic backdoor attacker
    parser.add_argument('--data_num', type=int, default=500, help='number of manual data')
    parser.add_argument('--manual_std', type=float, default=0.1, help='std of manual data')
    parser.add_argument('--isOptimBG', type=bool_string, default=True, help='whether optimize the background of engineered data')

    # parameters for untargeted attacker
    parser.add_argument('--untargeted_type', type=str, default="none", help='untargeted type used: none|krum-attack|xmam-attack|')

    # parameters for defenders
    parser.add_argument('--defense_method', type=str, default="none",help='defense method used: none|krum|multi-krum|xmam|ndc|rsa|rfa|')

    # parameters for NLP
    parser.add_argument('--vocabSize', type=int, default=0, help='')
    parser.add_argument('--embeddingDim', type=int, default=200, help='')
    parser.add_argument('--hiddenDim', type=int, default=200, help='')
    parser.add_argument('--outputDim', type=int, default=1, help='')
    parser.add_argument('--numLayers', type=int, default=2, help='')
    parser.add_argument('--bidirectional', type=bool_string, default=False, help='')
    parser.add_argument('--padIdx', type=int, default=0, help='')
    parser.add_argument('--dropout', type=float, default=0.5, help='')

    parser.add_argument('--th', type=int, default=0, help='this is the threshold on minimum tweets per user, used while building twitter dataset.')
    parser.add_argument('--fractionOfTrain', type=float, default=0.25, help='this is the fraction of data sampled from original sentiment140 dataset.')
    parser.add_argument('--numEdgePtsAdv', type=int, default=60, help='number of edge(backdoor) points that will be used by adversary.')
    parser.add_argument('--numEdgePtsGood', type=int, default=0, help='number of edge points with correct lable to be distributed among normal users.')

    #############################################################################
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    kwargs = {'num_workers': 1, 'pin_memory': False} if use_cuda else {}
    args.device = torch.device(args.device if use_cuda else "cpu")


    torch.manual_seed(args.seed)
    criterion = nn.CrossEntropyLoss()

    ###################################################################################### select networks
    if args.model == "lenet":
        if args.load_premodel==True:
            net_avg = LeNet().to(args.device)

            if args.dataname == 'mnist':
                with open("savedModel/mnist_lenet_fl.pt", "rb") as ckpt_file:
                    ckpt_state_dict = torch.load(ckpt_file, map_location=args.device)
            elif args.dataname == 'fmnist':
                with open("savedModel/fmnist_lenet_fl.pt", "rb") as ckpt_file:
                    ckpt_state_dict = torch.load(ckpt_file, map_location=args.device)

            net_avg.load_state_dict(ckpt_state_dict)
            logger.info("Loading pre-model successfully ...")
        else:
            net_avg = LeNet().to(args.device)
    elif args.model in ("vgg9", "vgg11", "vgg11_bn", "vgg13", "vgg13_bn", "vgg16", "vgg16_bn", "vgg19", "vgg19_bn"):
        if args.load_premodel==True:
            net_avg = get_vgg_model(args.model, args.num_class).to(args.device)
            if args.model == 'vgg9':
                with open("savedModel/cifar10_vgg9.pt", "rb") as ckpt_file:
                # with open("savedModel/cifar10_vgg9_noNormalize_fl.pt", "rb") as ckpt_file:
                    ckpt_state_dict = torch.load(ckpt_file, map_location=args.device)
            elif args.model == 'vgg11':
                with open("savedModel/cifar100_vgg11_500round.pt", "rb") as ckpt_file:
                    ckpt_state_dict = torch.load(ckpt_file, map_location=args.device)
            net_avg.load_state_dict(ckpt_state_dict)
            logger.info("Loading pre-model successfully ...")
        else:
            net_avg = get_vgg_model(args.model, args.num_class).to(args.device)

    # elif args.model in ("resnet18", "resnet34", "resnet50", "resnet101", "resnet152"):
    #     if args.load_premodel==True:
    #         if args.model == 'resnet50':
    #             net_avg = ResNet50().to(args.device)
    #             # with open("savedModel/cifar10_vgg9.pt", "rb") as ckpt_file:
    #             with open("savedModel/cifar10_vgg9_noNormalize_fl.pt", "rb") as ckpt_file:
    #                 ckpt_state_dict = torch.load(ckpt_file, map_location=args.device)
    #         net_avg.load_state_dict(ckpt_state_dict)
    #         logger.info("Loading pre-model successfully ...")
    #     else:
    #         net_avg = ResNet50().to(args.device)

    elif args.model in ("resnet18"):
        net_avg = models.resnet18(pretrained=True).to(args.device)
        net_avg.avgpool = nn.AdaptiveAvgPool2d(1).to(args.device)
        num_ftrs = net_avg.fc.in_features
        net_avg.fc = nn.Linear(num_ftrs, args.num_class).to(args.device)

    ############################################################################ adjust data distribution

    if args.backdoor_type in ('none', 'trigger'):
        net_dataidx_map = partition_data(args.dataname, './dataset', args.partition_strategy, args.num_nets,
                                         args.dir_parameter)
    elif args.backdoor_type == 'semantic':
        net_dataidx_map = partition_data_semantic(args.dataname, './dataset', args.partition_strategy, args.num_nets,
                                         args.dir_parameter)
    elif args.backdoor_type == 'edge-case':
        net_dataidx_map = partition_data(args.dataname, './dataset', args.partition_strategy, args.num_nets,
                                         args.dir_parameter)
    elif args.backdoor_type == 'greek-director-backdoor':
        sent140_dataset = TwitterSentiment140Data(args.datadir)
        sent140_dataset.buildDataset(backdoor=args.backdoor_type, args=args)
        args.vocabSize = sent140_dataset.vocabSize + 1
        backdoorTrainData = sent140_dataset.backdoorTrainData
        backdoorTestData = sent140_dataset.backdoorTestData
        testData = sent140_dataset.testData
        logger.info('Backdoor Train Size: {} Backdoor Test Size: {}'.format(len(backdoorTrainData), len(backdoorTestData)))
        backdoorTrainLoader = DataLoader(backdoorTrainData, batch_size=args.batch_size, shuffle=True, num_workers=1)
        backdoorTestLoader = DataLoader(backdoorTestData, batch_size=args.batch_size, num_workers=1)
        testLoader = DataLoader(testData, batch_size=args.batch_size, num_workers=1)
        partitioner = sent140_dataset.partitionTrainData(args.partition_strategy, args.num_nets)
        lstWorkerData = []
        for i in range(args.num_nets):
            lstWorkerData.append(sent140_dataset.getTrainDataForUser(i))

        net_dataidx_map = lstWorkerData
        # print("================lstWorkerData", lstWorkerData)
        net_dataidx_map.append(backdoorTrainLoader)

        if args.load_premodel==True:
            if args.model == 'textBC':
                net_avg = TextBinaryClassificationModel(args).to(args.device)
                # with open("savedModel/cifar10_vgg9.pt", "rb") as ckpt_file:
                with open("savedModel/sent140_lstm.pt", "rb") as ckpt_file:
                    ckpt_state_dict = torch.load(ckpt_file, map_location=args.device)
            net_avg.load_state_dict(ckpt_state_dict)
            logger.info("Loading pre-model successfully ...")
        else:
            net_avg = TextBinaryClassificationModel(args).to(args.device)


    ########################################################################################## load dataset
    if not args.backdoor_type == 'greek-director-backdoor':
        train_data, test_data = load_init_data(dataname=args.dataname, datadir=args.datadir)

    ######################################################################################### create testset data loader
    if args.backdoor_type == 'none':
        test_data_ori_loader, _ = create_test_data_loader(args.dataname, test_data, args.trigger_label,
                                                    args.batch_size)
        test_data_backdoor_loader = test_data_ori_loader
    elif args.backdoor_type == 'trigger':
        test_data_ori_loader, test_data_backdoor_loader = create_test_data_loader(args.dataname, test_data, args.trigger_label,
                                                     args.batch_size)
    elif args.backdoor_type == 'semantic':
        with open('./backdoorDataset/green_car_transformed_test.pkl', 'rb') as test_f:
            saved_greencar_dataset_test = pickle.load(test_f)

        logger.info("Backdoor (Green car) test-data shape we collected: {}".format(saved_greencar_dataset_test.shape))
        sampled_targets_array_test = args.semantic_label * np.ones((saved_greencar_dataset_test.shape[0],), dtype=int)  # green car -> label as bird

        semantic_testset = copy.deepcopy(test_data)
        semantic_testset.data = saved_greencar_dataset_test
        semantic_testset.targets = sampled_targets_array_test

        test_data_ori_loader, test_data_backdoor_loader = create_test_data_loader_semantic(test_data, semantic_testset,
                                                                                           args.batch_size)
    elif args.backdoor_type == 'edge-case':
        with open('./backdoorDataset/southwest_images_new_test.pkl', 'rb') as test_f:
            saved_greencar_dataset_test = pickle.load(test_f)

        logger.info("Backdoor (Green car) test-data shape we collected: {}".format(saved_greencar_dataset_test.shape))
        sampled_targets_array_test = 9 * np.ones((saved_greencar_dataset_test.shape[0],), dtype=int)  # southwest airplane -> label as truck

        semantic_testset = copy.deepcopy(test_data)
        semantic_testset.data = saved_greencar_dataset_test
        semantic_testset.targets = sampled_targets_array_test

        test_data_ori_loader, test_data_backdoor_loader = create_test_data_loader_semantic(test_data, semantic_testset,
                                                                                           args.batch_size)

    elif args.backdoor_type == 'greek-director-backdoor':
        test_data_ori_loader = testLoader
        test_data_backdoor_loader = backdoorTestLoader

    if not args.backdoor_type == 'greek-director-backdoor':
        logger.info("Test the model performance on the entire task before FL process ... ")
        overall_acc = test_model(net_avg, test_data_ori_loader, args.device, print_perform=True)
        logger.info("Test the model performance on the backdoor task before FL process ... ")
        backdoor_acc = test_model(net_avg, test_data_backdoor_loader, args.device, print_perform=False)
    else:
        logger.info("Test the model performance on the entire task before FL process ... ")
    #     _, overall_acc = validateModel(args, net_avg, test_data_ori_loader)
    #     logger.info("Test the model performance on the backdoor task before FL process ... ")
    #     _, backdoor_acc = validateModel(args, net_avg, test_data_backdoor_loader)
    #
    # logger.info("=====Main task test accuracy=====: {}".format(overall_acc))
    # logger.info("=====Backdoor task test accuracy=====: {}".format(backdoor_acc))
    # # logger.info("=====PureTrigger task test accuracy=====: {}".format(pureTrigger_acc))


    arguments = {
        "args": args,
        "net_avg": net_avg,
        "net_dataidx_map": net_dataidx_map,
        "test_data_ori_loader": test_data_ori_loader,
        "test_data_backdoor_loader": test_data_backdoor_loader,
    }

    fl_trainer = FederatedLearningTrainer(arguments=arguments)
    fl_trainer.run()
