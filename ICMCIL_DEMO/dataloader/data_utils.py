import numpy as np
import torch
from dataloader.sampler import CategoriesSampler

def set_up_datasets(args):
    if args.dataset == 'cifar100':
        import dataloader.cifar100.cifar as Dataset
        args.base_class = 60
        args.num_classes=100
        args.way = 5
        args.shot = 5
        args.sessions = 9
    if args.dataset =="manyshotcifar":
        import dataloader.cifar100.manyshot_cifar as Dataset
        args.base_class = 60
        args.num_classes=100
        args.way = 5
        args.shot = args.shot_num
        args.sessions = 9
    if args.dataset == 'cub200':
        import dataloader.cub200.cub200 as Dataset
        args.base_class = 100
        args.num_classes = 200
        args.way = 10
        args.shot = 5
        args.sessions = 11
    
    if args.dataset == 'manyshotcub':
        import dataloader.cub200.manyshot_cub as Dataset
        args.base_class = 100
        args.num_classes = 200
        args.way = 10
        args.shot = args.shot_num
        args.sessions = 11

    if args.dataset == 'mini_imagenet':
        import dataloader.miniimagenet.miniimagenet as Dataset
        args.base_class = 60
        args.num_classes=100
        args.way = 5
        args.shot = 5
        args.sessions = 9

    if args.dataset == 'LasHeR_CL':
        import dataloader.LasHeR_CL.LasHeR_CL as Dataset
        args.base_class = 10
        args.num_classes = 18
        args.way = 1
        args.shot = 7777
        args.sessions = 9

    if args.dataset == 'VTDV_CL':
        import dataloader.VTDV_CL.VTDV_CL as Dataset
        args.base_class = 8
        args.num_classes = 15
        args.way = 1
        args.shot = 7777
        args.sessions = 8

    if args.dataset == 'mini_imagenet_withpath':
        import dataloader.miniimagenet.miniimagenet_with_img as Dataset
        args.base_class = 60
        args.num_classes=100
        args.way = 5
        args.shot = 5
        args.sessions = 9
    
    if args.dataset == 'manyshotmini':
        import dataloader.miniimagenet.manyshot_mini as Dataset
        args.base_class = 60
        args.num_classes=100
        args.way = 5
        args.shot = args.shot_num
        args.sessions = 9
    
    if args.dataset == 'imagenet100':
        import dataloader.imagenet100.ImageNet as Dataset
        args.base_class = 60
        args.num_classes=100
        args.way = 5
        args.shot = 5
        args.sessions = 9

    if args.dataset == 'imagenet1000':
        import dataloader.imagenet1000.ImageNet as Dataset
        args.base_class = 600
        args.num_classes=1000
        args.way = 50
        args.shot = 5
        args.sessions = 9

    args.Dataset=Dataset
    return args

def set_up_datasets_i(args):
    if args.dataset == 'cifar100':
        import dataloader.cifar100.cifar as Dataset
        args.base_class = 60
        args.num_classes = 100
        args.way = 5
        args.shot = 5
        args.sessions = 9
    if args.dataset == "manyshotcifar":
        import dataloader.cifar100.manyshot_cifar as Dataset
        args.base_class = 60
        args.num_classes = 100
        args.way = 5
        args.shot = args.shot_num
        args.sessions = 9
    if args.dataset == 'cub200':
        import dataloader.cub200.cub200 as Dataset
        args.base_class = 100
        args.num_classes = 200
        args.way = 10
        args.shot = 5
        args.sessions = 11

    if args.dataset == 'manyshotcub':
        import dataloader.cub200.manyshot_cub as Dataset
        args.base_class = 100
        args.num_classes = 200
        args.way = 10
        args.shot = args.shot_num
        args.sessions = 11

    if args.dataset == 'mini_imagenet':
        import dataloader.miniimagenet.miniimagenet as Dataset
        args.base_class = 60
        args.num_classes = 100
        args.way = 5
        args.shot = 5
        args.sessions = 9

    if args.dataset == 'LasHeR_CL':
        import dataloader.LasHeR_CL.LasHeR_CL as Dataset
        args.base_class = 10
        args.num_classes = 18
        args.way = 1
        args.shot = 3000
        args.sessions = 9

    if args.dataset == 'VTDV_CL':
        import dataloader.VTDV_CL.VTDV_CL as Dataset
        args.base_class = 8
        args.num_classes = 15
        args.way = 1
        args.shot = 3000
        args.sessions = 8

    if args.dataset == 'mini_imagenet_withpath':
        import dataloader.miniimagenet.miniimagenet_with_img as Dataset
        args.base_class = 60
        args.num_classes = 100
        args.way = 5
        args.shot = 5
        args.sessions = 9

    if args.dataset == 'manyshotmini':
        import dataloader.miniimagenet.manyshot_mini as Dataset
        args.base_class = 60
        args.num_classes = 100
        args.way = 5
        args.shot = args.shot_num
        args.sessions = 9

    if args.dataset == 'imagenet100':
        import dataloader.imagenet100.ImageNet as Dataset
        args.base_class = 60
        args.num_classes = 100
        args.way = 5
        args.shot = 5
        args.sessions = 9

    if args.dataset == 'imagenet1000':
        import dataloader.imagenet1000.ImageNet as Dataset
        args.base_class = 600
        args.num_classes = 1000
        args.way = 50
        args.shot = 5
        args.sessions = 9

    args.Dataset = Dataset
    return args

def get_dataloader(args,session):
    if session == 0:
        trainset, trainloader, testloader = get_base_dataloader(args)
    else:
        trainset, trainloader, testloader = get_new_dataloader(args)
    return trainset, trainloader, testloader

def get_dataloader_i(args,session):
    if session == 0:
        trainset, trainloader, testloader = get_base_dataloader_i(args)
    else:
        trainset, trainloader, testloader = get_new_dataloader_i(args)
    return trainset, trainloader, testloader

def get_base_dataloader(args):
    txt_path = args.dataroot + "/index_list/" + args.dataset + "/session_" + str(0 + 1) + '.txt'
    class_index = np.arange(args.base_class)
    if args.dataset == 'cifar100':

        trainset = args.Dataset.CIFAR100(root=args.dataroot, train=True, download=True,
                                         index=class_index, base_sess=True)
        testset = args.Dataset.CIFAR100(root=args.dataroot, train=False, download=False,
                                        index=class_index, base_sess=True)

    if args.dataset == 'cub200':
        trainset = args.Dataset.CUB200(root=args.dataroot, train=True,
                                       index=class_index, base_sess=True)
        testset = args.Dataset.CUB200(root=args.dataroot, train=False, index=class_index)

    if args.dataset == 'mini_imagenet':
        trainset = args.Dataset.MiniImageNet(root=args.dataroot, train=True,
                                             index=class_index, base_sess=True)
        testset = args.Dataset.MiniImageNet(root=args.dataroot, train=False, index=class_index)

    if args.dataset == 'LasHeR_CL':
        trainset = args.Dataset.LasHeR_CL(root=args.dataroot, train=True,
                                             index=class_index, base_sess=True)
        testset = args.Dataset.LasHeR_CL(root=args.dataroot, train=False, index=class_index)

    if args.dataset == 'VTDV_CL':
        trainset = args.Dataset.VTDV_CL(root=args.dataroot, train=True,
                                             index=class_index, base_sess=True)
        testset = args.Dataset.VTDV_CL(root=args.dataroot, train=False, index=class_index)

    if args.dataset == 'imagenet100' or args.dataset == 'imagenet1000':
        trainset = args.Dataset.ImageNet(root=args.dataroot, train=True,
                                             index=class_index, base_sess=True)
        testset = args.Dataset.ImageNet(root=args.dataroot, train=False, index=class_index)

    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=args.batch_size_base, shuffle=True,
                                              num_workers=8, pin_memory=True)
    testloader = torch.utils.data.DataLoader(
        dataset=testset, batch_size=args.test_batch_size, shuffle=False, num_workers=8, pin_memory=True)

    return trainset, trainloader, testloader

def get_base_dataloader_i(args):
    txt_path = args.dataroot_i + "/index_list/" + args.dataset + "/session_" + str(0 + 1) + '.txt'
    class_index = np.arange(args.base_class)
    if args.dataset == 'cifar100':

        trainset = args.Dataset.CIFAR100(root=args.dataroot_i, train=True, download=True,
                                         index=class_index, base_sess=True)
        testset = args.Dataset.CIFAR100(root=args.dataroot_i, train=False, download=False,
                                        index=class_index, base_sess=True)

    if args.dataset == 'cub200':
        trainset = args.Dataset.CUB200(root=args.dataroot_i, train=True,
                                       index=class_index, base_sess=True)
        testset = args.Dataset.CUB200(root=args.dataroot_i, train=False, index=class_index)

    if args.dataset == 'mini_imagenet':
        trainset = args.Dataset.MiniImageNet(root=args.dataroot_i, train=True,
                                             index=class_index, base_sess=True)
        testset = args.Dataset.MiniImageNet(root=args.dataroot_i, train=False, index=class_index)

    if args.dataset == 'LasHeR_CL':
        trainset = args.Dataset.LasHeR_CL(root=args.dataroot_i, train=True,
                                             index=class_index, base_sess=True)
        testset = args.Dataset.LasHeR_CL(root=args.dataroot_i, train=False, index=class_index)

    if args.dataset == 'VTDV_CL':
        trainset = args.Dataset.VTDV_CL(root=args.dataroot_i, train=True,
                                             index=class_index, base_sess=True)
        testset = args.Dataset.VTDV_CL(root=args.dataroot_i, train=False, index=class_index)

    if args.dataset == 'imagenet100' or args.dataset == 'imagenet1000':
        trainset = args.Dataset.ImageNet(root=args.dataroot_i, train=True,
                                             index=class_index, base_sess=True)
        testset = args.Dataset.ImageNet(root=args.dataroot_i, train=False, index=class_index)

    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=args.batch_size_base, shuffle=True,
                                              num_workers=8, pin_memory=True)
    testloader = torch.utils.data.DataLoader(
        dataset=testset, batch_size=args.test_batch_size, shuffle=False, num_workers=8, pin_memory=True)

    return trainset, trainloader, testloader

def get_base_dataloader_meta(args):
    txt_path = args.dataroot + "/index_list/" + args.dataset + "/session_" + str(0 + 1) + '.txt'
    class_index = np.arange(args.base_class)
    if args.dataset == 'cifar100':
        trainset = args.Dataset.CIFAR100(root=args.dataroot, train=True, download=True,
                                         index=class_index, base_sess=True)
        testset = args.Dataset.CIFAR100(root=args.dataroot, train=False, download=False,
                                        index=class_index, base_sess=True)

    if args.dataset == 'cub200':
        trainset = args.Dataset.CUB200(root=args.dataroot, train=True,
                                       index_path=txt_path)
        testset = args.Dataset.CUB200(root=args.dataroot, train=False,
                                      index=class_index)
    if args.dataset == 'mini_imagenet':
        trainset = args.Dataset.MiniImageNet(root=args.dataroot, train=True,
                                             index_path=txt_path)
        testset = args.Dataset.MiniImageNet(root=args.dataroot, train=False,
                                            index=class_index)

    if args.dataset == 'LasHeR_CL':
        trainset = args.Dataset.LasHeR_CL(root=args.dataroot, train=True,
                                             index_path=txt_path)
        testset = args.Dataset.LasHeR_CL(root=args.dataroot, train=False,
                                            index=class_index)

    if args.dataset == 'VTDV_CL':
        trainset = args.Dataset.VTDV_CL(root=args.dataroot, train=True,
                                             index_path=txt_path)
        testset = args.Dataset.VTDV_CL(root=args.dataroot, train=False,
                                            index=class_index)

    # DataLoader(test_set, batch_sampler=sampler, num_workers=8, pin_memory=True)
    sampler = CategoriesSampler(trainset.targets, args.train_episode, args.episode_way,
                                args.episode_shot + args.episode_query)

    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_sampler=sampler, num_workers=args.num_workers,
                                              pin_memory=True)

    testloader = torch.utils.data.DataLoader(
        dataset=testset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    return trainset, trainloader, testloader

def get_base_dataloader_meta_i(args):
    txt_path = args.dataroot_i + "/index_list/" + args.dataset + "/session_" + str(0 + 1) + '.txt'
    class_index = np.arange(args.base_class)
    if args.dataset == 'cifar100':
        trainset = args.Dataset.CIFAR100(root=args.dataroot_i, train=True, download=True,
                                         index=class_index, base_sess=True)
        testset = args.Dataset.CIFAR100(root=args.dataroot_i, train=False, download=False,
                                        index=class_index, base_sess=True)

    if args.dataset == 'cub200':
        trainset = args.Dataset.CUB200(root=args.dataroot_i, train=True,
                                       index_path=txt_path)
        testset = args.Dataset.CUB200(root=args.dataroot_i, train=False,
                                      index=class_index)
    if args.dataset == 'mini_imagenet':
        trainset = args.Dataset.MiniImageNet(root=args.dataroot_i, train=True,
                                             index_path=txt_path)
        testset = args.Dataset.MiniImageNet(root=args.dataroot_i, train=False,
                                            index=class_index)

    if args.dataset == 'LasHeR_CL':
        trainset = args.Dataset.LasHeR_CL(root=args.dataroot_i, train=True,
                                             index_path=txt_path)
        testset = args.Dataset.LasHeR_CL(root=args.dataroot_i, train=False,
                                            index=class_index)

    if args.dataset == 'VTDV_CL':
        trainset = args.Dataset.VTDV_CL(root=args.dataroot_i, train=True,
                                             index_path=txt_path)
        testset = args.Dataset.VTDV_CL(root=args.dataroot_i, train=False,
                                            index=class_index)

    # DataLoader(test_set, batch_sampler=sampler, num_workers=8, pin_memory=True)
    sampler = CategoriesSampler(trainset.targets, args.train_episode, args.episode_way,
                                args.episode_shot + args.episode_query)

    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_sampler=sampler, num_workers=args.num_workers,
                                              pin_memory=True)

    testloader = torch.utils.data.DataLoader(
        dataset=testset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    return trainset, trainloader, testloader

def get_new_dataloader(args,session):
    txt_path = args.dataroot + "/index_list/" + args.dataset + "/session_" + str(session + 1) + '.txt'
    if args.dataset == 'cifar100':
        class_index = open(txt_path).read().splitlines()
        trainset = args.Dataset.CIFAR100(root=args.dataroot, train=True, download=False,
                                         index=class_index, base_sess=False)
    if args.dataset == 'cub200':
        trainset = args.Dataset.CUB200(root=args.dataroot, train=True,
                                       index_path=txt_path)
    if args.dataset == 'mini_imagenet':
        trainset = args.Dataset.MiniImageNet(root=args.dataroot, train=True,
                                       index_path=txt_path)
    if args.dataset == 'LasHeR_CL':
        trainset = args.Dataset.LasHeR_CL(root=args.dataroot, train=True,
                                             index_path=txt_path)
    if args.dataset == 'VTDV_CL':
        trainset = args.Dataset.VTDV_CL(root=args.dataroot, train=True,
                                             index_path=txt_path)
    if args.dataset == 'imagenet100' or args.dataset == 'imagenet1000':
        trainset = args.Dataset.ImageNet(root=args.dataroot, train=True,
                                       index_path=txt_path)

    if args.batch_size_new == 0:
        batch_size_new = trainset.__len__()
        trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size_new, shuffle=False,
                                                  num_workers=args.num_workers, pin_memory=True)
    else:
        trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=args.batch_size_new, shuffle=True,
                                                  num_workers=args.num_workers, pin_memory=True)

    # test on all encountered classes
    class_new = get_session_classes(args, session)

    if args.dataset == 'cifar100':
        testset = args.Dataset.CIFAR100(root=args.dataroot, train=False, download=False,
                                        index=class_new, base_sess=False)
    if args.dataset == 'cub200':
        testset = args.Dataset.CUB200(root=args.dataroot, train=False,
                                      index=class_new)
    if args.dataset == 'mini_imagenet':
        testset = args.Dataset.MiniImageNet(root=args.dataroot, train=False,
                                      index=class_new)
    if args.dataset == 'LasHeR_CL':
        testset = args.Dataset.LasHeR_CL(root=args.dataroot, train=False,
                                            index=class_new)
    if args.dataset == 'VTDV_CL':
        testset = args.Dataset.VTDV_CL(root=args.dataroot, train=False,
                                            index=class_new)
    if args.dataset == 'imagenet100' or args.dataset == 'imagenet1000':
        testset = args.Dataset.ImageNet(root=args.dataroot, train=False,
                                      index=class_new)

    testloader = torch.utils.data.DataLoader(dataset=testset, batch_size=args.test_batch_size, shuffle=False,
                                             num_workers=args.num_workers, pin_memory=True)

    return trainset, trainloader, testloader

def get_new_dataloader_i(args,session):
    txt_path = args.dataroot_i + "/index_list/" + args.dataset + "/session_" + str(session + 1) + '.txt'
    if args.dataset == 'cifar100':
        class_index = open(txt_path).read().splitlines()
        trainset = args.Dataset.CIFAR100(root=args.dataroot_i, train=True, download=False,
                                         index=class_index, base_sess=False)
    if args.dataset == 'cub200':
        trainset = args.Dataset.CUB200(root=args.dataroot_i, train=True,
                                       index_path=txt_path)
    if args.dataset == 'mini_imagenet':
        trainset = args.Dataset.MiniImageNet(root=args.dataroot_i, train=True,
                                       index_path=txt_path)
    if args.dataset == 'LasHeR_CL':
        trainset = args.Dataset.LasHeR_CL(root=args.dataroot_i, train=True,
                                             index_path=txt_path)
    if args.dataset == 'VTDV_CL':
        trainset = args.Dataset.VTDV_CL(root=args.dataroot_i, train=True,
                                             index_path=txt_path)
    if args.dataset == 'imagenet100' or args.dataset == 'imagenet1000':
        trainset = args.Dataset.ImageNet(root=args.dataroot_i, train=True,
                                       index_path=txt_path)

    if args.batch_size_new == 0:
        batch_size_new = trainset.__len__()
        trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size_new, shuffle=False,
                                                  num_workers=args.num_workers, pin_memory=True)
    else:
        trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=args.batch_size_new, shuffle=True,
                                                  num_workers=args.num_workers, pin_memory=True)

    # test on all encountered classes
    class_new = get_session_classes(args, session)

    if args.dataset == 'cifar100':
        testset = args.Dataset.CIFAR100(root=args.dataroot_i, train=False, download=False,
                                        index=class_new, base_sess=False)
    if args.dataset == 'cub200':
        testset = args.Dataset.CUB200(root=args.dataroot_i, train=False,
                                      index=class_new)
    if args.dataset == 'mini_imagenet':
        testset = args.Dataset.MiniImageNet(root=args.dataroot_i, train=False,
                                      index=class_new)
    if args.dataset == 'LasHeR_CL':
        testset = args.Dataset.LasHeR_CL(root=args.dataroot_i, train=False,
                                            index=class_new)
    if args.dataset == 'VTDV_CL':
        testset = args.Dataset.VTDV_CL(root=args.dataroot_i, train=False,
                                            index=class_new)
    if args.dataset == 'imagenet100' or args.dataset == 'imagenet1000':
        testset = args.Dataset.ImageNet(root=args.dataroot_i, train=False,
                                      index=class_new)

    testloader = torch.utils.data.DataLoader(dataset=testset, batch_size=args.test_batch_size, shuffle=False,
                                             num_workers=args.num_workers, pin_memory=True)

    return trainset, trainloader, testloader

def get_session_classes(args,session):
    class_list=np.arange(args.base_class + session * args.way)
    return class_list

def get_session_classes_i(args,session):
    class_list=np.arange(args.base_class + session * args.way)
    return class_list