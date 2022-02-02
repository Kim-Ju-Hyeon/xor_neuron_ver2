import os
import pickle
from collections import defaultdict
from tqdm import tqdm
import time

from foolbox import PyTorchModel, accuracy
from foolbox.attacks import LinfPGD, FGSM, L2CarliniWagnerAttack, LinfDeepFoolAttack, BoundaryAttack, EADAttack
from foolbox.attacks.blended_noise import LinearSearchBlendedUniformNoiseAttack

import torch.nn as nn
import torch.utils.data
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

from model import *
from model.model_for_foolbox import *
from utils.train_helper import load_model
from utils.logger import get_logger
from utils.corpus import Corpus

from autoattack import AutoAttack
from autoattack import checks

logger = get_logger('log_adv_attack')


class XorNeuronAttack(object):

    def __init__(self, config):
        self.config = config
        self.dataset_conf = config.dataset
        self.model_conf = config.model
        self.test_conf = config.test
        self.use_gpu = config.use_gpu
        self.gpus = config.gpus

        self.id = np.random.randint(1000, 9999, size=1)[0]

        if self.dataset_conf.name == 'mnist':
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
            self.test_dataset = datasets.MNIST(root=self.dataset_conf.data_path,
                                               train=False,
                                               transform=transform,
                                               download=False)

            self.axis = None
        elif self.dataset_conf.name == 'cifar10':
            transform = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
            self.test_dataset = datasets.CIFAR10(root=self.dataset_conf.data_path,
                                                 train=False,
                                                 transform=transform,
                                                 download=False)

            self.axis = -3

        elif self.dataset_conf.name == 'cifar100':
            transform = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
            self.test_dataset = datasets.CIFAR100(root=self.dataset_conf.data_path,
                                             train=False,
                                             transform=transform,
                                             download=False)
            self.axis = -3
        else:
            raise ValueError("Non-supported dataset!")

        if self.model_conf.name == "ComplexNeuronMLP":
            self.model = ComplexNeuronMLP_For_FoolBox(self.config)
        elif self.model_conf.name == "Control_MLP":
            self.model = Control_MLP_For_FoolBox(self.config)
        elif self.model_conf.name == "ComplexNeuronConv":
            self.model = ComplexNeuronConv_For_FoolBox(self.config)
        elif self.model_conf.name == "control_model":
            self.model = Control_Conv_For_FoolBox(self.config)
        elif self.model_conf.name == "Control_Conv":
            self.model = Control_Conv_For_FoolBox(self.config)
        elif self.model_conf.name == 'Xor_ResNet':
            self.model = ResNet20_Xor_adv_attack(self.config)
        elif self. model_conf.name == 'resnet20':
            self.model = ResNet20_adv_attack(self.config)
        else:
            raise ValueError("Non-supported Model!")

    def gradient_base_attack(self, attack_config):
        test_loader = DataLoader(dataset=self.test_dataset,
                                 batch_size=attack_config.attack.batch_size,
                                 shuffle=False)

        load_model(self.model, self.config.model_save + self.test_conf.test_model)

        device = torch.cuda.current_device()

        if self.use_gpu:
            # model = nn.DataParallel(self.model, device_ids=device).cuda()
            model = nn.DataParallel(self.model).cuda()

        preprocessing_mean = attack_config.preprocessing.mean
        preprocessing_std = attack_config.preprocessing.std
        preprocessing = dict(mean=preprocessing_mean, std=preprocessing_std, axis=self.axis)

        bounds = (0, 1)
        # attack_config.preprocessing.bounds

        _attack = attack_config.attack.name
        epsilons = attack_config.attack.epsilons

        if _attack == "LinfPGD":
            attack = LinfPGD()
        elif _attack == "FGSM":
            attack = FGSM()
        elif _attack == "LinfDeepFoolAttack":
            attack = LinfDeepFoolAttack()
        else:
            raise ValueError("Non-supported Attack!")

        model.eval()
        fmodel = PyTorchModel(model, bounds=bounds, device=device, preprocessing=preprocessing)

        score = torch.zeros(len(epsilons))
        correct = 0
        results = defaultdict(list)

        for i, (imgs, labels) in enumerate(tqdm(test_loader)):
            # imgs, labels = imgs.to(device), labels.to(device)
            imgs, labels = imgs.cuda(), labels.cuda()

            clean_acc = accuracy(fmodel, imgs, labels)
            correct += clean_acc

            raw_advs, clipped_advs, success = attack(fmodel, imgs, labels, epsilons=epsilons)
            success = success.float().mean(axis=-1).data.cpu()

            if (i + 1) % 10 == 0:
                logger.info(f"Clean Accuracy = {clean_acc:.4}")

                for n, raw_success in enumerate(success):
                    raw_robust = 1 - raw_success
                    logger.info(f"{i + 1}Iter epsilons:{epsilons[n]} Robustness Accuracy={raw_robust:.4}")

            score += success

        if attack_config.save_imgs:
            for j in range(len(raw_advs)):
                raw_advs[j] = raw_advs[j].data.cpu()
                raw_advs[j] = raw_advs[j][:5]
            imgs = imgs[:5]

            results["adv_img"] = raw_advs
            results["clean_img"] = imgs.data.cpu()

        score = score / (i + 1)
        correct = correct / (i + 1)
        robust_acc = 1 - score

        results["epsilons"] = epsilons
        results["clean_acc"] = correct
        results["robust_acc"] = robust_acc

        logger.info(f"Avg Test Accuracy = {correct:.4}")

        pickle.dump(results, open(os.path.join(self.config.save_dir, f'{_attack}_attack_results.p'), 'wb'))

    def boundary_attack(self, attack_config):
        test_loader = DataLoader(dataset=self.test_dataset,
                                 batch_size=attack_config.attack.batch_size,
                                 shuffle=False)

        load_model(self.model, self.config.model_save + self.test_conf.test_model)

        device = torch.cuda.current_device()

        if self.use_gpu:
            model = nn.DataParallel(self.model).cuda()

        preprocessing_mean = attack_config.preprocessing.mean
        preprocessing_std = attack_config.preprocessing.std
        preprocessing = dict(mean=preprocessing_mean, std=preprocessing_std, axis=self.axis)

        bounds = (0, 1)

        _attack = attack_config.attack.name

        if _attack == "BoundaryAttack":
            attack = BoundaryAttack(steps=5000)
        else:
            raise ValueError("Non-supported Attack!")

        init_attack = LinearSearchBlendedUniformNoiseAttack(steps=2000)

        model.eval()
        fmodel = PyTorchModel(model, bounds=bounds, device=device, preprocessing=preprocessing)

        total = 0
        correct = 0
        results = defaultdict(list)
        clean_correct = 0

        for i, (imgs, labels) in enumerate(tqdm(test_loader)):
            # imgs, labels = imgs.to(device), labels.to(device)
            imgs, labels = imgs.cuda(), labels.cuda()

            noise_imgs = init_attack.run(fmodel, imgs, labels)
            noise_out = fmodel(noise_imgs)
            _, noise_pred = torch.max(noise_out.data, 1)
            cls_samples = {}

            for cls in range(10):
                idx = noise_pred != cls
                cls_samples[cls] = noise_imgs[idx][0]

            starting_points = []
            for y in labels:
                starting_points.append(cls_samples[int(y)])
            starting_points = torch.stack(starting_points, dim=0).cuda()

            best_advs = attack.run(fmodel, imgs, labels, starting_points=starting_points)

            out = fmodel(best_advs)
            _, pred = torch.max(out.data, 1)
            total += labels.size(0)
            score = (pred == labels).sum().item()
            correct += score

            robust_acc = score / labels.size(0)

            clean_acc = accuracy(fmodel, imgs, labels)
            clean_correct += clean_acc

            if (i + 1) % 10 == 0:
                logger.info(f"Clean Accuracy = {clean_acc:.4}")
                logger.info(f"Robustness Accuracy = {robust_acc:.4}")

        imgs = imgs[:10].data.cpu()
        best_advs = best_advs[:10].data.cpu()

        results["adv_img"] = best_advs
        results["clean_img"] = imgs

        robust_accuracy = correct / total
        test_accuracy = clean_correct / (i + 1)

        results["clean_acc"] = test_accuracy
        results["robust_acc"] = robust_accuracy

        logger.info(f"Avg Test Accuracy = {clean_correct / (i + 1):.4}")

        pickle.dump(results, open(os.path.join(self.config.save_dir, f'{_attack}_attack_results.p'), 'wb'))

    def auto_attack(self, attack_config):
        test_loader = DataLoader(dataset=self.test_dataset,
                                 batch_size=attack_config.batch_size,
                                 shuffle=False)

        l = [x for (x, y) in test_loader]
        x_test = torch.cat(l, 0).cuda()
        l = [y for (x, y) in test_loader]
        y_test = torch.cat(l, 0).cuda()

        load_model(self.model, self.config.model_save + self.test_conf.test_model)

        if self.use_gpu:
            self.model = nn.DataParallel(self.model).cuda()

        log_path = self.config.save_dir + f"/log_adv_attack_AutoAttack_ver2_{self.id}.txt"

        with torch.no_grad():
            for eps in attack_config.epsilon:
                adversary = AutoAttack2(self.model, norm=attack_config.norm, eps=eps, version='rand',
                                        log_path=log_path)

                imgs, labels = x_test.cuda(), y_test.cuda()
                _, robust_accuracy_dict = adversary.run_standard_evaluation(imgs, labels, bs=128, return_labels=False)

        pickle.dump(robust_accuracy_dict, open(os.path.join(self.config.save_dir, 'auto_attack_result.p'), 'wb'))


class AutoAttack2(AutoAttack):
    def run_standard_evaluation(self, x_orig, y_orig, bs=128, return_labels=False):
        if self.verbose:
            print('using {} version including {}'.format(self.version,
                                                         ', '.join(self.attacks_to_run)))

        # checks on type of defense
        if self.version != 'rand':
            checks.check_randomized(self.get_logits, x_orig[:bs].to(self.device),
                                    y_orig[:bs].to(self.device), bs=bs, logger=self.logger)
        n_cls = checks.check_range_output(self.get_logits, x_orig[:bs].to(self.device),
                                          logger=self.logger)
        checks.check_dynamic(self.model, x_orig[:bs].to(self.device), self.is_tf_model,
                             logger=self.logger)
        # checks.check_n_classes(n_cls, self.attacks_to_run, self.apgd_targeted.n_target_classes,
        #                        self.fab.n_target_classes, logger=self.logger)

        with torch.no_grad():
            # calculate accuracy
            n_batches = int(np.ceil(x_orig.shape[0] / bs))
            robust_flags = torch.zeros(x_orig.shape[0], dtype=torch.bool, device=x_orig.device)
            y_adv = torch.empty_like(y_orig)
            for batch_idx in range(n_batches):
                start_idx = batch_idx * bs
                end_idx = min((batch_idx + 1) * bs, x_orig.shape[0])

                x = x_orig[start_idx:end_idx, :].clone().to(self.device)
                y = y_orig[start_idx:end_idx].clone().to(self.device)
                output = self.get_logits(x).max(dim=1)[1]
                y_adv[start_idx: end_idx] = output
                correct_batch = y.eq(output)
                robust_flags[start_idx:end_idx] = correct_batch.detach().to(robust_flags.device)

            robust_accuracy = torch.sum(robust_flags).item() / x_orig.shape[0]
            robust_accuracy_dict = {'clean': robust_accuracy}

            if self.verbose:
                self.logger.log('initial accuracy: {:.2%}'.format(robust_accuracy))

            x_adv = x_orig.clone().detach()
            startt = time.time()
            for attack in self.attacks_to_run:
                # item() is super important as pytorch int division uses floor rounding
                num_robust = torch.sum(robust_flags).item()

                if num_robust == 0:
                    break

                n_batches = int(np.ceil(num_robust / bs))

                robust_lin_idcs = torch.nonzero(robust_flags, as_tuple=False)
                if num_robust > 1:
                    robust_lin_idcs.squeeze_()

                for batch_idx in range(n_batches):
                    start_idx = batch_idx * bs
                    end_idx = min((batch_idx + 1) * bs, num_robust)

                    batch_datapoint_idcs = robust_lin_idcs[start_idx:end_idx]
                    if len(batch_datapoint_idcs.shape) > 1:
                        batch_datapoint_idcs.squeeze_(-1)
                    x = x_orig[batch_datapoint_idcs, :].clone().to(self.device)
                    y = y_orig[batch_datapoint_idcs].clone().to(self.device)

                    # make sure that x is a 4d tensor even if there is only a single datapoint left
                    if len(x.shape) == 3:
                        x.unsqueeze_(dim=0)

                    # run attack
                    if attack == 'apgd-ce':
                        # apgd on cross-entropy loss
                        self.apgd.loss = 'ce'
                        self.apgd.seed = self.get_seed()
                        adv_curr = self.apgd.perturb(x, y)  # cheap=True

                    elif attack == 'apgd-dlr':
                        # apgd on dlr loss
                        self.apgd.loss = 'dlr'
                        self.apgd.seed = self.get_seed()
                        adv_curr = self.apgd.perturb(x, y)  # cheap=True

                    elif attack == 'fab':
                        # fab
                        self.fab.targeted = False
                        self.fab.seed = self.get_seed()
                        adv_curr = self.fab.perturb(x, y)

                    elif attack == 'square':
                        # square
                        self.square.seed = self.get_seed()
                        adv_curr = self.square.perturb(x, y)

                    elif attack == 'apgd-t':
                        # targeted apgd
                        self.apgd_targeted.seed = self.get_seed()
                        adv_curr = self.apgd_targeted.perturb(x, y)  # cheap=True

                    elif attack == 'fab-t':
                        # fab targeted
                        self.fab.targeted = True
                        self.fab.n_restarts = 1
                        self.fab.seed = self.get_seed()
                        adv_curr = self.fab.perturb(x, y)

                    else:
                        raise ValueError('Attack not supported')

                    output = self.get_logits(adv_curr).max(dim=1)[1]
                    false_batch = ~y.eq(output).to(robust_flags.device)
                    non_robust_lin_idcs = batch_datapoint_idcs[false_batch]
                    robust_flags[non_robust_lin_idcs] = False

                    x_adv[non_robust_lin_idcs] = adv_curr[false_batch].detach().to(x_adv.device)
                    y_adv[non_robust_lin_idcs] = output[false_batch].detach().to(x_adv.device)

                    if self.verbose:
                        num_non_robust_batch = torch.sum(false_batch)
                        self.logger.log('{} - {}/{} - {} out of {} successfully perturbed'.format(
                            attack, batch_idx + 1, n_batches, num_non_robust_batch, x.shape[0]))

                robust_accuracy = torch.sum(robust_flags).item() / x_orig.shape[0]
                robust_accuracy_dict[attack] = robust_accuracy
                if self.verbose:
                    self.logger.log('robust accuracy after {}: {:.2%} (total time {:.1f} s)'.format(
                        attack.upper(), robust_accuracy, time.time() - startt))

            # check about square
            checks.check_square_sr(robust_accuracy_dict, logger=self.logger)

            # final check
            if self.verbose:
                if self.norm == 'Linf':
                    res = (x_adv - x_orig).abs().reshape(x_orig.shape[0], -1).max(1)[0]
                elif self.norm == 'L2':
                    res = ((x_adv - x_orig) ** 2).reshape(x_orig.shape[0], -1).sum(-1).sqrt()
                elif self.norm == 'L1':
                    res = (x_adv - x_orig).abs().reshape(x_orig.shape[0], -1).sum(dim=-1)
                self.logger.log('max {} perturbation: {:.5f}, nan in tensor: {}, max: {:.5f}, min: {:.5f}'.format(
                    self.norm, res.max(), (x_adv != x_adv).sum(), x_adv.max(), x_adv.min()))
                self.logger.log('robust accuracy: {:.2%}'.format(robust_accuracy))
        if return_labels:
            return x_adv, y_adv
        else:
            return x_adv, robust_accuracy_dict
