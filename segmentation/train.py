"""
UNet
Train Unet model
"""
import numpy as np
import os
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from dataset import get_train_valid_loader, get_test_loader
from utils import Option, encode_and_save, compute_iou
from skimage import io
from skimage.transform import resize
import matplotlib.pyplot as plt
import tensorflow as tf
from torch.autograd import Function
from model_raunet import UNet2

torch.cuda.set_device(Option.gpu_device)

#os.environ['CUDA_VISIBLE_DEVICES'] = "2,3"


def makedir(path):
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
    else:
        pass



class cDiceCoeff(Function):
    """Dice coeff for individual examples"""

    def forward(self, input, target):
        self.save_for_backward(input, target)
        self.inter = torch.dot(input.view(-1), target.view(-1)) + 1
        self.union = torch.sum(input) + torch.sum(target) + 1

        t = 2 * self.inter.float() / self.union.float()
        return t


def cDSC(input, target):
    """Dice coeff for batches"""
    if input.is_cuda:
        s = torch.FloatTensor(1).cuda().zero_()
    else:
        s = torch.FloatTensor(1).zero_()

    for i, c in enumerate(zip(input, target)):
        s = s + cDiceCoeff().forward(c[0], c[1])
        # print(s,DiceCoeff().forward(c[0], c[1]))

    return 1-(s / (i + 1))


class DiceCoeff(Function):
    """Dice coeff for individual examples"""

    def forward(self, input, target):
        self.save_for_backward(input, target)
        self.inter = torch.dot(input.view(-1), target.view(-1)) + 0.0001
        self.union = torch.sum(input) + torch.sum(target) + 0.0001

        t = 2 * self.inter.float() / self.union.float()
        return t

    # This function has only a single output, so it gets only one gradient
    # def backward(self, grad_output):
    #
    #     input, target = self.saved_variables
    #     grad_input = grad_target = None
    #
    #     if self.needs_input_grad[0]:
    #         grad_input = grad_output * 2 * (target * self.union + self.inter) \
    #                      / self.union * self.union
    #     if self.needs_input_grad[1]:
    #         grad_target = None
    #
    #     return grad_input, grad_target


def dice_coeff(input, target):
    """Dice coeff for batches"""
    if input.is_cuda:
        s = torch.FloatTensor(1).cuda().zero_()
    else:
        s = torch.FloatTensor(1).zero_()

    for i, c in enumerate(zip(input, target)):
        s = s + DiceCoeff().forward(c[0], c[1])
        # print(s,DiceCoeff().forward(c[0], c[1]))

    return 1-(s / (i + 1))

# smooth = 1.
# def dice_coef(y_true, y_pred):
#     y_true_f = K.flatten(y_true)
#     y_pred_f = K.flatten(y_pred)
#     intersection = K.sum(y_true_f * y_pred_f)
#     return (2. * intersection + smooth) / (K.sum(y_true_f * y_true_f) + K.sum(y_pred_f * y_pred_f) + smooth)
# def dice_coef_loss(y_true, y_pred):
#     return 1. - dice_coef(y_true, y_pred)

# def dice_coef_loss(target, prediction, axis=(1, 2), smooth=1.):
#     """
#     """
#     intersection = tf.reduce_sum(prediction * target, axis=axis)
#     p = tf.reduce_sum(prediction, axis=axis)
#     t = tf.reduce_sum(target, axis=axis)
#     numerator = tf.reduce_mean(intersection + smooth)
#     denominator = tf.reduce_mean(t + p + smooth)
#     dice_loss = -tf.log(2.*numerator) + tf.log(denominator)
#
#     return dice_loss

# class DiceLoss(nn.Module):
#     def __init__(self):
#         super(DiceLoss, self).__init__()
#
#     def forward(self, input, target):
#         N = target.size(0)
#         smooth = 1
#
#         input_flat = input.view(N, -1)
#         target_flat = target.view(N, -1)
#
#         intersection = input_flat * target_flat
#
#         loss = 2 * (intersection.sum(1) + smooth) / (input_flat.sum(1) + target_flat.sum(1) + smooth)
#         loss = 1 - loss.sum() / N
#
#         return loss
# def DiceLoss(input,target):
#     N = target.size(0)
#     smooth = 1
#
#     input_flat = input.view(N, -1)
#     target_flat = target.view(N, -1)
#
#     intersection = input_flat * target_flat
#
#     loss = 2 * (intersection.sum(1) + smooth) / (input_flat.sum(1) + target_flat.sum(1) + smooth)
#     loss = 1 - loss.sum() / N

def train(model, train_loader, opt, criterion, epoch,savepath):
    model.train()
    num_batches = 0
    avg_loss = 0
    txtpath = os.path.join(savepath,'logs.txt')
    with open(txtpath, 'a') as file:
        for batch_idx, sample_batched in enumerate(train_loader):
            data = sample_batched['image']
            target = sample_batched['mask']
            data, target = Variable(data.type(opt.dtype)), Variable(target.type(opt.dtype))
            optimizer.zero_grad()
            output = model(data)
            # output = (output > 0.5).type(opt.dtype)	# use more gpu memory, also, loss does not change if use this line
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            avg_loss += loss.item()  #data[0] to item()
            num_batches += 1
        avg_loss /= num_batches
        dice = 1-avg_loss
        # avg_loss /= len(train_loader.dataset)
        # print('epoch: ' + str(epoch) + ', train loss: ' + str(avg_loss)+',DSC:'+str(dice))
        print('epoch: ' + str(epoch) + ', train loss: ' + str(avg_loss))
        file.write('epoch: ' + str(epoch) + ', train loss: ' + str(avg_loss) + '\n')

def val(model, val_loader, opt, criterion, epoch):
    model.eval()
    num_batches = 0
    avg_loss = 0
    with open('logs.txt', 'a') as file:
        for batch_idx, sample_batched in enumerate(val_loader):
            data = sample_batched['image']
            target = sample_batched['mask']
            data, target = Variable(data.type(opt.dtype)), Variable(target.type(opt.dtype))
            output = model.forward(data)
            # output = (output > 0.5).type(opt.dtype)	# use more gpu memory, also, loss does not change if use this line
            loss = criterion(output, target)
            avg_loss += loss.data[0]
            num_batches += 1
        avg_loss /= num_batches
        # avg_loss /= len(val_loader.dataset)

        print('epoch: ' + str(epoch) + ', validation loss: ' + str(avg_loss))
        file.write('epoch: ' + str(epoch) + ', validation loss: ' + str(avg_loss) + '\n')

# train and validation
def run(model, train_loader, val_loader, opt, criterion):
    for epoch in range(1, opt.epochs):
        train(model, train_loader, opt, criterion, epoch)
        val(model, val_loader, opt, criterion, epoch)

# only train
def run_train(model, train_loader, opt, criterion,savepath):
    for epoch in range(1, opt.epochs):
        train(model, train_loader, opt, criterion, epoch,savepath)

# make prediction
def run_test(model, test_loader, opt):
    """
    predict the masks on testing set
    :param model: trained model
    :param test_loader: testing set
    :param opt: configurations
    :return:
        - predictions: list, for each elements, numpy array (Width, Height)
        - img_ids: list, for each elements, an image id string
    """
    model.eval()
    predictions = []
    img_ids = []
    for batch_idx, sample_batched in enumerate(test_loader):
        data, img_id, height, width = sample_batched['image'], sample_batched['img_id'], sample_batched['height'], sample_batched['width']
        data = Variable(data.type(opt.dtype))
        output = model.forward(data)
        # output = (output > 0.5)
        output = output.data.cpu().numpy()
        output = output.transpose((0, 2, 3, 1))    # transpose to (B,H,W,C)
        for i in range(0,output.shape[0]):
            pred_mask = np.squeeze(output[i])
            id = img_id[i]
            h = height[i]
            w = width[i]
            # in p219 the w and h above is int
            # in local the w and h above is LongTensor
            if not isinstance(h, int):
                h = h.cpu().numpy()
                w = w.cpu().numpy()
            pred_mask = resize(pred_mask, (h, w), mode='constant')
            pred_mask = (pred_mask > 0.5)
            predictions.append(pred_mask)
            img_ids.append(id)

    return predictions, img_ids

if __name__ == '__main__':
    """Train Unet model"""
    opt = Option()
    # for filename in os.listdir(opt.data_dir):
    #     csv_path = str(filename) + '-' + 'DSC.csv'
    #     #print(csv_path)
    #     if os.path.exists(os.path.join(opt.results_dir, csv_path)):
    #         print(filename,'has been finished!')
    #         continue
    #     else:
    #         print(filename,'is trainning')
    #         if os.path.exists(opt.root_dir):
    #             shutil.rmtree(opt.root_dir)
    #         if os.path.exists(opt.test_dir):
    #             shutil.rmtree(opt.test_dir)
    #         os.makedirs(opt.test_dir)
    #         os.makedirs(opt.root_dir)
    #         for traindir in os.listdir(opt.data_dir):
    #             for dir1 in os.listdir(os.path.join(opt.data_dir,traindir)):
    #                 one_dir = os.path.join(opt.data_dir,traindir,dir1)
    #                 #print(one_dir)
    #                 shutil.copytree(one_dir,os.path.join(opt.root_dir,dir1))
    #         for testdir in os.listdir(os.path.join(opt.data_dir,filename)):
    #             dir2 = os.path.join(opt.root_dir,testdir)
    #             shutil.move(dir2,opt.test_dir)

    for fold in range(5):
        fold_number = fold + 1
        train_dir = os.path.join('./Data/aug18_fold', 'Tr' + str(fold_number))
        test_dir = os.path.join('./Data/ourmodeltest', 'Te' + str(fold_number))
        csv_path = str(fold_number) + '-' + 'DSC.csv'
        savepath = os.path.join('./clude_results','raunet',str(fold_number))
        makedir(savepath)
            #train
            # if opt.mul_channel:
            #     input_channel = 3
            # else:
            #     input_channel = 1
        model = UNet2(1,1)
        if opt.is_train:
            # split all data to train and validation, set split = True
            # train_loader, val_loader = get_train_valid_loader(opt.root_dir, batch_size=opt.batch_size,
            #                                       split=False, shuffle=opt.shuffle,
            #                                       num_workers=opt.num_workers,
            #                                       val_ratio=0.1, pin_memory=opt.pin_memory)

            # load all data for training
            train_loader = get_train_valid_loader(train_dir, batch_size=opt.batch_size,
                                                  split=False, shuffle=opt.shuffle,
                                                  num_workers=opt.num_workers,
                                                  val_ratio=0.1, pin_memory=opt.pin_memory)
            test_loader = get_test_loader(test_dir, batch_size=opt.batch_size, shuffle=opt.shuffle,
                                          num_workers=opt.num_workers, pin_memory=opt.pin_memory)
            if opt.n_gpu > 1:
                model = nn.DataParallel(model,device_ids=[0,1])  #
            if opt.is_cuda:
                model = model.cuda()
            optimizer = optim.Adam(model.parameters(), lr=opt.learning_rate, weight_decay=opt.weight_decay)
            # criterion = nn.BCELoss().cuda()
            criterion = dice_coeff
            # start to run a training
            run_train(model, train_loader, opt, criterion,savepath)
            # make prediction on validation set
            predictions, img_ids = run_test(model, test_loader, opt)
            encode_and_save(predictions, img_ids,savepath)
            # compute IOU between prediction and ground truth masks
            compute_iou(fold_number,predictions, img_ids, test_loader,test_dir,savepath)
            # SAVE model
            if opt.save_model:
                torch.save(model.state_dict(), os.path.join(savepath, 'model-unet-dice.pth'))
        else:
            # load testing data for making predictions
            test_loader = get_test_loader(opt.test_dir, batch_size=opt.batch_size, shuffle=opt.shuffle,
                                          num_workers=opt.num_workers, pin_memory=opt.pin_memory)
            # load the model and run test
            model.load_state_dict(torch.load(os.path.join(opt.checkpoint_dir, 'model-unet.pth')))
            if opt.n_gpu > 1:
                model = nn.DataParallel(model)
            if opt.is_cuda:
                model = model.cuda()
            # print(1)
            predictions, img_ids = run_test(model, test_loader, opt)

            # run length encoding and save as csv
            encode_and_save(predictions, img_ids)
