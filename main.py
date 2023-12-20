import numpy as np
import cv2
import torchvision.transforms
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch
import pandas as pd
from PIL import Image

print(torch.cuda.is_available())

from torch.nn import Module
from torch.nn import Conv2d
from torch.nn import Linear
from torch.nn import MaxPool2d, MaxUnpool2d
from torch.nn import ReLU, Sigmoid, Softmax
from torch.nn import LogSoftmax
from torch import flatten
from torch.nn import MSELoss, CrossEntropyLoss
import torch.optim as optim
from tqdm import tqdm
from scipy import ndimage
import os

import torchvision.transforms as transforms




class FullImageDataset(Dataset):

    def __init__(self, image_names, labels):
        self.labels = labels
        self.dir = dir
        self.image_names = image_names


    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        return self.image_names[idx].float()






class FeatureNet(Module):

    def __init__(self):
        super(FeatureNet, self).__init__()
        self.conv1 = Conv2d(3, 16, 21, padding=10)
        self.max_pool = MaxPool2d(kernel_size=(2, 2), stride=(2, 2), return_indices=True)
        self.test = Conv2d(16, 16, 21, padding=10)
        self.conv2 = Conv2d(16, 16, 21, padding=10)
        self.up_conv1 = Conv2d(16, 16, 21, padding=10)
        self.max_unpool = MaxUnpool2d(kernel_size=(2, 2), stride=(2, 2))
        self.up_conv2 = Conv2d(32, 16, 21, padding=10)

        self.up_conv3 = Conv2d(32, 16, 21, padding=10)
        self.test2 = Conv2d(16, 16, 21, padding=10)

        self.before_sm = Conv2d(16, 8, 21, padding=10)

        self.to_sm = Conv2d(8, 3, 11, padding=5)
        self.after = Conv2d(3, 16, 21, padding=10)

        self.conv3 = Conv2d(16, 16, 21, padding=10)
        self.conv4 = Conv2d(16, 16, 21, padding=10)
        self.up_conv4 = Conv2d(16, 16, 21, padding=10)
        self.max_unpool = MaxUnpool2d(kernel_size=(2, 2), stride=(2, 2))
        self.up_conv5 = Conv2d(32, 16, 21, padding=10)
        self.up_conv6 = Conv2d(32, 16, 21, padding=10)

        self.before_final = Conv2d(16, 16, 21, padding=10)

        self.to_full = Conv2d(16, 3, 21, padding=10)

        self.relu = ReLU()
        self.sig = Sigmoid()
        self.sm = Softmax(dim=1)
        self.blur = torchvision.transforms.GaussianBlur(3, sigma=.1)

    def forward(self, x):
        into = self.relu(self.test(self.relu(self.conv1(x))))
        orig, ind1 = self.max_pool(into)
        c1, ind2 = self.max_pool(self.relu(self.conv2(orig)))
        c2 = self.relu(self.up_conv1(c1))
        c3 = self.max_unpool(c2, indices=ind2, output_size=ind1.size())

        c3 = torch.cat([c3, orig], dim=1)
        c4 = self.max_unpool(self.relu(self.up_conv2(c3)), indices=ind1)

        c4 = torch.cat([c4, into], dim=1)

        c5 = self.relu(self.up_conv3(c4))
        upinto = self.relu(self.test2(c5))
        c6 = self.relu(self.before_sm(upinto))
        c7 = self.to_sm(c6)
        preds = self.sm(c7)


        aft = self.relu(self.after(preds))

        d0, ind3 = self.max_pool(self.relu(self.conv3(aft)))
        d1, ind4 = self.max_pool(self.relu(self.conv4(d0)))
        d2 = self.relu(self.up_conv4(d1))
        d3 = self.max_unpool(d2, indices=ind4, output_size=d0.size())
        d3 = torch.cat([d3, d0], dim=1)
        d4 = self.max_unpool(self.relu(self.up_conv5(d3)), indices=ind3)
        d4 = torch.cat([d4, aft], dim=1)
        d5 = self.relu(self.up_conv6(d4))
        d6 = self.before_final(d5)
        d7 = self.relu(d6)

        full = self.sig(self.to_full(d7))

        return full, preds


torch.cuda.empty_cache()
#device = "cpu"#"cuda"
criterion = MSELoss()


def tv_regularization(predictions, alpha=.0005):
    return 0
    # predictions = torch.argmax(probabilites, dim=1)
    diff_x = predictions[:, :, 1:, :] - predictions[:, :, :-1, :]
    diff_y = predictions[:, :, :, 1:] - predictions[:, :, :, :-1]
    tv_loss = alpha * (torch.sum(torch.pow(diff_x, 2)) + torch.sum(torch.pow(diff_y, 2))) / predictions.shape[0] / \
              predictions.shape[1] / predictions.shape[2] / predictions.shape[3]

    return tv_loss


def few_channel_penalty(predictions, alpha=.01):
    return 0
    means = torch.mean(predictions, dim=[2, 3])
    batch_max_means = torch.max(means, dim=1).values - (1 / predictions.shape[1])
    return alpha * torch.mean(batch_max_means)


def uncertianty_penalty(predictions, alpha=.0001):
    return torch.mean(.2500001 - torch.pow(predictions - .5, 2)) * alpha


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


def Soft_Normalized_Cut_Loss(image, classes, r=1, s2=1):
    # Long lines because I keep running out of memory
    K = classes.shape[0]

    def weight(u, s):
        ret = torch.exp(torch.sum(torch.pow(
            (u - s.view(u.shape[0], u.shape[1], 1, 1).expand((u.shape[0], u.shape[1], u.shape[2], u.shape[3]))), 2),
                                  dim=(1)) / s2)
        ret[ret < r] = 0
        return ret

    t_loss = 0

    for k in range(K):
        for i in range(image.shape[2]):
            for j in range(image.shape[3]):
                w = weight(image, image[:, :, i, j])
                t_loss += torch.sum(w * classes[:, k, i, j].view(classes.shape[0], 1, 1).expand(
                    (classes.shape[0], classes.shape[2], classes.shape[3])) * classes[:, k, :, :]) / torch.sum(
                    w * classes[:, k, i, j].view(classes.shape[0], 1, 1).expand(
                        (classes.shape[0], classes.shape[2], classes.shape[3])))
                print(t_loss)

    return K - t_loss


def Modified_Soft_Normalized_Cut_Loss(image, classes):
    K = classes.shape[1]

    # def weighted_average(values, weights):
    #     updated = torch.where(weights < .000001, .000001, weights)
    #     val = torch.sum(updated*values)/torch.sum(updated)
    #     return val
    def avg(mask, values, weights):
        mask[mask == 0] = .00001
        values[values == 0] = .00001
        weights[weights == 0] = .00001

        mask = torch.stack([mask for i in range(values.shape[1])], dim=1)
        masked = values * weights
        return torch.mean(torch.sum(masked, dim=(0, 2, 3))) / (.000001 + torch.sum(weights))

    def weighted_average(mask, values, weights):

        weights[weights == 0] = .00001
        mask[mask == 0] = .00001
        values[values == 0] = .00001

        mask = torch.stack([mask for i in range(values.shape[1])], dim=1)
        masked = mask * values
        non_zero = torch.sum(mask[0])
        avg = torch.sum(masked, dim=(0, 2, 3)) / (.000001 + non_zero)
        assert values.shape[1] == avg.shape[0]

        for i in range(avg.shape[0]):
            masked[:, i, :, :] -= avg[i]

        masked *= weights * mask

        var = torch.pow(masked, 2)

        scaled_vars = torch.sum(var) / (torch.sum(mask) + .000001)

        return scaled_vars

    def distance(values, classes, k):

        max_classes = torch.argmax(classes, dim=1, keepdim=True)
        adjusted_max = torch.where(max_classes != k, 0, max_classes)
        mask = torch.where(adjusted_max == k, 1, adjusted_max)
        full_mask = torch.cat([mask[:, 0, :, :].unsqueeze(1) for i in range(values.shape[1])], dim=1)
        masked_image = values * full_mask

    def pairwise_distances(x):
        square_distances = torch.sum((x[:, None] - x) ** 2, dim=-1)
        distances = torch.sqrt(square_distances)
        return distances

    def average_distance(x):
        distances = pairwise_distances(x)
        num_points = x.shape[0]

        sum_distances = torch.sum(distances) / 2  # Divide by 2 to account for double counting

        return sum_distances

    def uncertainty(weights):
        return torch.mean(.25 - (.5 - weights) ** 2) * .0001

    losses = torch.zeros(K)
    means = torch.zeros((K, image.shape[1]))

    for k in range(K):
        image_copy = image.clone()
        classes_copy = classes.clone()
        classes_mask = (torch.argmax(classes_copy, dim=1) == k).float()
        means[k] = avg(classes_mask, image_copy,
                       torch.cat([classes_copy[:, k, :, :].unsqueeze(1) for i in range(image_copy.shape[1])], dim=1))

    r = - average_distance(means) * 1000
    return r

#######################################################################################################################
def run():
    data = pd.read_csv("cat_dog.csv")
    data = data[~(data["labels"]==1)]

    # Define the target size for the images
    target_size = (250, 250)  # You can choose the size you want

    # Create a list to store the resized tensors
    resized_tensors = []

    # Define a transformation to resize and convert images to tensors
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
    ])

    # Iterate over the image paths
    for image_path in data["image"]:
        try:
            # Open the image using PIL
            img = Image.open(f"cat_dog/{image_path}")

            # Apply the transformation
            resized_tensor = transform(img)

            # Append the resized tensor to the list
            resized_tensors.append(resized_tensor)
        except Exception as e:
            pass

    # Stack the tensors into a single tensor along a new dimension (batch dimension)
    final_tensor = torch.stack(resized_tensors)

    ds = FullImageDataset(final_tensor[:int(.8*len(final_tensor))], data["labels"].tolist()[:int(.8*len(final_tensor))])
    dl = DataLoader(ds, 64)

    ds_test = FullImageDataset(final_tensor[int(.8*len(final_tensor)):], data["labels"].tolist()[int(.8*len(final_tensor)):])
    dl_val = DataLoader(ds_test, 8)

    learning_rate = 0.001
    model = FeatureNet()
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    num_epochs = 10

    # blur = torchvision.transforms.GaussianBlur(101)
    last_improvement = 0
    best_val = float("inf")

    for epoch in tqdm(range(5)):
        running_loss = 0.0
        blur_loss = 0
        uncert_loss = 0
        few_loss = 0

        i = 0
        model.train()
        for data in tqdm(dl):
            # print(i)
            i += 1
            inputs = data
            inputs = inputs.to(device)

            optimizer.zero_grad()

            outputs, classes = model(inputs)

            # blur_classes = blur(classes)

            # # smoothness_loss = criterion(classes, blur_classes)*.005
            # up = 0#uncertianty_penalty(classes)
            # tp = 0# tv_regularization(classes)
            # fp = 0#few_channel_penalty(classes)
            # smoothness_loss = up + tp + fp
            # blur_loss += 0#tp.item()
            # uncert_loss += 0#up.item()
            # few_loss += 0#fp.item()
            # # smoothness_loss += torch.mean(ext_vals)
            # print(f"\t{inputs.shape}, {classes.shape}")
            # smoothness_loss = (Modified_Soft_Normalized_Cut_Loss(inputs.clone(), classes)).to(device)
            # #uc = uncertianty_penalty(classes).to(device)
            # #print(smoothness_loss.item(), uc.item())
            # #smoothness_loss += uc
            # smoothness_loss.backward()
            # optimizer.step()
            # optimizer.zero_grad()

            # blur_loss += smoothness_loss.item()
            # outputs, classes = model(inputs)

            acc_loss = criterion(outputs, inputs)
            loss = acc_loss  # + smoothness_loss * .01
            t_loss = loss  # + smoothness_loss
            acc_loss.backward()
            optimizer.step()
            running_loss += loss.item()
            # print(f"Batch results: {smoothness_loss.item()}, {loss.item()}, {running_loss}, {blur_loss}")



        print(f'Epoch [{epoch + 1}/{num_epochs}]')

        print(f'\t\tReconstruction Loss: {(running_loss) / len(dl)}')
        print(f'\t\tSegmentation Loss: {(blur_loss) / len(dl)}')

    model.eval()
    acc_loss = 0
    running_loss = 0
    for data in tqdm(dl_val):

        inputs = data
        inputs = inputs.to(device)


        outputs, classes = model(inputs)

        # blur_classes = blur(classes)

        # # smoothness_loss = criterion(classes, blur_classes)*.005
        # up = 0#uncertianty_penalty(classes)
        # tp = 0# tv_regularization(classes)
        # fp = 0#few_channel_penalty(classes)
        # smoothness_loss = up + tp + fp
        # blur_loss += 0#tp.item()
        # uncert_loss += 0#up.item()
        # few_loss += 0#fp.item()
        # # smoothness_loss += torch.mean(ext_vals)
        # print(f"\t{inputs.shape}, {classes.shape}")
        # smoothness_loss = (Modified_Soft_Normalized_Cut_Loss(inputs.clone(), classes)).to(device)
        # #uc = uncertianty_penalty(classes).to(device)
        # #print(smoothness_loss.item(), uc.item())
        # #smoothness_loss += uc
        # smoothness_loss.backward()
        # optimizer.step()
        # optimizer.zero_grad()

        # blur_loss += smoothness_loss.item()
        # outputs, classes = model(inputs)

        acc_loss = criterion(outputs, inputs)
        loss = acc_loss  # + smoothness_loss * .01
        running_loss += loss.item()



    print(f'\t\tReconstruction Loss: {(running_loss) / len(dl_val)}')

    torch.save(model.state_dict(), "model.pkl")


run()