from tqdm import tqdm
import torch
from torch.utils.data import Dataset
import numpy as np
import torch.nn as nn
from torch_snippets import read, randint
import matplotlib.pyplot as plt
from sklearn.utils import class_weight
import cv2
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class TrafficSignDataset(Dataset):
    """
    A custom dataset class, pertaining to the specifics of GTSRB dataset
    """
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, ix):
        row = self.df.iloc[ix].squeeze()
        file = row.iloc[0]
        int_label = row.iloc[3]
        img = read(file, 1)
        return img, int_label

    def choose(self):
        return self[randint(len(self))]

    def collate_fn(self, batch):
        images, classes = list(zip(*batch))
        if self.transform:
            images = [self.transform(img)[None] for img in images]
        classes = torch.tensor(classes).to(device)
        images = torch.cat(images).to(device)
        return images, classes


def convblock(ni, no, kernel_size, padding=1):
    """
    A function wrapping a common convolutional block:
    Dropout -> Conv2D -> ReLU -> Batch Normalization -> MaxPooling
    :param ni: Number of input channels
    :param no: Number of output channels
    :param kernel_size: Kernel dimensions
    :param padding: Padding dimensions, default equal to 1
    :return: A Sequential object containing the aforementioned lars
    """
    return nn.Sequential(
        nn.Conv2d(ni, no, kernel_size=kernel_size, padding=padding),
        nn.LeakyReLU(inplace=True),
        nn.BatchNorm2d(no),
        nn.MaxPool2d(2))


class TrafficSignClassifier(nn.Module):
    def __init__(self, loss_fn):
        super().__init__()
        self.model = nn.Sequential(
            convblock(3, 64, kernel_size=3),
            convblock(64, 128, kernel_size=3),
            nn.Dropout(0.4),
            convblock(128, 256, kernel_size=3),
            nn.Dropout(0.4),
            convblock(256, 128, kernel_size=3),
            nn.Flatten(),
            nn.Linear(512, 128),
            nn.Dropout(0.4),
            nn.LeakyReLU(inplace=True),
            nn.Linear(128, 43),
        )
        self.loss_fn = loss_fn

    def forward(self, x):
        return self.model(x)

    def compute_metrics(self, predictions, targets):
        loss = self.loss_fn(predictions, targets)
        acc = (torch.max(predictions, 1)[1] == targets).float().mean()
        return loss, acc


def class_weight_calc(train_df, class_mode='sklearn'):
    """
    A cumulative function that returns each classes' weights
    :param train_df: dataframe to extract train_labels from
    :param class_mode: Default = 'sklearn', computes class weights using sklearn utility
                       If class_mode == alphas, class weights are computed based on probabilities
                       If class_mode == custom, class weights are computed based on the inverse
                       of the probability of each class, then normalized by min weight
    :return: A dictionary containing the weights for each class
    """
    counts = train_df['5 Digit Int Label'].value_counts()
    labels = train_df['5 Digit Int Label']
    if class_mode == 'sklearn':
        class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(labels), y=labels)
        class_weights = dict(enumerate(class_weights.flatten(), 0))
        return class_weights
    elif class_mode == 'alphas':
        counts = counts.sort_index()
        alphas = []
        for i in range(len(counts)):
            alpha = counts[i] / len(labels)
            alphas.append(alpha)
        for i in range(len(alphas)):
            alphas[i] = 1 - alphas[i]
        mydict = {}
        for i in range(len(alphas)):
            mydict[i] = alphas[i]
        return mydict
    elif class_mode == 'custom':
        class_weights2 = {}
        weight = []
        for i in range(len(counts)):
            weight.append(len(counts) / (counts[i]))
        min_weights = min(weight)
        for i in range(len(counts)):
            class_weights2[i] = weight[i] / min_weights
        return class_weights2


def train_per_epoch(train_dl, model, optimizer):
    """
    Per epoch training process
    :param train_dl: training set dataloader
    :param model: model to use
    :param optimizer: optimizer to be used
    :return:
    train_loss: average loss per epoch
    """
    model.train()
    train_loss = []
    train_acc = []
    for batch_no, data in tqdm(enumerate(train_dl), total=len(train_dl), desc='Training',
                               unit='Batch', position=0, leave=True):
        # Unpack inputs and labels
        image, labels = data[0].to(device), data[1].to(device)
        # Zero gradients for each batch
        optimizer.zero_grad()
        # Predictions for this batch
        prediction = model(image)
        # Unpack and Compute loss
        loss, acc = model.compute_metrics(prediction, labels)
        # Append batch loss and accuracy
        train_loss.append(loss.item())
        train_acc.append(acc.item())
        # Back-propagate loss
        loss.backward()
        # Change weights
        optimizer.step()
    # Calculate mean loss and accuracy
    train_loss = np.mean(train_loss)
    train_acc = np.mean(train_acc)
    return train_loss, train_acc


def evaluation(dl, model, desc):
    """
    Function containing the evaluation process of the model.
    :param dl: Dataloader object
    :param model: Model to be used
    :param desc: Whether the function is used during validation or testing
    :return:
    avg_v_loss: loss during evaluation for one epoch
    avg_v_acc: accuracy during evaluation for one epoch
    """
    model.eval()
    with torch.no_grad():
        validation_loss = []
        per_batch_val_acc = []
        # Iterate through validation data
        for _, vdata in tqdm(enumerate(dl), total=len(dl), desc=desc,
                             unit='Batch', position=0, leave=True):
            img, label = vdata[0].to(device), vdata[1].to(device)
            prediction = model(img)
            # Compute val accuracy, val loss
            v_loss, v_acc = model.compute_metrics(prediction, label)
            per_batch_val_acc.append(v_acc.item())
            validation_loss.append(v_loss.item())
        # Compute the average of each metric for one epoch
        avg_v_loss = np.mean(validation_loss)
        avg_vacc = np.mean(per_batch_val_acc)
    return avg_v_loss, avg_vacc


def early_stopping(model, filename, mode):
    """
    Function implementing early stopping techniques, using the mode variable.
    :param model: model to save
    :param filename: path and name of the file
    :param mode: whether to save the model or restore the best model from a path
    :return: NULL
    """
    if mode == 'save':
        torch.save(model.state_dict(), filename)
    elif mode == 'restore':
        model.load_state_dict(torch.load(filename))
    else:
        print("Not valid mode")


def plot_metrics(train_acc, val_acc, train_loss, val_loss):
    """
    A simple function creating two plots to visualize accuracy
    and loss progression during training
    :param train_acc: list containing the training accuracy per epoch
    :param val_acc: list containing the validation accuracy per epoch
    :param train_loss: list containing the training loss per epoch
    :param val_loss: list containing the validation loss per epoch
    :return:
    """
    epochs = np.arange(1, len(train_acc) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax = ax.flat
    ax[0].plot(epochs, train_loss, 'bo', label='Training loss')
    ax[0].plot(epochs, val_loss, 'r', label='Validation loss')
    ax[0].legend()
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Loss')
    ax[0].set_title('Training and Validation loss over increasing epochs')
    ax[1].plot(epochs, train_acc, 'bo', label='Training Accuracy')
    ax[1].plot(epochs, val_acc, 'r', label='Validation Accuracy')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Accuracy')
    ax[1].set_title('Training and Validation accuracy over increasing epochs')
    ax[1].legend()
    plt.grid('off')
    plt.show()


def display_predictions(model, csvfile, dataset, transforms):
    """
    A function to display randomly selected predictions
    :param model: Model to be used
    :param csvfile: df containing class names
    :param dataset: dataset to extract predictions from
    :param transforms: transforms object to augment prediction images
    :return:
    """
    figure, axs = plt.subplots(3, 5, figsize=(10, 8), constrained_layout=True)
    for i in range(15):
        # Randomly select and return an image and its label
        img, label = dataset.choose()
        original_img = img.copy()
        # Augment the image to be an appropriate input to the model
        img = transforms(img).to(device)
        predictions = model(img[None])
        # Save predictions to cpu and find the prediction
        predictions = predictions.to('cpu').detach().numpy()
        predicted_label = np.argmax(predictions)
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        # Display in the titles the predictions and actual labels
        if i <= 4:
            # Find the sign name and display it
            predicted_sign = csvfile.loc[csvfile['ClassId'] == predicted_label]['SignName'].item()
            actual_sign = csvfile.loc[csvfile['ClassId'] == label]['SignName'].item()
            axs[0, i].imshow(original_img)
            axs[0, i].set_title('Predicted Sign: {}'.format(predicted_sign),
                                color=("green" if predicted_sign == actual_sign else "red"))
            axs[0, i].set_xlabel('Actual Sign: {}'.format(actual_sign))
        elif i <= 9:
            predicted_sign = csvfile.loc[csvfile['ClassId'] == predicted_label]['SignName'].item()
            actual_sign = csvfile.loc[csvfile['ClassId'] == label]['SignName'].item()
            axs[1, i - 5].imshow(original_img)
            axs[1, i - 5].set_title('Predicted Sign: {}'.format(predicted_sign),
                                    color=("green" if predicted_sign == actual_sign else "red"))
            axs[1, i - 5].set_xlabel('Actual Sign: {}'.format(actual_sign))
        else:
            predicted_sign = csvfile.loc[csvfile['ClassId'] == predicted_label]['SignName'].item()
            actual_sign = csvfile.loc[csvfile['ClassId'] == label]['SignName'].item()
            axs[2, i - 10].imshow(original_img)
            axs[2, i - 10].set_title('Predicted Sign: {}'.format(predicted_sign),
                                     color=("green" if predicted_sign == actual_sign else "red"))
            axs[2, i - 10].set_xlabel('Actual Sign: {}'.format(actual_sign))

    wm = plt.get_current_fig_manager()
    wm.window.state('zoomed')
    plt.show()


def prediction_metrics(model, test_dl, string_labels, confusion):
    """
    A function that uses test data to predict and provide the classification metrics
    and the confusion matrix
    :param model: Model to be used
    :param test_dl: Dataloader
    :param string_labels: dataframe column containing the labels in string format
    :param confusion: whether to plot confusion matrix
    :return:
    """
    y_pred = []
    y_true = []
    model.eval()
    with torch.no_grad():
        # Iterate through test data
        for _, test_data in tqdm(enumerate(test_dl), total=len(test_dl), desc='Prediction',
                                 unit='Batch', position=0, leave=True):
            img, label = test_data[0].to(device), test_data[1].to(device)
            prediction = model(img).cpu().numpy()
            # For each image, find the model prediction
            model_predictions = [np.argmax(arr) for arr in prediction]
            # Append to list the predictions and the ground truths
            y_pred.extend(model_predictions)
            y_true.extend(label.cpu().numpy())
    print(classification_report(y_pred=y_pred, y_true=y_true, target_names=list(string_labels), digits=3))
    # Check whether to plot confusion matrix
    if confusion:
        cf_matrix = confusion_matrix(y_true, y_pred)
        fig2, ax = plt.subplots(figsize=(12, 7))
        ConfusionMatrixDisplay(confusion_matrix=cf_matrix,
                               display_labels=string_labels).plot(xticks_rotation=-75, ax=ax)
        plt.title('Confusion Matrix')
        plt.show()
