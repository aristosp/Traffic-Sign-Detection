# TODO: add augments

# V9: Added Confusion matrix and other metrics
# V8: Added gaussian blur
# V7: Added class weights
# V6: Added random rotation
# V5: Added leaky relu instead of relu
# V4: Added prediction visualizations + code optimizations in utils.py and here
# V3: Added custom model
# V2: Added test set usage
# V1: Added early stopping, function use
import pandas as pd
from torchvision.transforms import v2
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from utils import *
from torchsummary import summary

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Set transforms
trn_tfms = v2.Compose([
    v2.ToPILImage(), v2.Resize(32),
    v2.CenterCrop(32),
    v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

val_tfms = v2.Compose([
    v2.ToPILImage(), v2.Resize(32),
    v2.CenterCrop(32),
    v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

# Create gaussian blur augmentations for each image
augments = v2.Compose([
    v2.ColorJitter(brightness=0, contrast=1, saturation=0.3, hue=0),
    v2.ToPILImage(), v2.Resize(32),
    v2.CenterCrop(32),
    v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

augments2 = v2.Compose([
    v2.GaussianBlur(kernel_size=5, sigma=2),
    v2.ToPILImage(), v2.Resize(32),
    v2.CenterCrop(32),
    v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

# Read csv containing training data
train_csv = pd.read_csv('train.csv')
# Split training files and initialize dataloaders
train_files, val_files = train_test_split(train_csv, test_size=0.2, random_state=0, shuffle=True)
aug1 = TrafficSignDataset(train_files, augments)
aug2 = TrafficSignDataset(train_files, augments2)

train_ds = TrafficSignDataset(train_files, transform=trn_tfms)
val_ds = TrafficSignDataset(val_files, transform=val_tfms)
train_dataset = torch.utils.data.ConcatDataset([train_ds, aug1, aug2])

train_dl = DataLoader(train_dataset, 32, shuffle=True, collate_fn=train_ds.collate_fn)
validation_dl = DataLoader(val_ds, 32, shuffle=False, collate_fn=val_ds.collate_fn)
# Initialize classifier
# Class weights in case they are needed
# weights = torch.FloatTensor(list(class_weight_calc(train_csv, class_mode='sklearn'))).to(device)
loss_function = nn.CrossEntropyLoss()
model = TrafficSignClassifier(loss_fn=loss_function).to(device)
summary(model, input_size=(3, 32, 32))
# Hyper parameter setting
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
n_epochs = 10
early_stop = 2
best_epoch = 0
min_loss = torch.tensor(float('inf'))
# List to append metrics
train_loss = []
validation_loss = []
train_acc = []
validation_acc = []
# Training loop
for epoch in tqdm(range(n_epochs), desc='Epoch', unit='Epoch', position=0, leave=True):
    # Begin training mode, i.e. gradient changes
    model.train(True)
    avg_loss, avg_acc = train_per_epoch(train_dl, model, optimizer)
    # Save loss and accuracy per epoch for later visualizations
    train_loss.append(avg_loss)
    train_acc.append(avg_acc)
    # Begin evaluation mode, no gradient changes
    model.eval()
    val_loss, val_acc = evaluation(validation_dl, model, 'Validation')
    # Append to lists for later visualizations
    validation_loss.append(val_loss)
    validation_acc.append(val_acc)
    print('\n')
    info = f'''Epoch: {epoch + 1:02d}\tTrain Loss: {avg_loss:.4f}\tTrain Accuracy: {avg_acc:.3f}\t'''
    info += f'\nValidation Loss: {val_loss:.4f}\tValidation Accuracy: {val_acc:.3f}\n'
    print(info)
    # Implement early stopping
    if val_loss < min_loss:
        min_loss = val_loss
        best_epoch = epoch
        early_stopping(model, "best_model.pth", 'save')
    elif epoch - best_epoch + 1 > early_stop:
        print("Early stopping training at epoch %d" % best_epoch)
        early_stopping(model, "best_model.pth", 'restore')
        break  # terminate the training loop
# Restore best model (if early stopping occurred then the same model will be restored)
early_stopping(model, "best_model.pth", 'restore')
# Plot metrics
# plot_metrics(train_acc, validation_acc, train_loss, validation_loss)

# Evaluate on test set
test_df = pd.read_csv('GT-final_test.csv', sep=';')
test_df['Filename'] = 'Final_Test/Images/' + test_df['Filename']
# Drop some columns to make test dataframe similar (in length) to training csv
test_df = test_df.drop(['Width', 'Height', 'Roi.X1', 'Roi.Y1'], axis=1)
# Create the dataset and dataloader
test_ds = TrafficSignDataset(test_df, transform=val_tfms)
test_dl = DataLoader(test_ds, 256, shuffle=False, collate_fn=test_ds.collate_fn)
test_loss, test_acc = evaluation(test_dl, model, 'Test')
print("Test Accuracy : {:.4f}, Test Loss : {:.4f}".format(test_acc, test_loss))
signs = pd.read_csv('signnames.csv')
# Display random predictions on test data
display_predictions(model, signs, test_ds, val_tfms)
# Report on classification metrics and display confusion matrix
prediction_metrics(model, test_dl, signs['SignName'], confusion=False)
