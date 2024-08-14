import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data.dataloader import default_collate
from tqdm import tqdm
import cv2
import numpy as np
from PIL import Image


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # np.random.seed(seed)
    # random.seed(seed)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv = nn.Conv2d(1, 32, 3)
        self.dropout = nn.Dropout2d(0.25)
        self.fc = nn.Linear(5408, 10)  # 10 classes

    def forward(self, x):
        x = self.conv(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        output = F.log_softmax(x, dim=1)  # log prob for numerical stability
        return output


def train(model, train_loader, loss_fn, optimizer, epochs, log_interval, device):
    model.to(device)
    model.train()
    for epoch in range(1, epochs + 1):
        for data, target in tqdm(train_loader):
            # Send data to device
            data, target = data.to(device), target.to(device)

            # 1. Forward propagation
            output = model(data)

            # 2. Calculate loss
            loss = loss_fn(output, target)
            # loss = F.nll_loss(output, target)

            # 3. Optimizer zero grad
            optimizer.zero_grad()

            # 4. Back propagation
            loss.backward()

            # 5. Parameter update
            optimizer.step()

            # Log training info
            # if batch_idx % log_interval == 0:
            #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #         epoch, batch_idx * len(data), len(train_loader.dataset),
            #                100. * batch_idx / len(train_loader), loss.item()))


def test(model, test_loader, loss_fn, device):
    model.to(device)
    model.eval()
    test_loss = 0
    correct = 0
    with torch.inference_mode():  # disable gradient calculation for efficiency
        for data, target in test_loader:
            # Send data to device
            data, target = data.to(device), target.to(device)

            # Prediction
            output = model(data)

            # Compute loss & accuracy
            test_loss += loss_fn(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()  # how many predictions in this batch are correct

    test_loss /= len(test_loader.dataset)

    # Log testing info
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def webcam_testing(model, transform, device):
    model.to(device)
    # Open the webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()


    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Convert the image to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Resize the image to 28x28 pixels
        gray = cv2.resize(gray, (28, 28))

        # Convert the image to PIL format
        img = Image.fromarray(gray)

        # Apply the transform to the image
        img = transform(img)

        # Add an extra dimension to the image
        img = img.unsqueeze(0)

        # Make a prediction
        img = img.to(device)
        output = model(img)
        _, predicted = torch.max(output.data, 1)

        # Display the prediction on the frame
        cv2.putText(frame, 'Predicted: %d' % predicted.item(), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                    cv2.LINE_AA)

        # Display the resulting frame
        cv2.imshow('frame', frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
    


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Training settings
    # torch.manual_seed(42)
    setup_seed(42)
    BATCH_SIZE = 64
    EPOCHS = 2
    LOG_INTERVAL = 10

    # Define image transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    # Load dataset
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Create network, loss_function, optimizer
    model = CNN()
    loss_fn = F.nll_loss
    optimizer = optim.Adam(model.parameters())

    # Test
    print("\nBefore training:")
    test(model, test_loader, loss_fn, device)

    # Train
    train(model, train_loader, loss_fn, optimizer, EPOCHS, LOG_INTERVAL, device)

    # Save and load model (for reference in case you are separating train and test files)
    torch.save(model.state_dict(), "mnist_cnn.pt")
    model = CNN()
    model.load_state_dict(torch.load("mnist_cnn.pt"))

    # Test
    print("After training:")
    test(model, test_loader, loss_fn, device)

    webcam_testing(model, transform, device)


if __name__ == '__main__':
    main()
