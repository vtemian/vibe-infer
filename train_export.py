import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import os

model = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
loader = DataLoader(train_data, batch_size=64, shuffle=True)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(3):
    total_loss = 0
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.view(-1, 784)
        output = model(images)
        loss = loss_fn(output, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += (output.argmax(1) == labels).sum().item()
        total += labels.size(0)
    print(f"Epoch {epoch+1}: loss={total_loss/len(loader):.4f} acc={correct/total:.4f}")

os.makedirs('weights', exist_ok=True)

# Export: W1 (784x128), bias1 (128), W2 (128x10), bias2 (10)
# nn.Linear stores weights as (out_features, in_features), so W1 is (128, 784)
# We transpose to (784, 128) for row-major matmul: input[1x784] @ W1[784x128]
w1 = model[0].weight.data.T.numpy().astype(np.float32)
b1 = model[0].bias.data.numpy().astype(np.float32)
w2 = model[2].weight.data.T.numpy().astype(np.float32)
b2 = model[2].bias.data.numpy().astype(np.float32)

w1.tofile('weights/w1.bin')
b1.tofile('weights/b1.bin')
w2.tofile('weights/w2.bin')
b2.tofile('weights/b2.bin')

print(f"\nExported weights:")
print(f"  w1.bin: {w1.shape} ({w1.nbytes} bytes)")
print(f"  b1.bin: {b1.shape} ({b1.nbytes} bytes)")
print(f"  w2.bin: {w2.shape} ({w2.nbytes} bytes)")
print(f"  b2.bin: {b2.shape} ({b2.nbytes} bytes)")

# Also export a few test images for verification
test_data = datasets.MNIST('./data', train=False, download=True, transform=transform)
test_images = []
test_labels = []
for i in range(10):
    img, label = test_data[i]
    test_images.append(img.view(784).numpy())
    test_labels.append(label)

np.array(test_images, dtype=np.float32).tofile('weights/test_images.bin')
np.array(test_labels, dtype=np.int32).tofile('weights/test_labels.bin')
print(f"  test_images.bin: 10 images ({10*784*4} bytes)")
print(f"  test_labels.bin: {test_labels}")
