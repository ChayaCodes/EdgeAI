import datasets
import matplotlib.pyplot as plt

cifar10 = datasets.load_dataset("cifar10")

def display_image(image, title, cmap=None):
    plt.imshow(image, cmap=cmap)
    plt.title(title)
    plt.axis('off')
    plt.show()


cifar_sample = cifar10['train'][0]
cifar10_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                  'dog', 'frog', 'horse', 'ship', 'truck']
display_image(
    cifar_sample['img'],
    f"CIFAR-10 Label: {cifar10_labels[cifar_sample['label']]}")
