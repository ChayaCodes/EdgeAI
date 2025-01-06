import datasets
import matplotlib.pyplot as plt

mnist = datasets.load_dataset("mnist")

def display_image(image, title, cmap=None):
    plt.imshow(image, cmap=cmap)
    plt.title(title)
    plt.axis('off')
    plt.show()

# MNIST Sample
mnist_sample = mnist['train'][0]
display_image(mnist_sample['image'], f"MNIST Label: {mnist_sample['label']}", cmap='gray')