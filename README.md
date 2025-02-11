# DEEP-LEARNING-PROJECT

COMPANY: CODETECH IT SOLUTIONS

NAME: DEEKSHITH KUMAR

INTERN ID: CODHC14

DOMAIN: Data science

DURATION: 8 WEEKS

MENTOR: NEELA SANTHOSH

##DESCIPTION
This project involves implementing deep learning models using TensorFlow and PyTorch to classify images from two well-known datasets: CIFAR-10 and MNIST. The CIFAR-10 dataset consists of 60,000 color images (32x32 pixels) categorized into 10 classes, such as airplanes, birds, and trucks. The MNIST dataset, on the other hand, contains 70,000 grayscale images (28x28 pixels) of handwritten digits (0-9). The TensorFlow-based CIFAR-10 model is a Convolutional Neural Network (CNN) built using keras.Sequential(). It comprises three convolutional layers (Conv2D), max-pooling layers (MaxPooling2D), a fully connected (Dense) layer, and a final classification layer with softmax activation. The model is compiled using the Adam optimizer and trained with sparse categorical cross-entropy loss for 10 epochs. During training, real-time validation accuracy is tracked using the test dataset. Once trained, the model is evaluated, and a training accuracy curve is plotted to assess performance over epochs. Additionally, sample test images are visualized along with their predicted labels, with correct predictions shown in green and incorrect ones in red. The PyTorch-based model follows a similar approach but is applied to the MNIST dataset. The data is normalized and transformed into PyTorch tensors using torchvision.transforms. A custom CNN architecture is defined using nn.Module, containing two convolutional layers, ReLU activation, max pooling, and fully connected layers. The model is trained using CrossEntropyLoss and optimized using the Adam optimizer. Training is performed over five epochs, tracking both training loss and accuracy. Once trained, the model is evaluated on the test set, and a training loss curve is plotted. Finally, a batch of test images is visualized along with their predicted labels. The PyTorch-based model is structured using DataLoader objects, making it efficient for handling large datasets. Both models are designed to perform image classification tasks and are applicable in real-world scenarios like object recognition, medical imaging, security systems, and self-driving cars. TensorFlow’s high-level API (keras) makes it easier to build and train deep learning models, while PyTorch provides flexibility for research and experimentation. These deep learning implementations can be extended to more complex tasks such as fine-tuning pre-trained models (e.g., ResNet, VGG), image segmentation, or GAN-based image generation. Additionally, performance can be improved by adding dropout layers, increasing CNN depth, or applying data augmentation techniques. Furthermore, both models can be converted to mobile-compatible formats for real-time inference. Overall, this project serves as a strong foundation for building robust image classification models, demonstrating how CNN architectures effectively extract features from images to perform accurate classification tasks in practical applications.

##OUTPUT

![Image](https://github.com/user-attachments/assets/4cee478e-189c-4735-a430-33f4b5fbe0f5)

![Image](https://github.com/user-attachments/assets/2442763c-fec5-4156-a818-aee7c13fd135)

![Image](https://github.com/user-attachments/assets/3d2f3341-8c1a-4d19-841e-285c7b2ef818)

![Image](https://github.com/user-attachments/assets/d8954bef-ef08-4278-8a57-0120b423b27c)

![Image](https://github.com/user-attachments/assets/d5f75254-1f28-4469-af29-39c9e0fd99e0)
