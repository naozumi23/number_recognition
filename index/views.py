from django.shortcuts import render
from django.views.generic import TemplateView
import base64
from PIL import Image
from io import BytesIO
from sklearn.svm import LinearSVC
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn import metrics
from sklearn import model_selection
import time
import tensorflow as tf
from keras.datasets import mnist

# Create your views here.
class IndexView(TemplateView):
    template_name = "index.html"

    # get request
    def get(self, request):
        #learning_svc()
        #learning_cnn()
        return render(request, self.template_name)

    # post request
    def post(self, request):

        # data preprocessing
        img_test = np.empty((0, 784))
        image = request.POST['newImg']
        img = base64_to_pil(image)
        img = img.convert('L')
        img = img.resize((224, 224))
        img_data256 = np.array([])
        for y in range(28):
            for x in range(28):
                crop = np.asarray(img.crop(
                    (x * 8, y * 8, x * 8 + 8, y * 8 + 8)))
                bright = 255 - crop.mean() ** 2 / 255
                img_data256 = np.append(img_data256, bright)
        img_test = np.r_[img_test, img_data256.astype(np.uint8).reshape(1, -1)]
        img_test = (255 - img_test)
        img_test = img_test * 255 / max(max(img_test))
        img = np.reshape(img_test, (28, 28))
        img = np.array(img, dtype='uint8')
        plt.imshow(img)
        plt.show()

        # SVC predict
        svc_model = pickle.load(open('index/ml/LinearSVC_model.sav', 'rb'))
        svc_answer = svc_model.predict(img_test)[0]

        # CNN predict
        cnn_model = tf.keras.models.load_model('index/ml/cnn_model')
        cnn_answer = np.argmax(cnn_model.predict(img.reshape(1, 28, 28)))

        # return
        params = {"svc_answer": svc_answer, "cnn_answer": cnn_answer}
        return render(request, self.template_name, params)

# Convert base64 image data to PIL image
def base64_to_pil(img_str):
    if "base64," in img_str:
        img_str = img_str.split(",")[1]
    img_raw = base64.b64decode(img_str)
    img = Image.open(BytesIO(img_raw))
    return img

# learning svc
def learning_svc():
    digits = fetch_openml(name='mnist_784')
    train_size = 1000
    test_size = 100
    x, y = digits["data"], digits["target"]
    x = x / 256 * 16
    data_train, data_test, label_train, label_test = model_selection.train_test_split(x,
                                                                                      y,
                                                                                      test_size=test_size,
                                                                                      train_size=train_size)
    start_time = time.time()
    classifier = LinearSVC(penalty='l2', loss='squared_hinge', dual=False, C=1.0, class_weight='balanced',
                           random_state=0, multi_class='ovr')
    classifier.fit(data_train, label_train)
    end_time = time.time()
    print("learning finish:" + str(end_time - start_time))
    pickle.dump(classifier, open('index/ml/LinearSVC_model.sav', 'wb'))

# learning cnn
def learning_cnn():
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    model.fit(train_images, train_labels, epochs=10,
              validation_data=(test_images, test_labels))
    model.save('./index/ml/cnn_model')
