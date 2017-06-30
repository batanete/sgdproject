# import the necessary packages
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
import os
import pickle

from time import time



PATH='food-101'
N_JOBS=-1

def run(sc=None):
    if sc is not None:
        logger = sc._jvm.org.apache.log4j
        logger.LogManager.getLogger("org"). setLevel( logger.Level.ERROR )
        logger.LogManager.getLogger("akka").setLevel( logger.Level.ERROR )
    def image_to_feature_vector(image):
        # resize the image to a fixed size, then flatten the image into
        # a list of raw pixel intensities
        return cv2.resize(image, (32,32)).flatten()

    def extract_color_histogram(image):
        # extract a 3D color histogram from the HSV color space using
        # the supplied number of `bins` per channel
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, (8,8,8),
        	[0, 180, 0, 256, 0, 256])

        # handle normalizing the histogram if we are using OpenCV 2.4.X
        if imutils.is_cv2():
        	hist = cv2.normalize(hist)

        # otherwise, perform "in place" normalization in OpenCV 3 (I
        # personally hate the way this is done
        else:
        	cv2.normalize(hist, hist)

        # return the flattened histogram as the feature vector
        return hist.flatten()

    def do_both(image):
        return image_to_feature_vector(image),extract_color_histogram(image)

    # construct the argument parse and parse the arguments
    """ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", required=True,
    	help="path to input dataset")
    ap.add_argument("-k", "--neighbors", type=int, default=1,
    	help="# of nearest neighbors for classification")
    ap.add_argument("-j", "--jobs", type=int, default=-1,
    	help="# of jobs for k-NN distance (-1 uses all available cores)")
    args = vars(ap.parse_args())
    """


    # grab the list of images that we'll be describing
    print("[INFO] describing images...")
    imagePaths = list(paths.list_images(PATH))


    # initialize the raw pixel intensities matrix, the features matrix,
    # and labels list

    rawImages = []
    features = []
    labels = []
    images=[]
    start=time()
    t=0
    # loop over the input images
    for (i, imagePath) in enumerate(imagePaths):

        # load the image and extract the class label (assuming that our
        # path as the format: /path/to/dataset/{class}.{image_num}.jpg
        image = cv2.imread(imagePath)
        label = imagePath.split(os.path.sep)[-2]
        images.append(image)

        # extract raw pixel intensity "features", followed by a color
        # histogram to characterize the color distribution of the pixels
        # in the image
        pixels = image_to_feature_vector(image)
        #pixels=image
        hist = extract_color_histogram(image)

        # update the raw images, features, and labels matricies,
        # respectively
        labels.append(label)


        # show an update every 1,000 images
        if i > 0 and (i % 90 == 0 or i==100999):
            start=time()
            if sc is not None:
                images=sc.parallelize(images)
                images.cache()
                images.map(lambda im: do_both(im))
                res=images.reduce(lambda im1,im2: [im1]+[im2])
                images.unpersist()
            else:
                for im in images:
                    res=do_both(im)

            images=[]

            #rawImages+=res[0]
            #features+=res[1]


            t+=time()-start
        if i > 0 and i % 1000 == 0:
            print("[INFO] processed {}/{}.time:{}".format(i, len(imagePaths),t))

    print("[MEASURES] Image load time:",t)

    """
    with open('features.pickle', 'wb') as f:
        pickle.dump(features, f)

    with open('labels.pickle', 'wb') as f:
        pickle.dump(labels, f)
    print('saved to file')


    #read features and labels from file
    with open('labels.pickle', 'rb') as f:
        labels = pickle.load(f)
    with open('features.pickle', 'rb') as f:
        features = pickle.load(f)
    print("read features and labels from file")

    # show some information on the memory consumed by the raw images
    # matrix and features matrix
    #rawImages = np.array(rawImages)
    features = np.array(features)
    labels = np.array(labels)
    #print("[INFO] pixels matrix: {:.2f}MB".format(
    #    rawImages.nbytes / (1024 * 1000.0)))
    print("[INFO] features matrix: {:.2f}MB".format(
        features.nbytes / (1024 * 1000.0)))

    # partition the data into training and testing splits, using 75%
    # of the data for training and the remaining 25% for testing
    #(trainRI, testRI, trainRL, testRL) = train_test_split(
    #	rawImages, labels, test_size=0.25, random_state=42)
    #(trainFeat, testFeat, trainLabels, testLabels) = train_test_split(
    #	features, labels, test_size=0.25, random_state=42)

    # train and evaluate a k-NN classifer on the raw pixel intensities

    print("[INFO] evaluating raw pixel accuracy...")
    model = KNeighborsClassifier(n_neighbors=args["neighbors"],
        n_jobs=args["jobs"])
    model.fit(trainRI, trainRL)
    acc = model.score(testRI, testRL)
    print("[INFO] raw pixel accuracy: {:.2f}%".format(acc * 100))

    # train and evaluate a k-NN classifer on the histogram
    # representations




    (trainFeat, testFeat, trainLabels, testLabels) = train_test_split(
    	features, labels, test_size=0.25, random_state=42)

    print("[INFO] evaluating histogram accuracy...")
    for k in range(1000,1001):
        start=time()
        model = KNeighborsClassifier(k,
            n_jobs=N_JOBS,weights='distance',p=2)
        model.fit(trainFeat, trainLabels)
        acc = model.score(testFeat, testLabels)
        s1="[INFO] histogram accuracy for k="+str(k)+": {:.2f}%".format(acc * 100)
        s2="[INFO] Time taken for k="+str(k)+":",time()-start

        return s1,s2

        #print(s1)
        #print()
    """
#run(None)
