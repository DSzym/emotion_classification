import cv2
import numpy as np
import os
import glob


class Face:
    def __init__(self, data, name):
        self.name = name
        self.data = data


def read_frames():
    folder = './faces'      # folder where images of faces are stored in three sub folders - happy, sad and surprised
    sub_folders = [name for name in os.listdir(folder) if os.path.isdir(os.path.join(folder, name))]
    print(sub_folders)

    # creating list of pairs - images and emotion (based on the name of sub folder)
    faces_list = []
    for name in sub_folders:
        print(name)
        try:
            faces_list += [Face(
                np.array(cv2.resize(cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2GRAY), (200, 200), cv2.INTER_AREA)),
                name) for file in glob.glob(f"faces/{name}/*.jpg")]

        except Exception as e:
            print(e)
            continue
    print(faces_list)
    faces = faces_list

    return faces


# based on scores of compatibility with other images, representative images are chosen and placed in separate folder
def read_photos_rep():
    folder = './representation'
    sub_folders = [name for name in os.listdir(folder) if os.path.isdir(os.path.join(folder, name))]
    print(sub_folders)

    faces_list = []
    for name in sub_folders:
        print(name)
        try:
            faces_list += [Face(
                np.array(cv2.resize(cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2GRAY), (200, 200), cv2.INTER_AREA)),
                name) for file in glob.glob(f"representation/{name}/*.jpg")]

        except Exception as e:
            print(e)
            continue
    faces = faces_list

    return faces


def split_data_to_train_test(data):
    train_data, test_data = train_test_split(data, train_size=0.75)

    train_x = np.array([face.data for face in train_data])
    train_y = [face.name for face in train_data]

    test_x = np.array([face.data for face in test_data])
    test_y = [face.name for face in test_data]

    return train_x, train_y, test_x, test_y


def get_data(data):         # get data from class
    x = np.array([face.data for face in data])
    y = [face.name for face in data]

    return x, y


def detect_face(image):     # detects frame where face is, returns that frame
    # using prepared cascade classifier from cv2 library
    face_detectors = ['haarcascade_frontalface_alt2.xml', 'haarcascade_frontalface_default.xml',
                      'haarcascade_frontalface_alt.xml']
    face = []
    for detector in face_detectors:
        face_cascade = cv2.CascadeClassifier(detector)
        # Detect faces
        face = face_cascade.detectMultiScale(image, 1.1, 5)

    if len(face) == 0:
        return []
    (x, y, w, h) = face[0]
    face = image[y-5:y + h+15, x:x + w]

    return face


def detect_face_keypoints(face_contour):
    sift = cv2.xfeatures2d.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(face_contour, None)
    sift_image = cv2.drawKeypoints(face_contour, keypoints, face_contour)       # useful for visualisation

    return keypoints, descriptors


def lower_half_of_picture(img):
    height = img.shape[0]
    height_cutoff = height // 2
    s1 = img[height_cutoff:, :]
    return s1


def adjust_classification_params(train_x, train_y, repr_x, repr_y):     # adjust classification parameters
    happy = list(np.zeros(2))  # [min limit, medium]
    sad = list(np.zeros(2))  # [min limit, medium]
    suprised = list(np.zeros(2))  # [min limit, medium]
    happy[0] = 1000
    sad[0] = 1000
    suprised[0] = 1000
    for i in range(len(train_x)):
        face1 = detect_face(train_x[i])     # get frame with face
        if len(face1) == 0:     # if classifier couldn't find face on photo then skip that image in analysis
            continue

        emotion_face1 = train_y[i]
        face1 = lower_half_of_picture(face1)        # get part of image that contains mouth
        keypoints_in_face1, descriptors1 = detect_face_keypoints(face1)

        # list of scores
        scores_happy = []
        scores_sad = []
        scores_suprised = []

        # loop to adjust classification parameters by comparing train and representation data and saving them in lists
        for k in range(len(repr_x)):
            face_data = repr_x[k]
            face_name = repr_y[k]
            face2 = detect_face(face_data)
            if len(face2) == 0:
                continue
            face2 = lower_half_of_picture(face2)
            keypoints_in_face2, descriptors2 = detect_face_keypoints(face2)

            bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
            matches = bf.match(descriptors1, descriptors2)
            matches = sorted(matches, key=lambda x: x.distance)

            representation_emotion = face_name

            if emotion_face1 == 'happy' and representation_emotion == 'happy':
                scores_happy.append(len(matches))

            elif emotion_face1 == 'sad' and representation_emotion == 'sad':
                scores_sad.append(len(matches))

            elif emotion_face1 == 'suprised' and representation_emotion == 'suprised':
                scores_suprised.append(len(matches))

        # parameters correction
        if emotion_face1 == 'happy':
            scores_happy.sort(reverse=True)
            temp = (scores_happy[0] + scores_happy[1]) / 2
            if happy[0] > temp:
                happy[0] = temp

            temp = sum(scores_happy) / len(scores_happy)
            happy[1] = (happy[1] + temp) / 2

        elif emotion_face1 == 'sad':
            scores_sad.sort(reverse=True)
            temp = (scores_sad[0] + scores_sad[1]) / 2
            if sad[0] > temp:
                sad[0] = temp

            temp = sum(scores_sad) / len(scores_sad)
            sad[1] = (sad[1] + temp) / 2

        elif emotion_face1 == 'suprised':
            scores_suprised.sort(reverse=True)
            temp = (scores_suprised[0] + scores_suprised[1]) / 2
            if suprised[0] > temp:
                suprised[0] = temp

            temp = sum(scores_suprised) / len(scores_suprised)
            suprised[1] = (suprised[1] + temp) / 2

    return happy, sad, suprised


# test how many of test images were classified correctly
def test_classificaton(test_x, test_y, repr_x, repr_y, happy, sad, suprised):
    test_happy_result = []
    test_sad_result = []
    test_suprised_result = []
    result = []

    for i in range(len(test_x)):
        test_photo = detect_face(test_x[i])
        test_emotion = test_y[i]
        if len(test_photo) == 0:
            continue

        test_photo = lower_half_of_picture(test_photo)
        keypoints_in_test_photo, descriptors_in_test_photo = detect_face_keypoints(test_photo)

        for k in range(len(repr_x)):
            # print("test     i = ", i, " k = ", k)
            face_data = repr_x[k]

            face_name = repr_y[k]
            face = detect_face(face_data)
            emotion_face_r = face_name
            if len(face) == 0:
                continue
            face = lower_half_of_picture(face)
            keypoints_in_face, descriptors = detect_face_keypoints(face)

            bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
            matches = bf.match(descriptors_in_test_photo, descriptors)
            matches = sorted(matches, key=lambda x: x.distance)

            if emotion_face_r == 'happy':
                test_happy_result.append(len(matches))
            elif emotion_face_r == 'sad':
                test_sad_result.append(len(matches))
            elif emotion_face_r == 'suprised':
                test_suprised_result.append(len(matches))
            else:
                print('error1')

        test_happy_result.sort(reverse=True)
        test_sad_result.sort(reverse=True)
        test_suprised_result.sort(reverse=True)

        happy_medium_score = sum(test_happy_result) / len(test_happy_result)
        happy_max_score = (test_happy_result[0] + test_happy_result[1]) / 2
        sad_medium_score = sum(test_sad_result) / len(test_sad_result)
        sad_max_score = (test_sad_result[0] + test_sad_result[1]) / 2
        suprised_medium_score = sum(test_suprised_result) / len(test_suprised_result)
        suprised_max_score = (test_suprised_result[0] + test_suprised_result[1]) / 2

        detected_emotion = []

        # First step: check if image's score correspond with criteria (minimal value and mean value)
        if happy_max_score > happy[0] and happy_medium_score >= happy[1]:
            detected_emotion.append('happy')

        if sad_max_score > sad[0] and sad_medium_score >= sad[1]:
            detected_emotion.append('sad')

        if suprised_max_score > suprised[0] and suprised_medium_score >= suprised[1]:
            detected_emotion.append('suprised')

        # Second step: check if mean value doesn't exclude image from being classified to emotion
        happy_param = happy_max_score - happy[0] + happy_medium_score - happy[1]
        sad_param = sad_max_score - sad[0] + sad_medium_score - sad[1]
        suprised_param = suprised_max_score - suprised[0] + suprised_medium_score - suprised[1]

        if happy_param > sad_param and happy_param > suprised_param:
            if 'happy' not in detected_emotion:
                detected_emotion.append('happy')

        if sad_param > happy_param and sad_param > suprised_param:
            if 'sad' not in detected_emotion:
                detected_emotion.append('sad')

        if suprised_param > happy_param and suprised_param > sad_param:
            if 'suprised' not in detected_emotion:
                detected_emotion.append('suprised')

        # check if result is correct, save it
        if test_emotion in detected_emotion:
            result.append(1)
        else:
            result.append(0)
        print("detected emotions: ", detected_emotion)
        test_happy_result = []
        test_sad_result = []
        test_suprised_result = []


def detect_emotion(image, repr_x, repr_y, happy, sad, suprised):
    image = detect_face(image)
    image_score_happy = []
    image_score_sad = []
    image_score_suprised = []

    if len(image) == 0:
        print("No face detected")
        return 0

    image = lower_half_of_picture(image)
    keypoints_in_image, descriptors_in_image = detect_face_keypoints(image)

    # compare to representation images
    for k in range(len(repr_x)):
        face_data = repr_x[k]
        face_name = repr_y[k]
        face = detect_face(face_data)
        emotion_face_r = face_name
        if len(face) == 0:
            continue
        face = lower_half_of_picture(face)
        keypoints_in_face, descriptors = detect_face_keypoints(face)

        bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
        matches = bf.match(descriptors_in_image, descriptors)
        matches = sorted(matches, key=lambda x: x.distance)

        if emotion_face_r == 'happy':
            image_score_happy.append(len(matches))
        elif emotion_face_r == 'sad':
            image_score_sad.append(len(matches))
        elif emotion_face_r == 'suprised':
            image_score_suprised.append(len(matches))
        else:
            print('error1')

    image_score_happy.sort(reverse=True)
    image_score_sad.sort(reverse=True)
    image_score_suprised.sort(reverse=True)

    happy_medium_score = sum(image_score_happy) / len(image_score_happy)
    happy_max_score = (image_score_happy[0] + image_score_happy[1]) / 2
    sad_medium_score = sum(image_score_sad) / len(image_score_sad)
    sad_max_score = (image_score_sad[0] + image_score_sad[1]) / 2
    suprised_medium_score = sum(image_score_suprised) / len(image_score_suprised)
    suprised_max_score = (image_score_suprised[0] + image_score_suprised[1]) / 2

    detected_emotion = []

    # First step: check if image's score correspond with criteria (minimal value and mean value)
    if happy_max_score >= happy[0] and happy_medium_score >= happy[1]:
        detected_emotion.append('happy')

    if sad_max_score > sad[0] and sad_medium_score >= sad[1]:
        detected_emotion.append('sad')

    if suprised_max_score > suprised[0] and suprised_medium_score >= suprised[1]:
        detected_emotion.append('suprised')

    # Second step: check if mean value doesn't exclude image from being classified to emotion
    happy_param = happy_max_score - happy[0] + happy_medium_score - happy[1]
    sad_param = sad_max_score - sad[0] + sad_medium_score - sad[1]
    suprised_param = suprised_max_score - suprised[0] + suprised_medium_score - suprised[1]

    if happy_param > sad_param and happy_param > suprised_param:
        if 'happy' not in detected_emotion:
            detected_emotion.append('happy')

    if sad_param > happy_param and sad_param > suprised_param:
        if 'sad' not in detected_emotion:
            detected_emotion.append('sad')

    if suprised_param > happy_param and suprised_param > sad_param:
        if 'suprised' not in detected_emotion:
            detected_emotion.append('suprised')

    return detected_emotion


faces_to_model = read_frames()
train_x, train_y, test_x, test_y = split_data_to_train_test(faces_to_model)

ascent = train_x[len(train_x)-1]

repr_photos = read_photos_rep()     #import representation image
repr_x, repr_y = get_data(repr_photos)

# lists, that keep classification params
happy, sad, suprised = adjust_classification_params(train_x, train_y, repr_x, repr_y)
test_classificaton(test_x, test_y, repr_x, repr_y, happy, sad, suprised)
