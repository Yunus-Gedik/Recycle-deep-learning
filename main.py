from fastai.vision.all import *
import zipfile as zf
import random
import timeit

import cv2
import os
import pathlib
from scipy import stats


## splits indices for a folder into train, and test indices with random sampling
## input: folder path,train ratio, random seed1, random seed 2
## output: train and test indices
def split_indices(folder, _train, _test, seed1, seed2):
    n = len(os.listdir(folder))
    full_set = list(range(1, n + 1))

    ## train indices
    random.seed(seed1)
    train = random.sample(list(range(1, n + 1)), int(_train * n))

    ## temp
    remain = list(set(full_set) - set(train))

    # Test indices
    random.seed(seed2)
    test = random.sample(remain, int(_test * len(full_set)))

    return (train, test)


## gets file names for a particular type of trash, given indices
## input: waste category and indices
## output: file names
def get_names(waste_type, indices):
    file_names = [waste_type + str(i) + ".jpg" for i in indices]
    return (file_names)


def _basename(path):
    # A basename() variant which first strips the trailing slash, if present.
    # Thus we always get the last component of the path, even for directories.
    sep = os.path.sep + (os.path.altsep or '')
    return os.path.basename(path.rstrip(sep))


## moves group of source files to another folder
## input: list of source files and destination folder
## no output
def move_files(source_files, destination_folder):
    for file in source_files:
        real_dst = os.path.join(destination_folder, _basename(file))
        if os.path.exists(real_dst):
            continue
        shutil.move(file, destination_folder)


def splitter(waste_types, train, test, seed1, seed2):
    ## move files to destination folders for each waste type
    for waste_type in waste_types:
        source_folder = os.path.join('dataset-resized', waste_type)
        train_ind, test_ind = split_indices(source_folder, train, test, seed1, seed2)
        ## move source files to train
        train_names = get_names(waste_type, train_ind)
        train_source_files = [os.path.join(source_folder, name) for name in train_names]
        move_files(train_source_files, "train")

        ## move source files to test
        test_names = get_names(waste_type, test_ind)
        test_source_files = [os.path.join(source_folder, name) for name in test_names]
        ## I use data/test here because the images can be mixed up
        move_files(test_source_files, "test")


def yer(filename):
    if filename.name[0:2] == "ca":  # cardboard
        return "cardboard"
    if filename.name[0:2] == "gl":  # glass
        return "glass"
    if filename.name[0:2] == "me":  # metal
        return "metal"
    if filename.name[0:2] == "pa":  # paper
        return "paper"
    if filename.name[0:2] == "pl":  # plastic
        return "plastic"
    if filename.name[0:2] == "tr":  # trash
        return "trash"


def prepare(data_zip, train_ratio, test_ratio):
    # Delete previous test train folders

    if os.path.exists('test'):
        path = os.path.join(os.getcwd(), "test")
        shutil.rmtree(path)
    if os.path.exists('train'):
        path = os.path.join(os.getcwd(), "train")
        shutil.rmtree(path)

    # Open .zip
    files = zf.ZipFile(data_zip, 'r')
    files.extractall()
    files.close()

    waste_types = ['cardboard', 'glass', 'metal', 'paper', 'plastic']

    if not os.path.exists('test'):
        os.makedirs('test')
    if not os.path.exists('train'):
        os.makedirs('train')

    # Distribute dataset into train test folders.
    splitter(waste_types, train_ratio, test_ratio, random.randint(0, 10), random.randint(0, 10))

    # Delete no longer needed file
    path = os.path.join(os.getcwd(), "dataset-resized")
    shutil.rmtree(path)


def find_appropriate_lr(model:Learner, lr_diff:int = 15, loss_threshold:float = .05, adjust_value:float = 1, plot:bool = False) -> float:
    #Run the Learning Rate Finder
    model.lr_find()

    #Get loss values and their corresponding gradients, and get lr values
    losses = np.array(model.recorder.losses)
    assert(lr_diff < len(losses))
    loss_grad = np.gradient(losses)
    lrs = model.recorder.lrs

    #Search for index in gradients where loss is lowest before the loss spike
    #Initialize right and left idx using the lr_diff as a spacing unit
    #Set the local min lr as -1 to signify if threshold is too low
    r_idx = -1
    l_idx = r_idx - lr_diff
    while (l_idx >= -len(losses)) and (abs(loss_grad[r_idx] - loss_grad[l_idx]) > loss_threshold):
        local_min_lr = lrs[l_idx]
        r_idx -= 1
        l_idx -= 1

    lr_to_use = local_min_lr * adjust_value

    if plot:
        # plots the gradients of the losses in respect to the learning rate change
        plt.plot(loss_grad)
        plt.plot(len(losses)+l_idx, loss_grad[l_idx],markersize=10,marker='o',color='red')
        plt.ylabel("Loss")
        plt.xlabel("Index of LRs")
        plt.show()

        plt.plot(np.log10(lrs), losses)
        plt.ylabel("Loss")
        plt.xlabel("Log 10 Transform of Learning Rate")
        loss_coord = np.interp(np.log10(lr_to_use), np.log10(lrs), losses)
        plt.plot(np.log10(lr_to_use), loss_coord, markersize=10,marker='o',color='red')
        plt.show()

    return lr_to_use


def video_to_image(video_path,target_folder,freq):
    # Read the video from specified path
    cam = cv2.VideoCapture(video_path)

    try:
        # creating a folder named data
        if not os.path.exists(target_folder):
            os.makedirs(target_folder)

    # if not created then raise error
    except OSError:
        print('Error: Creating directory of data')
    # frame
    currentframe = 0
    frame_counter = 0

    while(True):
        # reading from frame
        ret,frame = cam.read()
        if ret:
            if frame_counter % freq == 0:
                # if video is still left continue creating images
                name = './'+target_folder+'/frame' + str(currentframe) + '.jpg'
                # print('Creating...' + name)

                # writing the extracted images
                cv2.imwrite(name, frame)

                # increasing counter so that it will
                # show how many frames are created
                currentframe += freq
        else:
            break
        frame_counter += 1

    # Release all space and windows once done
    cam.release()
    cv2.destroyAllWindows()


def read_model(model_file):
    temp = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath
    learn = load_learner(model_file)
    return learn


def predict_folder(folder_name,model,delete=True,prnt=False):
    temp = os.path.join(os.getcwd(), folder_name)
    test_files = os.listdir(temp)
    os.chdir(temp)

    predictions = []

    corrects = 0
    all = 0

    time_passed = 0.0
    max_time = 0.0

    for test_f in test_files:
        start_time = timeit.default_timer()
        predicted = model.predict(test_f)
        predictions.append(predicted[0])
        time_passed += timeit.default_timer() - start_time
        if timeit.default_timer() - start_time > max_time:
            max_time = timeit.default_timer() - start_time
        if prnt:
            print("%s %s" % (test_f, predicted[0]))
        if test_f[0:2] == predicted[0][0:2]:
            corrects += 1
        all += 1

    if prnt:
        print("Accuracy is %f" % (corrects / all))
        print("Average prediction time is %f s" % (time_passed / all))
        print("Longest prediction time is %f s" % max_time)

    os.chdir("..")

    if delete:
        path = os.path.join(os.getcwd(), folder_name)
        shutil.rmtree(path)
    return folder_name,stats.mode(predictions)[0][0]



# Load optimized model
learn = read_model("2.pkl")

# Predict test data
prepare("dataset-resized.zip", train_ratio=0.9, test_ratio=0.1)
print(predict_folder("test",learn,delete=False,prnt=True))

# Predict hand-held created images and videos
for item in os.listdir("videom"):
    video_to_image("videom/" + item, item,20)
    print(predict_folder(item,learn,True,False))

predict_folder("ben",learn,delete=False,prnt=True)

for item in os.listdir("videom_2"):
    video_to_image("videom_2/" + item, item,20)
    print(predict_folder(item,learn,True,False))







# Model creation and export

"""
prepare("dataset-resized.zip", train_ratio=0.9, test_ratio=0.1)

# blocks = I am working on images, and classifying images.
# get_items = I am gonna include image files into this data block.
# get_y = How to classify images? In our example, by their names.
# splitter = Splits data into validate and train sets.
# item_tfms = Resize images according to parameters.
# batch_tfms = includes augmented versions of images into model.

fields = DataBlock(blocks=(ImageBlock, CategoryBlock),
                   get_items=get_image_files,
                   get_y=yer,
                   splitter=RandomSplitter(valid_pct=0.2, seed=6),
                   item_tfms=RandomResizedCrop(224, min_scale=0.5),
                   batch_tfms=aug_transforms(do_flip=True, flip_vert=False)
                   )


dls = fields.dataloaders(os.path.join(Path(os.getcwd()), "train"), num_workers=0, bs=16)

dls.show_batch()

# Benefits from already created resnet34 model in image classifier
learn = cnn_learner(dls, resnet34, metrics=error_rate)

# Uses transfer learning to fine tune pretrained model
learn.fine_tune(6)

learn.show_results()

# Exporting and saving model
# learn.export()
"""






pass
