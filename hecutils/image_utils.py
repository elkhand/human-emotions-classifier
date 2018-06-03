from shutil import copyfile, rmtree
import os, glob
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.imagenet_utils import preprocess_input
import numpy as np

def printXy(X,y):
    print("===============================")
    for x_i, y_i in zip(X, y):
        print(x_i, y_i)
    print("===============================")


def copy_imgs_into(src_dir, img_name_list, dst_dir):
	# Delete test images dir
	rmtree(dst_dir, ignore_errors=True)
	os.makedirs(dst_dir)
	for image_name in img_name_list:
		src = src_dir + "/" + image_name
		dst = dst_dir + "/" + image_name
		copyfile(src, dst)

def get_nb_files(directory):
    """Get number of files by searching directory recursively"""
    if not os.path.exists(directory):
        return 0
    cnt = 0
    for r, dirs, files in os.walk(directory):
        for dr in dirs:
            cnt += len(glob.glob(os.path.join(r, dr + "/*")))
    return cnt		

def getImageDataGenerator():
	# data prep
	return ImageDataGenerator(
							preprocessing_function=preprocess_input,
                            rotation_range=40,
                            width_shift_range=0.2,
                            height_shift_range=0.2,
                            shear_range=0.2,
                            zoom_range=0.2,
                            channel_shift_range=10,
                            horizontal_flip=True,
							fill_mode='nearest')

def get_data_generator(dataDir, config, isForTrain):
	if isForTrain:
		image_augmenter = getImageDataGenerator()
	else:
		image_augmenter = ImageDataGenerator(
				preprocessing_function=preprocess_input
			)
	return image_augmenter.flow_from_directory(
	            dataDir,  
	            target_size=(config['img_height'], config['img_width']), 
	            interpolation='bicubic', 
	            batch_size=config['batch_size'],
	            class_mode='categorical',
	            shuffle=isForTrain)  

def get_data_generator_for_test(dataDir, config):
	image_augmenter = ImageDataGenerator(
			preprocessing_function=preprocess_input
		)
	return image_augmenter.flow_from_directory(
	            dataDir,  
	            target_size=(config['img_height'], config['img_width']), 
	            #interpolation='bicubic', 
	            batch_size=config['batch_size'],
	            class_mode='categorical',  # only data, no labels
	            shuffle=False)  # keep data in same order as labels


def get_metrics(useF1Score):
	if useF1Score:
		metrics=['accuracy', pt.f1, pt.recall, pt.precision]
	else:
		metrics=['accuracy']
	return metrics

def conver_predictions_to_classes(predictions, class_to_index):
    predictions = np.argmax(predictions, axis=-1) #multiple categories
    index_to_class = dict((int(v),int(k)) for k,v in class_to_index.items()) #flip k,v
    predictions = [index_to_class[k] for k in predictions]
    return predictions

def get_label_map_from_train_generator(config):
    trainDir = input_images_classified + "/" + "train" + "/"
    isForTrain = True
    config['batch_size'] = batch_size
    train_batches = cm.get_data_generator(trainDir, config, isForTrain)  
    return train_batches.class_indices

def get_truth_labels_test_data(test_filenames):
    y_true = []
    for test_file_name in test_filenames:
        # 'negative/Ambulance 1.jpg'
        label, test_file_name = test_file_name.split("/")
        assert label == imageNameToLabel[test_file_name], \
                "test_file_name did not match: " + test_file_name + \
                "\t imageNameToLabel:" + imageNameToLabel[test_file_name]                                                    
        y_true.append(imageNameToLabel[test_file_name])
    return y_true	