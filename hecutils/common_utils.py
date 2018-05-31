from shutil import copyfile, rmtree
import os, glob
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_resnet_v2 import preprocess_input

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
		        rotation_range=30,
		        width_shift_range=0.2,
		        height_shift_range=0.2,
		        shear_range=0.2,
		        zoom_range=0.2,
		        horizontal_flip=True)

def get_data_generator(dataDir, config):
	image_augmenter = getImageDataGenerator()
	return image_augmenter.flow_from_directory(
            dataDir,  
            target_size=(config['img_height'], config['img_width']),  
            batch_size=config['batch_size'],
            class_mode='categorical')  

def get_metrics(useF1Score):
	if useF1Score:
		metrics=['accuracy', pt.f1, pt.recall, pt.precision]
	else:
		metrics=['accuracy']
	return metrics