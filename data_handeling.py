from PIL import Image
import glob
import numpy as np
import os
#the filesize
FILESIZE = 256
#expand2square will padd the image with black pixels until it is square
def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result

def main():
    #dataset = []
    path = "PokemonData"
    #get a list of all the folder names (f.path for path) in the directory specified in path
    list_subfolders_with_paths = [f.name for f in os.scandir(path) if f.is_dir()]
    #for each folder we go through the contents and get all the images which are squared, resized and normalized with the folder name as label
    for folder in list_subfolders_with_paths:
        for filename in glob.iglob( f"PokemonData/{folder}"+ '**/*.jpg', recursive=True):
            im = Image.open(filename)
            im_square = expand2square(im,(0,0,0))
            im_resized = im_square.resize((FILESIZE, FILESIZE))
            im_array = np.array(im_resized)/255.0
            im_labeled = [im_array,folder]
            #add to dataset
    #save dataset
if __name__ == "__main__":
    main()
