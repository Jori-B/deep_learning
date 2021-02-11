from PIL import Image
import glob
import numpy as np
#im = Image.open("abra.jpg")
im2 = Image.open("abra2.jpg")
#im.show()

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
im_new2 = expand2square(im2,(0,0,0))
#im_new2.show()
im_final = im_new2.resize((256, 256))
im_final = np.array(im_final)/255.0
im_label = [im_final,"abra"]
print(im_label[0][1])
print(im_label[1])
#im_final.show()
#im_new = expand2square(im, (0, 0, 0))
#im_new2 = expand2square(im2,(0,0,0))
#im_new.save('data/dst/astronaut_expand_square.jpg', quality=95)
#im_new2.show()
#im_new.show()
#count = 0
#for filename in glob.iglob( "archive/PokemonData/Abra"+ '**/*.jpg', recursive=True):
#    count +=1
#    print(count)
