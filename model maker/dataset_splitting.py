import os, random, shutil
try:
    os.mkdir('dataset/train')
    os.mkdir('dataset/test')
except:
    print("train test directory already exits.")


image_paths = os.listdir('dataset/images')
random.shuffle(image_paths)


for i, image_path in enumerate(image_paths):
  if i < int(len(image_paths) * 0.8):
    try:
        shutil.copy(f'dataset/images/{image_path}', 'dataset/train')
        shutil.copy(f'dataset/annotations/{image_path.replace("jpg", "xml")}', 'dataset/train')
    except:
        shutil.copy(f'dataset/images/{image_path}', 'dataset/train')
        shutil.copy(f'dataset/annotations/{image_path.replace("JPG", "xml")}', 'dataset/train')
    
  else:
    try:
        shutil.copy(f'dataset/images/{image_path}', 'dataset/test')
        shutil.copy(f'dataset/annotations/{image_path.replace("jpg", "xml")}', 'dataset/test')
    except:
        shutil.copy(f'dataset/images/{image_path}', 'dataset/test')
        shutil.copy(f'dataset/annotations/{image_path.replace("JPG", "xml")}', 'dataset/test')