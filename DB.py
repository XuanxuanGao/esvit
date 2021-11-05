# coding=utf-8

import pandas as pd
import os

class Database(object):
    """
    Generate DB_csv based on images in DB_dir.

    The organization of DB_dir:
    DB_dir
    |__image1_xxx.jpg
    |__image2_xx.jpg
    |__image3.jpg

    The DB_csv organizes the above information into a .csv file, then gives data & classes:
    img                                     cls
    fullpath_prefix/DB_dir/image1_xxx.jpg   image1
    fullpath_prefix/DB_dir/image2_xx.jpg   image2
    fullpath_prefix/DB_dir/image3.jpg       image3

    """
    def __init__(self, DB_dir, DB_csv):
      self.DB_dir = DB_dir
      self.DB_csv = DB_csv
      self._gen_csv()
      self.data = pd.read_csv(DB_csv)
      self.classes = set(self.data["cls"])

    def _gen_csv(self):
      if os.path.exists(self.DB_csv):
        print("%s already exists, exit..." % self.DB_csv)
        return
      with open(self.DB_csv, 'w', encoding='UTF-8') as f:
        f.write("img,cls")
        # for root, _, files in os.walk(self.DB_dir, topdown=False):
        for image in os.listdir(self.DB_dir):
          if not image.endswith('.jpg'):
              continue
          cls = image.split('.jpg')[0].split('_')[0]
          img = os.path.join(self.DB_dir, image)
          f.write("\n{},{}".format(img, cls))

    def __len__(self):
      return len(self.data)

    def get_class(self):
      return self.classes

    def get_data(self):
      return self.data


if __name__ == "__main__":
    cur_dir = os.getcwd() # CBIR/src/
    root_dir = os.path.split(cur_dir)[0] # CBIR/
    DB_dir = os.path.join(root_dir, 'image/zhixinlian/original')
    print("DB_dir: ", DB_dir)
    DB_csv = os.path.join(root_dir, 'image/zhixinlian/original.csv')
    print("DB_csv: ", DB_csv)
    db = Database(DB_dir, DB_csv)
    data = db.get_data()
    classes = db.get_class()
    print("DB length:", len(db))
    print("Num of classes:", len(classes))
