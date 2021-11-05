"""
CBIR pipeline:
1. get query embeddings and cached, get gallery embeddings and cached
2. add gallery embeddings into faiss, conduct recall using query embeddings
3. calculate top1-acc based on the recall result
"""

# -*- coding: utf-8 -*-

from __future__ import print_function

def get_image_info_single(DB_dir):
  """
  Generate single db & db.csv
  """
  import os
  from DB import Database

  print("dir: ", DB_dir)
  DB_csv = os.path.join(os.path.split(DB_dir)[0], os.path.split(DB_dir)[1] + '.csv')
  print("csv: ", DB_csv)
  db = Database(DB_dir, DB_csv)
  data = db.get_data()
  classes = db.get_class()
  print("DB length:", len(db))
  print("num of classes:", len(classes))

  return db

def get_image_info(testset = "augmented"):
  """
  Generate image/base_gallery.csv, image/tesetset/query.csv, image/tesetset/original.csv, which contains the images path and its class.

  the organization of image dir:
  image
  |
  |__base_gallery: large number of irrelative images
  |  |__image1.jpg
  |  |__...
  |  |__image50w.jpg
  |
  |__zhixinlian: testset1
  |  |
  |  |__query: augmented images
  |  |  |__image123_augment1.jpg
  |  |  |__image123_augment2.jpg
  |  |  |__...
  |  |  |__image25_augment3.jpg
  |  |  
  |  |__original: original images
  |     |__image123.jpg
  |     |__...
  |     |__image25.jpg
  |
  |__other tesetset

  """
  import os
  from DB import Database

  cur_dir = os.getcwd() # check cmd path: CBIR/src/
  root_dir = os.path.split(cur_dir)[0] # CBIR/

  # get base gallery image info
  DB_dir = os.path.join(root_dir, 'image/base_gallery')
  print("base gallery dir: ", DB_dir)
  DB_csv = os.path.join(root_dir, 'image/base_gallery.csv')
  print("base gallery csv: ", DB_csv)
  db_base = Database(DB_dir, DB_csv)
  data = db_base.get_data()
  classes = db_base.get_class()
  print("base gallery DB length:", len(db_base))
  print("base gallery num of classes:", len(classes))

  # get augmented query image info
  DB_dir = os.path.join(root_dir, 'image/%s/query' % testset)
  print("query image dir: ", DB_dir)
  DB_csv = os.path.join(root_dir, 'image/%s/query.csv' % testset) #
  print("query image csv: ", DB_csv)
  db_query = Database(DB_dir, DB_csv)
  data = db_query.get_data()
  classes = db_query.get_class()
  print("query image DB length:", len(db_query))
  print("query image num of classes:", len(classes))

  # get original image info
  DB_dir = os.path.join(root_dir, 'image/%s/original' % testset)
  print("original image dir: ", DB_dir)
  DB_csv = os.path.join(root_dir, 'image/%s/original.csv' % testset)
  print("original image csv: ", DB_csv)
  db_original = Database(DB_dir, DB_csv)
  data = db_original.get_data()
  classes = db_original.get_class()
  print("original image DB length:", len(db_original))
  print("original image num of classes:", len(classes))

  return db_base, db_query, db_original


def get_embedding(use_script, use_model, use_layer, use_db, use_cache):
  """
  Generate l2-normalized embedding of images in use_db, and save embedding into cache_file
  """
  if use_script == "resnet":
    from resnet import ResNetFeat
    samples = ResNetFeat(use_model, use_layer, use_cache).make_samples(use_db)
    print(samples[:5])
  elif use_script == "resnet_simsiam":
    from resnet_simsiam import ResNetFeat
    samples = ResNetFeat(use_model, use_layer, use_cache).make_samples(use_db)
  elif use_script == "resnet_simsiam_imagenet":
    from resnet_simsiam_imagenet import ResNetFeat
    samples = ResNetFeat(use_model, use_layer, use_cache).make_samples(use_db)
  else:
    print("use_script not implemented!")
    pass

def insert_faiss(base_cache_file, original_cache_file, testset = "zhixinlian"):
  """
  Read original embeddings & base_gallery embeddings in cache_file, and add into faiss
  ref: https://waltyou.github.io/Faiss-Introduce/
  """
  import faiss  # make faiss available
  import numpy as np
  from six.moves import cPickle
  import os

  # 1. prepare train data
  samples_base = []
  samples_original = []
  if os.path.exists(base_cache_file):
    base_cache_dir = os.path.dirname(base_cache_file)
    base_cache_file = os.path.basename(base_cache_file)
    # print(base_cache_dir)
    files = os.listdir(base_cache_dir)
    use_files = [x for x in files if x.startswith(base_cache_file)]
    print(use_files)
    for f in use_files:
      tmp = cPickle.load(open(os.path.join(base_cache_dir, f), "rb", True))
      samples_base += tmp
  # print(samples_base[:5])
  print("num of samples_base: ", len(samples_base))
  if os.path.exists(original_cache_file):
    samples_original = cPickle.load(open(original_cache_file, "rb", True))
  print("num of samples_original: ", len(samples_original))
  samples = samples_base + samples_original
  embedding = np.asarray([x['hist'] for x in samples])
  print("embedding.shape: ", embedding.shape)
  image_embedding = np.ascontiguousarray(embedding.astype(np.float32))
  d = image_embedding.shape[1] # dimension
  nb = image_embedding.shape[0]  # database size
  # print(image_embedding.shape)

  # 2. create index, add train data
  # index = faiss.IndexFlatL2(d)  # bruteforce L2-distance, add embeddings directly, no need to train
  index = faiss.IndexFlatIP(d) # InnerProduct
  print(index.is_trained)
  index.add(image_embedding)  # add vectors to the index
  print(index.ntotal)

  # 3. check search
  k = 4  # we want to see 4 nearest neighbors
  query_embedding = image_embedding[:5]
  D, I = index.search(query_embedding, k)  # sanity check
  print(I)  # N x k, the top-k neighbors' ID of each query embedding
  print(D)  # N x k, the top-k neighbors' Distance of each query embedding

  faiss.write_index(index, "../faiss_index/%s.index" % testset)

  # mapping ID to cls
  embedding_cls = [x['cls'] for x in samples]
  assert len(embedding_cls) == embedding.shape[0], "the dim of embedding and cls is not equal"
  index2cls = {}
  for i in range(len(embedding_cls)):
    index2cls[i] = embedding_cls[i]
  # print(index2cls[1])
  cPickle.dump(index2cls, open("../faiss_index/%s_index2cls" % testset, "wb", True))
  # index2cls = cPickle.load(open(../faiss_index/%s_index2cls" % testset, "rb", True))
  # print(index2cls[1])

  # mapping ID to image_path
  embedding_img = [x['img'] for x in samples]
  assert len(embedding_img) == embedding.shape[0], "the dim of embedding and img is not equal"
  index2img = {}
  for i in range(len(embedding_img)):
    index2img[i] = embedding_img[i]
  # print(index2img[1])
  cPickle.dump(index2img, open("../faiss_index/%s_index2img" % testset, "wb", True))
  # index2img = cPickle.load(open("../faiss_index/%s_index2img" % testset, "rb", True))
  # print(index2img[1])

def search_faiss(query_cache_file, k = 1, testset = "zhixinlian"):
  """
  Read query embeddings in cache_file, and conduct recall test from faiss
  k : recall top-k closest result
  """
  import faiss  # make faiss available
  import numpy as np
  from six.moves import cPickle

  test_num = 14625

  # 1. prepare query data
  samples = cPickle.load(open(query_cache_file, "rb", True))
  embedding = np.asarray([x['hist'] for x in samples])
  # print(embedding.shape)
  query_embedding = np.ascontiguousarray(embedding.astype(np.float32))
  print("query_embedding.shape: ", query_embedding.shape)
  query_embedding = query_embedding[:test_num]

  # 2. read faiss index of gallery
  index = faiss.read_index("../faiss_index/%s.index" % testset)

  # 3. search
  D, I = index.search(query_embedding, k)

  # mapping ID to cls
  embedding_cls = [x['cls'] for x in samples[:test_num]]
  assert len(embedding_cls) == query_embedding.shape[0], "the dim of embedding and cls is not equal"
  index2cls = {}
  for i in range(len(embedding_cls)):
    index2cls[i] = embedding_cls[i]
  # print(index2cls[1])
  cPickle.dump(index2cls, open("../faiss_index/%s_index2cls_query" % testset, "wb", True))
  # index2cls = cPickle.load(open("../faiss_index/%s_index2cls_query" % testset, "rb", True))
  # print(index2cls[1])

  # mapping ID to image_path
  embedding_img = [x['img'] for x in samples[:test_num]]
  assert len(embedding_img) == query_embedding.shape[0], "the dim of embedding and img is not equal"
  index2img = {}
  for i in range(len(embedding_img)):
    index2img[i] = embedding_img[i]
  # print(index2img[1])
  cPickle.dump(index2img, open("../faiss_index/%s_index2img_query" % testset, "wb", True))
  # index2img = cPickle.load(open("../faiss_index/%s_index2img_query" % testset, "rb", True))
  # print(index2img[1])

  return D, I
  

def evaluate_acc(db_base, db_query, db_original, D, I, testset = "zhixinlian", threshold = 0.9):
  """
  top1, judge the class by threshold (if less than threshold, set class 0), calculate acc
  """
  from six.moves import cPickle
  import numpy as np
  import os
  import matplotlib.pyplot as plt # plt 用于显示图片
  import matplotlib.image as mpimg # mpimg 用于读取图片
  import glob

  index2cls = cPickle.load(open("../faiss_index/%s_index2cls" % testset, "rb", True))
  index2cls_query = cPickle.load(open("../faiss_index/%s_index2cls_query" % testset, "rb", True))

  index2img = cPickle.load(open("../faiss_index/%s_index2img" % testset, "rb", True))
  index2img_query = cPickle.load(open("../faiss_index/%s_index2img_query" % testset, "rb", True))

  # get label
  base_cls = db_base.get_class()
  original_cls = db_original.get_class()
  gallery_cls_set = base_cls | original_cls
  query_cls = [index2cls_query[x] for x in range(len(index2cls_query))]
  query_cls_list = [x if x in gallery_cls_set else 0 for x in query_cls] #label (if a query_cls is not in gallery_cls_set, set to 0)
  
  # get y_pred
  index = np.squeeze(I[:, 0]).tolist()
  index_cls = [index2cls[x] for x in index]
  distance = np.squeeze(D[:, 0]).tolist()
  recall_cls_list = [] #y_pred (if a recall_cls is less than threshold, set to 0)
  for i, (index_cls_i, distance_i) in enumerate(zip(index_cls, distance)):
    if distance_i > threshold:
      recall_cls_list.append(index_cls_i)
    else:
      recall_cls_list.append(0)

  print("label:", query_cls_list[:5], len(query_cls_list))
  print("y_pred:", recall_cls_list[:5], len(recall_cls_list))

  # calculate total acc
  assert len(query_cls_list) == len(recall_cls_list), "length of label and y_pred is not equal"
  result = np.asarray(query_cls_list) == np.asarray(recall_cls_list)
  acc = result.sum() / len(query_cls_list)
  print("total acc:", acc)


  query_img = [index2img_query[x] for x in range(len(index2img_query))]
  index_img = [index2img[x] for x in index]

  # # plot case
  # wrong_index = np.where(result == 0)[0].tolist()
  # print("wrong num: ", len(wrong_index), "10 wrong index: ", wrong_index[:10])
  # # for wrong_index_i in wrong_index[:10]:
  # #   print("query image: %s, recall image: %s, distance %.2f" % (query_img[wrong_index_i], index_img[wrong_index_i], distance[wrong_index_i]))
  # for wrong_index_i in wrong_index[:10]:
  #   plt.figure()
  #   plt.subplot(1, 2, 1)
  #   img = mpimg.imread(query_img[wrong_index_i])
  #   plt.imshow(img)
  #   plt.subplot(1, 2, 2)
  #   img = mpimg.imread(index_img[wrong_index_i])
  #   plt.imshow(img)
  #   plt.axis('off') # 不显示坐标轴
  #   query_img_name = os.path.basename(query_img[wrong_index_i]).split('.')[0]
  #   recall_img_name = os.path.basename(index_img[wrong_index_i]).split('.')[0]
  #   plt.savefig('./total_wrong_%s_%s_%d.jpg' % (query_img_name, recall_img_name, int(distance[wrong_index_i]*1000)))
  #   # plt.show()
  #
  # true_index = np.where(result == 1)[0].tolist()
  # print("true num: ", len(true_index), "10 true index: ", true_index[:10])
  # for true_index_i in true_index[:10]:
  #   plt.figure()
  #   plt.subplot(1, 2, 1)
  #   img = mpimg.imread(query_img[true_index_i])
  #   plt.imshow(img)
  #   plt.subplot(1, 2, 2)
  #   img = mpimg.imread(index_img[true_index_i])
  #   plt.imshow(img)
  #   plt.axis('off') # 不显示坐标轴
  #   query_img_name = os.path.basename(query_img[true_index_i]).split('.')[0]
  #   recall_img_name = os.path.basename(index_img[true_index_i]).split('.')[0]
  #   plt.savefig('./total_true_%s_%s_%d.jpg' % (query_img_name, recall_img_name, int(distance[true_index_i]*1000)))
  #   # plt.show()

  # calculate > threshold acc
  index_recall_cls_list_not_zero = [i for i, e in enumerate(recall_cls_list) if e != 0]
  cnt_true = 0
  index_wrong_mark = []
  for i in index_recall_cls_list_not_zero:
    if (recall_cls_list[i] == query_cls_list[i]):
      cnt_true += 1
    else:
      index_wrong_mark.append(i)
  precision = cnt_true / len(index_recall_cls_list_not_zero)
  print("> threshold acc rate %.4f, cnt_true %d, total_cnt %d." % (precision, cnt_true, len(index_recall_cls_list_not_zero)))

  # calculate > threshold lose
  index_recall_cls_list_zero = [i for i, e in enumerate(recall_cls_list) if e == 0]
  cnt_lose = 0
  index_lose_mark = []
  for i in index_recall_cls_list_zero:
    if (recall_cls_list[i] == 0) and (query_cls_list[i] != 0):
      cnt_lose += 1
      index_lose_mark.append(i)
  lose = cnt_lose / len(recall_cls_list)
  print("> threshold lose rate %.4f, cnt_lose %d." % (lose, cnt_lose))

  # # wash data
  with open('./wash_data.sh', 'w') as f:
    for wrong_index_i in index_wrong_mark:
      query_path = query_img[wrong_index_i]
      original_path = index_img[wrong_index_i]
      query_dir = os.path.dirname(query_path)
      original_dir = os.path.dirname(original_path)
      query_name = os.path.basename(query_path)
      original_name = os.path.basename(original_path)
      query_cls = query_name.split('_')[0]
      original_cls = original_name.split('.')[0]

      f.writelines("mv %s %s\n" % (original_path, os.path.join(query_dir, query_cls)+'_100001.jpg'))
      for filename in glob.glob(os.path.join(query_dir, original_cls) + '*'):
        new_filename = filename.replace(original_cls, query_cls)
        f.writelines("mv %s %s\n" % (os.path.join(query_dir, filename), os.path.join(query_dir, new_filename)))

  # # plot case
  # for wrong_index_i in index_wrong_mark[:10]:
  #   plt.figure()
  #   plt.subplot(1, 2, 1)
  #   img = mpimg.imread(query_img[wrong_index_i])
  #   plt.imshow(img)
  #   plt.subplot(1, 2, 2)
  #   img = mpimg.imread(index_img[wrong_index_i])
  #   plt.imshow(img)
  #   plt.axis('off') # 不显示坐标轴
  #   query_img_name = os.path.basename(query_img[wrong_index_i]).split('.')[0]
  #   recall_img_name = os.path.basename(index_img[wrong_index_i]).split('.')[0]
  #   plt.savefig('./threshold_wrong_%s_%s_%d.jpg' % (query_img_name, recall_img_name, int(distance[wrong_index_i]*1000)))
  #   # plt.show()

  # for lose_index_i in index_lose_mark[:10]:
  #   plt.figure()
  #   plt.subplot(1, 2, 1)
  #   img = mpimg.imread(query_img[lose_index_i])
  #   plt.imshow(img)
  #   plt.subplot(1, 2, 2)
  #   img = mpimg.imread(index_img[lose_index_i])
  #   plt.imshow(img)
  #   plt.axis('off') # 不显示坐标轴
  #   query_img_name = os.path.basename(query_img[lose_index_i]).split('.')[0]
  #   recall_img_name = os.path.basename(index_img[lose_index_i]).split('.')[0]
  #   plt.savefig('./threshold_lose_%s_%s_%d.jpg' % (query_img_name, recall_img_name, int(distance[lose_index_i]*1000)))
  #   # plt.show()

def evaluate_recallk_by_file(input_file, k = 1):
    """
    Recall@1, Recall@5, Recall@10
    """
    # add_gaussian_noise/78089261739417600.jpg:78089261739417600,0.995399;
    hit_cnt = 0
    total_cnt = 0
    with open(input_file, mode='r') as f:
        results = f.readlines()
        for result in results:
            result = result.strip().strip(';')
            target = result.split(':')[0]
            # print(target)
            recall_list = result.split(':')[1].split(';')
            recall_list = [y[0] for y in [x.split(',') for x in recall_list]]
            # print(recall_list)
            if target in recall_list[:k]:
                hit_cnt += 1
            else:
                # print(target, recall_list)
                pass
            total_cnt += 1
    return hit_cnt/total_cnt, k


def evaluate_recallk(db_base, db_query, db_original, D, I, testset = "augmented", threshold = 0.9, test_type="query"):
  """
  calculate recall@1, recall@5, recall@10
  """
  from six.moves import cPickle
  import numpy as np
  import os
  import matplotlib.pyplot as plt # plt 用于显示图片
  import matplotlib.image as mpimg # mpimg 用于读取图片
  import glob

  index2cls = cPickle.load(open("../faiss_index/%s_index2cls" % testset, "rb", True))
  index2cls_query = cPickle.load(open("../faiss_index/%s_index2cls_query" % testset, "rb", True))

  index2img = cPickle.load(open("../faiss_index/%s_index2img" % testset, "rb", True))
  index2img_query = cPickle.load(open("../faiss_index/%s_index2img_query" % testset, "rb", True))

  # get label
  base_cls = db_base.get_class()
  original_cls = db_original.get_class()
  gallery_cls_set = base_cls | original_cls
  query_cls = [index2cls_query[x] for x in range(len(index2cls_query))]
  query_cls_list = [x if x in gallery_cls_set else 0 for x in query_cls] #label (if a query_cls is not in gallery_cls_set, set to 0)
  
  # get y_pred
  index = I.tolist()
  index_cls = [[index2cls[x] for x in item] for item in index]
  distance = D.tolist()

  with open('./%s_%s_output.txt' % (testset, test_type), 'w') as f:
    for i in range(len(query_cls_list)):
      f.writelines('%s:' % query_cls_list[i])
      for j in range(len(index_cls[0])):
        f.writelines('%s,%f;' % (index_cls[i][j], distance[i][j]))
      f.writelines('\n')
  
  recallk, k = evaluate_recallk_by_file(input_file = './%s_%s_output.txt' % (testset, test_type), k = 1)
  print("**********Test type: %s, Recall@%d: %f*************\n" % (test_type, k, recallk))
  with open('./result.csv', 'a') as f:
    f.writelines("%.3f\n" % recallk)
    


def eval_resnet():
  # *********************** resnet50 *******************
  # # generate image info
  db_base, db_query, db_original = get_image_info(testset = "zhixinlian")

  # get embedding and save in cache
  # get_embedding(use_script = "resnet", use_model = "resnet50", use_layer = "avg_normalized", use_db = db_original, use_cache = "zhixinlian_original")
  # get_embedding(use_script = "resnet", use_model = "resnet50", use_layer = "avg_normalized", use_db = db_query, use_cache = "zhixinlian_query")
  # get_embedding(use_script = "resnet", use_model = "resnet50", use_layer = "avg_normalized", use_db = db_base, use_cache = "base")

  # # # # add gallery embedding into faiss
  insert_faiss(base_cache_file = "../cache/base/resnet50-avg_normalized", original_cache_file = "../cache/zhixinlian_original/resnet50-avg_normalized")

  # # # # search faiss
  D, I = search_faiss(query_cache_file = "../cache/zhixinlian_query/resnet50-avg_normalized", k = 1, testset = "zhixinlian")

  # # # # calculate acc
  evaluate_acc(db_base, db_query, db_original, D, I, testset = "zhixinlian", threshold = 0.7)

def eval_resnet_simsiam():
  # *********************** resnet50 *******************
  # # generate image info
  db_base, db_query, db_original = get_image_info(testset = "zhixinlian")

  # # # get embedding and save in cache
  # get_embedding(use_script = "resnet_simsiam", use_model = "simsiam", use_layer = "projection", use_db = db_original, use_cache = "zhixinlian_original")
  # get_embedding(use_script = "resnet_simsiam", use_model = "simsiam", use_layer = "projection", use_db = db_query, use_cache = "zhixinlian_query")
  # get_embedding(use_script = "resnet_simsiam", use_model = "simsiam", use_layer = "projection", use_db = db_base, use_cache = "base")

  # # # # add gallery embedding into faiss
  insert_faiss(base_cache_file = "../cache/base/simsiam-projection", original_cache_file = "../cache/zhixinlian_original/simsiam-projection")

  # # # # search faiss
  D, I = search_faiss(query_cache_file = "../cache/zhixinlian_query/simsiam-projection", k = 1, testset = "zhixinlian")

  # # # # calculate acc
  evaluate_acc(db_base, db_query, db_original, D, I, testset = "zhixinlian", threshold = 0.7)

def eval_resnet_simsiam_imagenet():
  # *********************** resnet50 *******************
  # # generate image info
  db_base, db_query, db_original = get_image_info(testset = "zhixinlian")

  # # # get embedding and save in cache
  get_embedding(use_script = "resnet_simsiam_imagenet", use_model = "simsiam_imagenet_epoch50", use_layer = "projection", use_db = db_original, use_cache = "zhixinlian_original")
  get_embedding(use_script = "resnet_simsiam_imagenet", use_model = "simsiam_imagenet_epoch50", use_layer = "projection", use_db = db_query, use_cache = "zhixinlian_query")
  get_embedding(use_script = "resnet_simsiam_imagenet", use_model = "simsiam_imagenet_epoch50", use_layer = "projection", use_db = db_base, use_cache = "base")

  # # # # add gallery embedding into faiss
  insert_faiss(base_cache_file = "../cache/base/simsiam_imagenet_epoch50-projection", original_cache_file = "../cache/zhixinlian_original/simsiam_imagenet_epoch50-projection")

  # # # # search faiss
  D, I = search_faiss(query_cache_file = "../cache/zhixinlian_query/simsiam_imagenet_epoch50-projection", k = 1, testset = "zhixinlian")

  # # # # calculate acc
  evaluate_acc(db_base, db_query, db_original, D, I, testset = "zhixinlian", threshold = 0.990)


def eval_augmented(testset, db_base_dir, db_original_dir, use_script, use_model_base, use_model_original, use_layer,
                   use_cache_base, use_cache_original, use_cache_query, base_cache_file, original_cache_file, query_cache_file,
                   k, threshold, test_type, db_query_dir, use_model_query):
  """
  evaluate the recall@1 performance of each augmented type
  """
  db_base = get_image_info_single(db_base_dir)
  db_original = get_image_info_single(db_original_dir)
  db_query = get_image_info_single(db_query_dir)
  
  get_embedding(use_script = use_script, use_model = use_model_base, use_layer = use_layer, use_db = db_base, use_cache = use_cache_base)
  get_embedding(use_script = use_script, use_model = use_model_original, use_layer = use_layer, use_db = db_original, use_cache = use_cache_original)
  get_embedding(use_script = use_script, use_model = use_model_query, use_layer = use_layer, use_db = db_query, use_cache = use_cache_query)
  
  insert_faiss(base_cache_file = base_cache_file, original_cache_file = original_cache_file, testset = testset)
  
  D, I = search_faiss(query_cache_file = query_cache_file, k = k, testset = testset)

  # # # # # # calculate acc
  # evaluate_acc(db_base, db_query, db_original, D, I, testset = testset, threshold = threshold)

  # # # # # calculate recall@k
  evaluate_recallk(db_base, db_query, db_original, D, I, testset = testset, threshold = threshold, test_type=test_type)


def eval_recallk_resnet():
  import os

  ### config data source
  testset = "augmented"
  db_base_dir = '../image/base_gallery'
  db_original_dir = '../image/%s/original' % testset
  
  ### config model
  use_script = "resnet" ###
  # use_model: embedding name
  use_model_base = "resnet50" ###
  use_model_original = "resnet50_original" ###
  use_layer = "avg_normalized" ###
  # use_cache: embedding dir
  use_cache_base = "base"
  use_cache_original = testset
  use_cache_query = testset

  ### config cache dir
  base_cache_file = "../cache/base/%s-%s" % (use_model_base, use_layer)
  original_cache_file = "../cache/%s/%s-%s" % (testset, use_model_original, use_layer)

  ### config search
  k = 10
  threshold = 0.99

  if os.path.exists('result.csv'):
    os.remove('result.csv')

  test_type_list = ["query", "h_flip", "v_flip", "gaussian_blur", "gray", "image_Resize_100", "pad", "random_perspective",
                    "add_gaussian_noise", "color_jitter", "random_erasing", "random_resized_crop", "random_rotation", 
                    "random_affine", "add_salt_pepper_noise"]
  for test_type in test_type_list:
    db_query_dir = '../image/%s/%s' % (testset, test_type)
    use_model_query = "resnet50_%s" % test_type###
    query_cache_file = "../cache/%s/%s-%s" % (testset, use_model_query, use_layer)
    eval_augmented(testset, db_base_dir, db_original_dir, use_script, use_model_base, use_model_original, use_layer,
                   use_cache_base, use_cache_original, use_cache_query, base_cache_file, original_cache_file, query_cache_file,
                   k, threshold, test_type, db_query_dir, use_model_query)

def eval_recallk_resnet_simsiam_backbone():
  import os

  ### config data source
  testset = "augmented"
  db_base_dir = '../image/base_gallery'
  db_original_dir = '../image/%s/original' % testset
  
  ### config model
  use_script = "resnet_simsiam_imagenet" ###
  # use_model: embedding name
  use_model_base = "simsiam_imagenet_epoch100" ###
  use_model_original = "simsiam_imagenet_epoch100_original" ###
  use_layer = "backbone" ###
  # use_cache: embedding dir
  use_cache_base = "base"
  use_cache_original = testset
  use_cache_query = testset

  ### config cache dir
  base_cache_file = "../cache/base/%s-%s" % (use_model_base, use_layer)
  original_cache_file = "../cache/%s/%s-%s" % (testset, use_model_original, use_layer)

  ### config search
  k = 10
  threshold = 0.99

  if os.path.exists('result.csv'):
    os.remove('result.csv')

  test_type_list = ["query", "h_flip", "v_flip", "gaussian_blur", "gray", "image_Resize_100", "pad", "random_perspective",
                    "add_gaussian_noise", "color_jitter", "random_erasing", "random_resized_crop", "random_rotation", 
                    "random_affine", "add_salt_pepper_noise"]
  for test_type in test_type_list:
    db_query_dir = '../image/%s/%s' % (testset, test_type)
    use_model_query = "simsiam_imagenet_epoch100_%s" % test_type ###
    query_cache_file = "../cache/%s/%s-%s" % (testset, use_model_query, use_layer)
    eval_augmented(testset, db_base_dir, db_original_dir, use_script, use_model_base, use_model_original, use_layer,
                   use_cache_base, use_cache_original, use_cache_query, base_cache_file, original_cache_file, query_cache_file,
                   k, threshold, test_type, db_query_dir, use_model_query)


def eval_recallk_resnet_simsiam_projection():
  import os

  ### config data source
  testset = "augmented"
  db_base_dir = '../image/base_gallery'
  db_original_dir = '../image/%s/original' % testset
  
  ### config model
  use_script = "resnet_simsiam_imagenet" ###
  # use_model: embedding name
  use_model_base = "model-imagenet-horovod-augmented-single-30" ###
  use_model_original = "model-imagenet-horovod-augmented-single-30_original" ###
  use_layer = "projection" ###
  # use_cache: embedding dir
  use_cache_base = "base"
  use_cache_original = testset
  use_cache_query = testset

  ### config cache dir
  base_cache_file = "../cache/base/%s-%s" % (use_model_base, use_layer)
  original_cache_file = "../cache/%s/%s-%s" % (testset, use_model_original, use_layer)

  ### config search
  k = 10
  threshold = 0.99

  if os.path.exists('result.csv'):
    os.remove('result.csv')

  test_type_list = ["query", "h_flip", "v_flip", "gaussian_blur", "gray", "image_Resize_100", "pad", "random_perspective",
                    "add_gaussian_noise", "color_jitter", "random_erasing", "random_resized_crop", "random_rotation", 
                    "random_affine", "add_salt_pepper_noise"]
  for test_type in test_type_list:
    db_query_dir = '../image/%s/%s' % (testset, test_type)
    use_model_query = "model-imagenet-horovod-augmented-single-30_%s" % test_type ###
    query_cache_file = "../cache/%s/%s-%s" % (testset, use_model_query, use_layer)
    eval_augmented(testset, db_base_dir, db_original_dir, use_script, use_model_base, use_model_original, use_layer,
                   use_cache_base, use_cache_original, use_cache_query, base_cache_file, original_cache_file, query_cache_file,
                   k, threshold, test_type, db_query_dir, use_model_query)
  

if __name__ == "__main__":
  # eval_resnet()
  # eval_resnet_simsiam()
  # eval_resnet_simsiam_imagenet()

  # eval_recallk_resnet() #dim=2048
  # eval_recallk_resnet_simsiam_backbone() #dim=2048
  eval_recallk_resnet_simsiam_projection() #dim=512
  

  

