# coding=utf-8

"""
CBIR pipeline:
1. get query embeddings and cached, get gallery embeddings and cached
2. add gallery embeddings into faiss, conduct recall using query embeddings
3. calculate "top1-acc(recall@1)" and "presision-recall curve" based on the recall result
"""

import argparse
import os
import faiss  # make faiss available
import numpy as np
from six.moves import cPickle
from DB import Database


def get_image_info_single(DB_dir):
    """
    Generate single db & db.csv
    """
    DB_csv = os.path.join(os.path.split(DB_dir)[0], os.path.split(DB_dir)[1] + '.csv')
    db = Database(DB_dir, DB_csv)
    data = db.get_data()
    classes = db.get_class()
    print("organize image from dir %s, generate csv file %s, get db of length %d and classes %d." % 
          (DB_dir, DB_csv, len(db), len(classes)))
    return db


def get_embedding(use_model, use_db, use_cache, args):
    """
    Generate l2-normalized embedding of images in use_db, and save embedding into cache_file
    """
    if args.use_script == "resnet":
      from resnet import ResNetFeat
      samples = ResNetFeat(use_model, args.use_layer, use_cache).make_samples(use_db)
      print(samples[:5])
    elif args.use_script == "resnet_simsiam":
      from resnet_simsiam import ResNetFeat
      samples = ResNetFeat(use_model, args.use_layer, use_cache).make_samples(use_db)
    elif args.use_script == "resnet_simsiam_imagenet":
      from resnet_simsiam_imagenet import ResNetFeat
      samples = ResNetFeat(use_model, args.use_layer, use_cache).make_samples(use_db)
    elif args.use_script == 'esvit':
      from esvit_embeddings import EsViTFeat
      samples = EsViTFeat(use_model, args.use_layer, use_cache, args).make_samples(use_db)
    else:
      print("use_script not implemented!")
      pass

def insert_faiss(args):
    """
    Read base_gallery embeddings in cache_file, and add into faiss
    ref: https://waltyou.github.io/Faiss-Introduce/
    """
    # 读取base embedding
    samples_base = []
    if os.path.exists(args.base_cache_file):
        base_cache_dir = os.path.dirname(args.base_cache_file)
        base_cache_file = os.path.basename(args.base_cache_file)
        # print(base_cache_dir)
        files = os.listdir(base_cache_dir)
        use_files = [x for x in files if x.startswith(base_cache_file)]
        print("load base embddings from files: ", use_files)
        for f in use_files:
            tmp = cPickle.load(open(os.path.join(base_cache_dir, f), "rb", True))
            samples_base += tmp
    print("num of samples_base: ", len(samples_base))
    samples = samples_base
    
    # 获取所有被查询的embedding  
    embedding = np.asarray([x['hist'] for x in samples])
    # print("embedding.shape: ", embedding.shape)
    image_embedding = np.ascontiguousarray(embedding.astype(np.float32))
    d = image_embedding.shape[1] # dimension
    nb = image_embedding.shape[0]  # database size
    # print(image_embedding.shape)

    # 创建faiss索引index，把被查询的embedding加进去
    # index = faiss.IndexFlatL2(d)  # bruteforce L2-distance, add embeddings directly, no need to train
    index = faiss.IndexFlatIP(d) # InnerProduct
    # print(index.is_trained)
    index.add(image_embedding)  # add vectors to the index
    # print(index.ntotal)
    
    # # 检查下faiss检索是否ok
    # k = 4  # we want to see 4 nearest neighbors
    # query_embedding = image_embedding[:5]
    # D, I = index.search(query_embedding, k)  # sanity check
    # print("check faiss search: ")
    # print("I:", I)  # N x k, the top-k neighbors' ID of each query embedding
    # print("D:", D)  # N x k, the top-k neighbors' Distance of each query embedding
    
    # 将建好的索引存下来
    save_path = os.path.join(args.base_dir, "faiss_index/index.index")
    faiss.write_index(index, save_path)
    print("save faiss index into file: ", save_path)

    # 将gallery embedding的IDID映射到类别cls
    embedding_cls = [x['cls'] for x in samples]
    assert len(embedding_cls) == embedding.shape[0], "the dim of embedding and cls is not equal"
    index2cls = {}
    for i in range(len(embedding_cls)):
        index2cls[i] = embedding_cls[i]
    cPickle.dump(index2cls, open(os.path.join(args.base_dir, "faiss_index/index2cls"), "wb", True))

    # 将gallery embedding的ID的ID映射到图片位置image_path
    embedding_img = [x['img'] for x in samples]
    assert len(embedding_img) == embedding.shape[0], "the dim of embedding and img is not equal"
    index2img = {}
    for i in range(len(embedding_img)):
        index2img[i] = embedding_img[i]
    cPickle.dump(index2img, open(os.path.join(args.base_dir, "faiss_index/index2img"), "wb", True))


def search_faiss(args):
    """
    Read query embeddings in cache_file, and conduct recall test from faiss
    """
    test_num = 14625 #???
    
    # 读取query embedding
    samples = cPickle.load(open(args.query_cache_file, "rb", True))
    embedding = np.asarray([x['hist'] for x in samples])
    query_embedding = np.ascontiguousarray(embedding.astype(np.float32))
    query_embedding = query_embedding[:test_num]
    print("query_embedding shape: ", query_embedding.shape)

    # 读取已经建好的faiss index
    index = faiss.read_index(os.path.join(args.base_dir, "faiss_index/index.index"))

    # 用query embdding检索faiss，得到Nxtopk的distance和index
    D, I = index.search(query_embedding, args.topk)
    # print("I:", I)  # N x k, the top-k neighbors' ID of each query embedding
    # print("D:", D)  # N x k, the top-k neighbors' Distance of each query embedding

    # 将query embedding的ID映射到类别cls
    embedding_cls = [x['cls'] for x in samples[:test_num]]
    assert len(embedding_cls) == query_embedding.shape[0], "the dim of embedding and cls is not equal"
    index2cls = {}
    for i in range(len(embedding_cls)):
        index2cls[i] = embedding_cls[i]
    cPickle.dump(index2cls, open(os.path.join(args.base_dir, "faiss_index/index2cls_query"), "wb", True))

    # 将query embedding的ID映射到图片位置image_path
    embedding_img = [x['img'] for x in samples[:test_num]]
    assert len(embedding_img) == query_embedding.shape[0], "the dim of embedding and img is not equal"
    index2img = {}
    for i in range(len(embedding_img)):
        index2img[i] = embedding_img[i]
    cPickle.dump(index2img, open(os.path.join(args.base_dir, "faiss_index/index2img_query"), "wb", True))
    
    return D, I
 

def calculate_metric(threshold, args):
    """
    给定阈值得到预测类别后，计算precision和recall，其中precision是avg_sample(hit_cnt/pred_cnt), recall是avg_sample(hit_cnt/label_cnt)，在当前场景下label_cnt恒等于1
    """
    # 格式：label:pred_cls,pred_score;pred_cls,pred_score;
    # 78089261739417600:78089261739417600,0.995399;
    
    # 获取label_list，按threshold划分得到pred_list
    label_list = []
    pred_list = []
    with open(args.search_result_file, mode='r') as f:
        results = f.readlines()
        for result in results:
            # print("result:", result)
            # label
            label = result.strip().split(':')[0] # 对于每个query图片，label是一个值
            label_list.append(label)
            # pred
            pred_result = result.strip().split(':')[1].strip(';').split(';')
            # print("pred_result:", pred_result)
            pred_cls_list = [] # 对于每个query图片，pred是一个list
            for item in pred_result:
                try:
                    pred_cls, pred_score = item.split(',')
                    if float(pred_score) >= threshold:
                        pred_cls_list.append(pred_cls)
                    else:
                        break
                except:
                    print("item:", item)
            pred_list.append(pred_cls_list)
    
    
    # 求presision和recall
    assert len(label_list) == len(pred_list), "len(label_list) != len(pred_list)"
    
    sum_precision = 0.0
    sum_recall = 0.0
    """
    tp: base里有该query，且检索结果是有
    fp: base里没有该query，但检索结果是有
    tn: base里没有该query，且检索结果是没有
    fn: base里有该query，但检索结果是没有
    """
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    
    for i in range(len(label_list)):
        if label_list[i] in pred_list[i]: # base里有该query，且检索结果是有
            upper = 1
            tp += 1
        elif (label_list[i] == '0') and (len(pred_list[i]) == 0): # base里没有该query，且检索结果是没有
            upper = 1
            tn += 1
        elif (label_list[i] == '0') and (len(pred_list[i]) != 0): # base里没有该query，但检索结果是有
            upper = 0
            fp += 1
        else: # base里有该query，但检索结果是没有
            upper = 0
            fn += 1
        
        lower_precision = max(len(pred_list[i]), 1)
        lower_recall = 1
        
        sum_precision += upper / lower_precision
        sum_recall += upper / lower_recall
      
    precision = sum_precision / len(label_list)
    recall = sum_recall / len(label_list)
    
    pos_precision = tp / max(tp + fp, 1)
    pos_recall = tp / max(tp + fn, 1)
    pos_samples = tp + fn
    
    neg_precision = tn / max(tn + fn, 1)
    neg_recall = tn / max(tn + fp, 1)
    neg_samples = tn + fp
    
      
    return precision, recall, pos_precision, pos_recall, pos_samples, neg_precision, neg_recall, neg_samples


def prepare_search_result(db_base, db_query, D, I, args):
    """
    given threshold, calculate top-k precision & recall
    """
    # 读取映射
    # gallery embedding的id到类别
    index2cls = cPickle.load(open(os.path.join(args.base_dir, "faiss_index/index2cls"), "rb", True))
    # query embedding的id到类别
    index2cls_query = cPickle.load(open(os.path.join(args.base_dir, "faiss_index/index2cls_query"), "rb", True))
    # gallery embedding的id到图片路径
    index2img = cPickle.load(open(os.path.join(args.base_dir, "faiss_index/index2img"), "rb", True))
    # query embedding的id到图片路径
    index2img_query = cPickle.load(open(os.path.join(args.base_dir, "faiss_index/index2img_query"), "rb", True))

    # 获取query图片的label：如果在gallery类别里面，那就是query cls本身；如果不在，那就是'0'
    base_cls = db_base.get_class()
    query_cls = [index2cls_query[x] for x in range(len(index2cls_query))]
    query_cls_list = [x if x in base_cls else '0' for x in query_cls]
    print("!!! %d query img not in gallery" % query_cls_list.count('0'))
    
    # 将query检索结果按指定格式写到文件里（为了和C++的对齐，统一评测）
    # 每一行是一张query图片的topk检索结果，格式如下：
    # label:pred_cls,pred_score;...
    index = I.tolist()
    index_cls = [[index2cls[x] for x in item] for item in index]
    distance = D.tolist()
    with open(args.search_result_file, 'w') as f:
        for i in range(len(query_cls_list)):
            f.writelines('%s:' % query_cls_list[i])
            for j in range(len(index_cls[0])):
                f.writelines('%s,%f;' % (index_cls[i][j], distance[i][j]))
            f.writelines('\n')
    print("saved faiss search formatted result file into: ", args.search_result_file)


def eval_metric(args):
    """
    evaluate the precision recall performance
    
    把query查faiss的过程看作是对query每张图片进行多分类+多标签分类的过程，
    多分类是取最大的k个类别即可（看top-k acc），多标签分类要对各类预测分数划threshold，来决定是否命中该类，
    我们是二者的结合：先取top-k，再根据threshold来决定是否命中。
    
    对于每张query图片：
    label：如果在图库里有这个类别（图片id），则label是图片id；如果在图库里没有这个类别（图片id），则label记为0
    pred：查faiss，取相似度topk的类别，取其中大于阈值的类别，作为预测结果；如果没有大于阈值的类别，则预测结果是label 0。最后结果是类别列表，长度范围是[1, topk]
    
    多标签分类的metric：https://xie.infoq.cn/article/42bb9839d31c994e001b0162b
    给定阈值得到预测类别后，可以计算precision和recall，其中precision是avg_sample(hit_cnt/pred_cnt), recall是avg_sample(hit_cnt/label_cnt)，在当前场景下label_cnt恒等于1
    
    看不同阈值下的p和r，选出合适的阈值。threshold越大，p越小、r越大；threshold越小，p越大、r越小
    """
    print("********** Begin evaluate type %s *************" % args.test_type)
    with open(args.eval_result_file, 'a') as f:
        f.writelines("threshold, precision, recall, pos_precision, pos_recall, pos_samples, neg_precision, neg_recall, neg_samples\n")
                     
    # 从图片文件夹组织出图片及其类别，类别是其id
    db_query = get_image_info_single(args.db_query_dir)
    db_base = get_image_info_single(args.db_base_dir)
    
    # 获取query、base的图片embedding，存到对应的cache文件夹下
    get_embedding(use_model=args.use_model_query, use_db=db_query, use_cache=args.use_cache_query, args=args)
    get_embedding(use_model=args.use_model_base, use_db=db_base, use_cache=args.use_cache_base, args=args)
    
    # 生成base embedding加入faiss后的faiss_index
    insert_faiss(args)
    
    # 用query的embedding去查faiss，把查到的结果存下来
    D, I = search_faiss(args)

    # 准备label和query查faiss的结果，按指定格式写成结果文件，对齐C++的文件格式
    prepare_search_result(db_base, db_query, D, I, args)
    
    # 计算各阈值下的presion和recall
    threshold_list = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
    for threshold in threshold_list:
        precision, recall, pos_precision, pos_recall, pos_samples, neg_precision, neg_recall, neg_samples = calculate_metric(threshold, args)
        with open(args.eval_result_file, 'a') as f:
            f.writelines("%.3f, %.3f, %.3f, %.3f, %.3f, %d, %.3f, %.3f, %d\n" % (threshold, precision, recall, pos_precision, pos_recall, pos_samples, neg_precision, neg_recall, neg_samples))
    print("********** Done evaluate type: %s *************\n" % args.test_type)
    
    

def parse_args():
    parser = argparse.ArgumentParser('eval_top1acc_precisionrecallcurve')
    
    parser.add_argument('--base_dir', type=str, default='./data/coverimage/',  #/src/apdcephfs/private_xuanxuangao_2/ImageDeduplicate/CBIR/
                        help='data source project')
    parser.add_argument('--use_script', type=str, default='esvit',
                        help='which extract embedding script to be used')
    parser.add_argument('--use_layer', type=str, default='projection',
                        help='part of embedding name')
    parser.add_argument('--use_model_base', type=str, default='model-esvit-swinT',
                        help='part of embedding name')
    parser.add_argument('--use_cache_base', type=str, default='base',
                        help='part of embedding dir')
    parser.add_argument('--use_cache_query', type=str, default='query',
                        help='part of embedding dir')
    
    parser.add_argument('--topk', type=int, default=20, help='search top-k')
    parser.add_argument('--threshold', type=float, default=0.99, help='search threshold')
    parser.add_argument('--search_result', type=str, default='search_output.txt', help='the faiss search result formatted file')
    
    args = parser.parse_args()
    
    args.db_base_dir = os.path.join(args.base_dir, 'image/')
    args.base_cache_file = os.path.join(args.base_dir, "cache/base/%s-%s" % (args.use_model_base, args.use_layer)) # base embedding file
    return args


def main():
    args = parse_args()
    print(args)

    # 要测评的query类型
    test_type_list = ["image_query", "h_flip", "v_flip", "gaussian_blur", "resize", "pad", "random_perspective",
                      "add_gaussian_noise", "color_jitter", "random_erasing", "random_resized_crop", "random_rotation", 
                      "random_affine", "add_salt_pepper_noise", "gray", "center_crop", "random_crop"]
    
    for test_type in test_type_list:
        args.test_type = test_type
        args.db_query_dir = os.path.join(args.base_dir, args.test_type) # query图片所在文件夹
        args.use_model_query = "%s-%s" % (args.use_model_base, args.test_type) 
        args.query_cache_file = os.path.join(args.base_dir, "cache/%s-%s" % (args.use_model_query, args.use_layer)) # query embedding file
        args.search_result_file = './evaluate_performance_result/%s_%s' % (args.test_type, args.search_result)
        args.eval_result_file = './evaluate_performance_result/eval_result_%s.csv' % args.test_type
        
        # 删掉上次的结果文件
        if os.path.exists(args.eval_result_file):
            os.remove(args.eval_result_file)
            
        eval_metric(args)

if __name__ == "__main__":
    main()
    
"""
after run preprocess/generate_augmented_image.py

python3 eval_precision_recall.py
"""