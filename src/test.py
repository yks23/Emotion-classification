from gensim.models import KeyedVectors

# 加载 word2vec 的二进制模型
wv = KeyedVectors.load_word2vec_format('作业2/Dataset/wiki_word2vec_50.bin', binary=True)

# 查看某个词的向量
print(len(wv['中国']))

# 打印前 5 个词
print(list(wv.index_to_key[:5]))