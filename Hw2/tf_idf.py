# -*- coding: utf-8 -*-

import jieba
import codecs
from collections import Counter
import math

## 去除停用词的2个函数
# 创建停用词list
def stopwordslist(filepath):
    stopwords = [line.strip() for line in codecs.open(filepath, 'r', encoding='utf-8').readlines()]
    return stopwords

# 对句子去除停用词
def movestopwords(sentence):
    stopwords = stopwordslist('/Users/sousic/SIAT/2nd_semester/bio_info/Hw/Hw2/stopwords.txt')  # 这里加载停用词的路径
    outstr = ''
    for word in sentence:
        if word not in stopwords:
            if word != '\t' and '\n':
                outstr += word
                outstr += " "
    return outstr

def segment(inputFile, outputFile):
	path='/Users/sousic/SIAT/2nd_semester/bio_info/Hw/Hw2/corpus_self/'
	inputs = codecs.open(path+inputFile, 'r', encoding='utf-8')
	outputs = codecs.open(path+outputFile, 'w', encoding='utf-8')
	lines = inputs.readlines()
	for line in lines:
		#分词
		line_seged = jieba.cut(line.strip(), cut_all=False)
		#除去停用词
		line_seged_stoped = movestopwords(line_seged)
		#写入
		outputs.write(line_seged_stoped)
	outputs.close()
	inputs.close()


##tf-idf
#tf
# word可以通过count得到，count可以通过countlist得到
# count[word]可以得到每个单词的词频， sum(count.values())得到整个句子的单词总数
def tf(word, count):
	tf_num = float(count[word]) / sum(count.values())
	return tf_num

# 统计的是含有该单词的句子数
def n_containing(word, count_list):
	n_containing = sum(1 for count in count_list if word in count)
	return n_containing
 
# len(count_list)是指句子的总数，n_containing(word, count_list)是指含有该单词的句子的总数，加1是为了防止分母为0
def idf(word, count_list):
	idf_num = math.log((1 + len(count_list)) / (1 + n_containing(word, count_list)))
	return idf_num

# 将tf和idf相乘
def tfidf(word, count, count_list):
	# print 'tf',tf(word, count)
	# print 'idf',idf(word, count_list)
	tfidf_num = tf(word, count) * idf(word, count_list)
	return tfidf_num


def extract_tfidf(inputFiles, outputFiles, save = True, view = False):
	'''
	提取tfidf分数排前20的词汇，返回一个字典，key=词，value=tfidf值
	'''
	path = '/Users/sousic/SIAT/2nd_semester/bio_info/Hw/Hw2/corpus_self/'
	#inputFiles = ['keji.txt', 'junshi.txt', 'caijing.txt', 'tiyu.txt']
	#outputFiles = ['keji_seg.txt', 'junshi_seg.txt', 'caijing_seg.txt', 'tiyu_seg.txt']
	if save == True:
		for inputFile,outputFile in zip(inputFiles,outputFiles):
			segment(inputFile, outputFile)

	corpus = []
	#读分词后的文件
	for outputFile in outputFiles:
		reader = codecs.open(path+outputFile, 'r' , encoding='utf-8')
		lines = reader.readlines()
		#print lines[0]
		corpus.append(lines[0])
	# print len(corpus) #4

	words_list = []
	for i in range(len(corpus)):
		words_list.append(corpus[i].split(' '))
	# print len(words_list) #4

	countlist = []
	for i in range(len(words_list)):
	    count = Counter(words_list[i])
	    countlist.append(count)
	# print len(countlist) #4

	# for j in range(len(countlist[0])):
	# 	print countlist[0].keys()[j],':',countlist[0].values()[j]


	tfidf_sorted_dic = {}
	tf_sorted_dic = {}
	idf_sorted_dic = {}	
	for i in range(len(countlist)):
		tf_dic ={}
		idf_dic = {}
		tfidf_dic = {}
		for j in range(len(countlist[i].keys())):
			tf_num = tf(countlist[i].keys()[j], countlist[i])
			# print tf_num
			tf_dic[countlist[i].keys()[j]] = tf_num
			idf_num = idf(countlist[i].keys()[j], countlist)
			# print idf_num
			idf_dic[countlist[i].keys()[j]] = idf_num
			tfidf_num = tfidf(countlist[i].keys()[j], countlist[i], countlist)
			# print tfidf_num
			tfidf_dic[countlist[i].keys()[j]] = tfidf_num

		tf_sorted = sorted(tf_dic.items(), key = lambda x:x[1], reverse = True)[:20]
		tf_sorted_dic[inputFiles[i]] = tf_sorted
		idf_sorted = sorted(idf_dic.items(), key = lambda x:x[1], reverse = True)[:20]
		idf_sorted_dic[inputFiles[i]] = idf_sorted
		tfidf_sorted = sorted(tfidf_dic.items(), key = lambda x:x[1], reverse = True)[:20]
		tfidf_sorted_dic[inputFiles[i]] = tfidf_sorted
		if view == True:
			for k in range(len(tfidf_sorted)):
				print tfidf_sorted[k][0],':',tfidf_sorted[k][1]

	# print tfidf_sorted_dic[inputFiles[0]] #第一篇文章选出来的词
	return tfidf_sorted_dic, tf_sorted_dic, idf_sorted_dic

def show_sorted(tf_sorted_dic, idf_sorted_dic, tfidf_sorted_dic , files):
	#tf
	for i in range(len(tf_sorted_dic)):
		print '\n-----------------%s文章tf前20词汇----------------'%(files[i])
		for j in range(len(tf_sorted_dic[files[0]])):
			print tf_sorted_dic[files[i]][j][0]+':'+str(tf_sorted_dic[files[i]][j][1]),
	print '\n-----------------------------------------------------------'
	#idf
	for i in range(len(idf_sorted_dic)):
		print '\n-----------------%s文章idf前20词汇----------------'%(files[i])
		for j in range(len(idf_sorted_dic[files[0]])):
			print idf_sorted_dic[files[i]][j][0]+':'+str(idf_sorted_dic[files[i]][j][1]),
	print '\n-----------------------------------------------------------'
	#tfidf
	for i in range(len(tfidf_sorted_dic)):
		print '\n-----------------%s文章tfidf前20词汇----------------'%(files[i])
		for j in range(len(tfidf_sorted_dic[files[0]])):
			print tfidf_sorted_dic[files[i]][j][0]+':'+str(tfidf_sorted_dic[files[i]][j][1]),
	print '\n-----------------------------------------------------------'

def get_outputFiles(inputFiles):
	outputFiles = []
	for  name in inputFiles:
		outputFiles.append(name[:-4]+'_seg'+name[-4:])
	return outputFiles

def get_samilarest_index(words_lists):
	same_word = []
	for i in range(len(words_lists)-1):
		sum = 0
		for j in range(len(words_lists[4])):
			if words_lists[4][j] in words_lists[i]:
				sum += 1
		same_word.append(sum)
	return same_word.index(max(same_word)),same_word

def main():
	inputFiles = ['keji.txt', 'junshi.txt', 'caijing.txt', 'tiyu.txt', 'keji_test.txt'] #四种类型的文章和一篇测试文章
	outputFiles = get_outputFiles(inputFiles)
	tfidf_sorted_dic, tf_sorted_dic, idf_sorted_dic = extract_tfidf(inputFiles, outputFiles, save=False, view=False)
	
	show_sorted(tf_sorted_dic, idf_sorted_dic, tfidf_sorted_dic, inputFiles) #中间过程：输出tfidf分数排前20的

	words_lists = []
	for i in range(len(inputFiles)):
		words_list = []
		for j in range(len(tfidf_sorted_dic[inputFiles[i]])):
			words_list.append(tfidf_sorted_dic[inputFiles[i]][j][0])
		words_lists.append(words_list)

	samilar_index,same_word = get_samilarest_index(words_lists)

	print '与4篇文章的相同词数：',same_word
	print '同类文章为：',inputFiles[samilar_index]

	countlist = []
	for i in range(len(words_lists)):
		count = Counter(words_lists[i])
		countlist.append(count)


if __name__ == '__main__':
	main()