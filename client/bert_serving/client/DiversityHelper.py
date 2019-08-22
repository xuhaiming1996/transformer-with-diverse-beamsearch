
'''

该类主要是保留生成文本的多样性

'''

import re

from nltk.translate.bleu_score import sentence_bleu
import jieba

class DiversityHelper:
    @staticmethod
    def beamSeachDeduplicationByPunc(sens):
        '''
        考虑到有一些句子之间差异只有标点符号，根据标点符号去重
        :param sens:
        :return:
        '''
        sens_new = []
        sens_tmp=[]
        for sen in sens:
            sen_no_punc = DiversityHelper._removePunctuation(sen.strip())
            if sen_no_punc not in sens_tmp:
                sens_tmp.append(sen_no_punc)
                sens_new.append(sen)

        return sens_new




    @staticmethod
    def sort_by_bleu(input,outputs):
        '''
        根据和input之间的编辑距离对outputs进行排序
        :param input:
        :param outputs:
        :return:
        '''
        dict_tmp={}
        for output in outputs:
            score = sentence_bleu([list(jieba.cut(input))],list(jieba.cut(output)))
            dict_tmp[output]=score
        # 对字典按照value从大到小进行排序、，返回keys
        res = []
        for sen,score in sorted(dict_tmp.items(), key=lambda item: item[1]):
            res.append(sen)
        return res


    @staticmethod
    def sort_by_TER(input,outputs):
        '''
        根据和input之间的编辑距离对outputs进行排序
        :param input:
        :param outputs:
        :return:
        '''
        dict_tmp={}
        for output in outputs:
            tER = DiversityHelper._minEditDistance(input,output)
            dict_tmp[output]=tER
        # 对字典按照value从大到小进行排序、，返回keys
        res = []
        for sen,ter in sorted(dict_tmp.items(), key=lambda item: item[1],reverse=True):
            res.append(sen)
        return res






    @staticmethod
    def _removePunctuation(text):
        text = re.sub(r'[{}]+'.format('（）！，。：；“‘’”、？!,;:?"\''), '', text)
        return text.strip().lower()


    @staticmethod
    def _find_LCS(s1, s2):
        '''
        求最长公共子串
        :param s1:
        :param s2:
        :return:
        '''
        m = [[0 for i in range(len(s2) + 1)] for j in range(len(s1) + 1)]  # 生成0矩阵，为方便后续计算，比字符串长度多了一列
        mmax = 0  # 最长匹配的长度
        p = 0  # 最长匹配对应在s1中的最后一位
        for i in range(len(s1)):
            for j in range(len(s2)):
                if s1[i] == s2[j]:
                    m[i + 1][j + 1] = m[i][j] + 1
                    if m[i + 1][j + 1] > mmax:
                        mmax = m[i + 1][j + 1]
                        p = i + 1
        return  mmax  # 最长匹配的长度


    @staticmethod
    def _minEditDistance(word1, word2):
        """
        编辑距离
        :type word1: str
        :type word2: str
        :rtype: int
        """
        len_word1 = len(word1)
        len_word2 = len(word2)
        dp = []
        for row in range(len_word1 + 1):
            this_tmp = []
            for col in range(len_word2 + 1):
                if row == 0:
                    this_tmp.append(col)
                elif col == 0:
                    this_tmp.append(row)
                else:
                    this_tmp.append(False)
            dp.append(this_tmp)
        # print(dp)
        for row in range(1, len_word1 + 1):
            for col in range(1, len_word2 + 1):
                if word1[row - 1] == word2[col - 1]:
                    dp[row][col] = dp[row - 1][col - 1]
                else:
                    dp[row][col] = min(dp[row - 1][col], dp[row - 1][col - 1], dp[row][col - 1]) + 1
        return dp[len_word1][len_word2]


if __name__=="__main__":

    #
    # text = "表示全局（global）模式，即模式将被应用于所有字符串，而非在发现第一个匹配项时立即停止； "
    # print(removePunctuation(text))
    # #

    diversityHelper =DiversityHelper(filepath="./views_40.txt")
    diversityHelper._readfile()
