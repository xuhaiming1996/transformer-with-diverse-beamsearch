### 技巧代代码

class TrickUtil:
    @staticmethod
    def sortByIndex(arrs,index):
        '''

        :param arrs: lsit(list())
        :param index:  按照哪一个位置进行排序
        :return:
        '''
        arrs.sort(key=lambda x: x[index])

