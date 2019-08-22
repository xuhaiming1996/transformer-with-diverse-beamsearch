import sys
'''
给你一个字符串 S、一个字符串 T，请在字符串 S 里面找出：包含 T 所有字母的最小子串。

'''
class Solution:
    @staticmethod
    def minWindow(s: str, t: str) -> str:
        start = 0 #标记开始位置
        min_len = sys.maxsize # 标记包含子串的长度
        needs = dict()
        windows = dict()
        for ch in t:
            if ch in needs:
                needs[ch] += 1
            else:
                needs[ch] = 1

        left,right=0, 0
        macth=0
        while right<len(s):
            ch1 = s[right]
            if ch1 in needs:
                if ch1 in windows:
                    windows[ch1]+=1
                else:
                    windows[ch1] =1

                if windows[ch1]==needs[ch1]:
                    macth+=1

            right+=1
            while macth==len(needs):
                if right-left<min_len:
                    min_len=right-left
                    start=left
                c2 = s[left]
                if c2 in needs:
                    windows[c2] -= 1
                    if windows[c2] < needs[c2]:
                        macth -= 1
                left += 1

        if min_len!=sys.maxsize:
            return s[start:start+min_len]
        return ""






if __name__=="__main__":
    S = sys.stdin.readline().strip()
    T = sys.stdin.readline().strip()
    print(Solution.minWindow(S,T))

