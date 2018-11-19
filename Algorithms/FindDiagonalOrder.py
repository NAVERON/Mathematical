class Solution(object):
    def findDiagonalOrder(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: List[int]
        """
        if not matrix:
            return []
        m = len(matrix)   #行数
        n = len(matrix[0])  #列数
        max_mn = m+n-1  #元素总数
        # i = 0
        # j = 0
        res_index = []
        print(max_mn)
        flag = 0
        for sum_ij in range(max_mn):
            i = flag
            j = sum_ij - flag
            if j == n:   # 如果一行完成
                flag += 1
                i = flag
                j = sum_ij-flag
                # print("j==n:",i,j)
            temp = []
            # print(i, j)
            # print("----")
            while 0 <=i< m and 0 <=j< n:
                temp.append(matrix[i][j])
                i += 1
                j -= 1
                # print(i, j)
            res_index.append(temp)
        l = len(res_index)
        res = []
        # 偶数取反,奇数取正
        for k in range(l):
            if k%2 == 0:
                res += res_index[k][::-1]
            else:
                res += res_index[k]
        return res
