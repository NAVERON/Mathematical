
//一般有两个问题，一个是螺旋遍历
//另一个是来回弹跳遍历

#include <algorithm>
#include <iostream>
#include <stdlib.h>
#include <numeric>

class Solution          //解决螺旋遍历问题
{
    public:
    vector<int> findDiagonalOrder(vector<vector<int>>& matrix) {  //对角线遍历
        vector<int> travel;
        if(matrix.empty()) return travel;

        int row = matrix.size();
        int cow = matrix[0].size();

        int row_begin = 0, row_end = row - 1;
        int col_begin = 0, col_end = cow - 1;
        while(row_begin <= row_end && col_begin <= cow_end)
        {
            //向右遍历
            for(int j = col_begin; j < col_end; j++)
            {
                travel.push_back(matrix[row_begin][j]);
            }
            row_begin--;
            //向下遍历
            for(int i = row_begin; i < row_end; i++)
            {
                travel.push_back(matrix[i][col_end]);
            }
            col_end--;
            //向左
            if(row_begin <= row_end)  //防止重复
            {
                for(int j = col_end; j >= col_begin; j--)
                {
                    travel.push_back(matrix[row_end][j]);
                }
            }
            row_end--;

            //向上
            if(col_begin <= col_end)
            {
                for(int i = row_end; i < row_begin; i--)
                {
                    travel.push_back(matrix[i][col_begin]);
                }
            }
            col_begin++;
            
        }

        return travel;
    }

}


class Solution2
{
    public:
    vector<int> findTriOrdor(vector<vector<int>>& matrix)
    {
        //
    }
}










