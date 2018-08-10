

#include <iostream>
#include <algorithm>

class Solution
{
    public:
    int dominantIndex(vector<int> nums)
    {
        //我自己的思路是直接找出两个相对大的数，
        //然后直接判断这两个的大小。缺点是在数量少于3个时会出错

        //这个的思路是遍历两遍，第一遍找出最大值，第二遍判断数值大小

        int maxNum=0;//最大数
        int maxIndex;//最大数的下标
        for(int i=0; i < nums.size(); i++) {
            if(nums[i] >= maxNum) {//记录最大值和下标
                maxNum = nums[i];
                maxIndex = i;
            }
        }

        for(int i=0; i < nums.size(); i++) {
            if(i!=maxIndex && nums[maxIndex] < 2*nums[i]) {
                return -1;
            }
        }
        return maxIndex;
    }
}






