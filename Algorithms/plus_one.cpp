

#include <iostream>
#include <algorithm>
#include <numeric>


class Solution
{
    //题目要求：给定一个数组表示的数字，在这个数字上加一，求取结果   难点在于加一后进位的运算
    public:
    vector<int> plusOne(vector<int> digits)
    {
        int n = digits.size();
        for(int i = n-1; i >= 0; i--)
        {
            if(digits[i] == 9)
            {
                digits[i] = 0;
            }
            else
            {
                digits[i] += 1;
                return digits;
            }
        }
        
        if(digits.front() == 0)
        {
            digits.insert(digits.begin(), 1);
        }
        return digits;
    }
}





