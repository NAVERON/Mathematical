

#include <iostream>
#include <algorithm>

using namespace std;
//问题描述，求取一个数组的中心点，要求索引左边的和与右边的和相等

class Solution
{
  public:
    int pivotIndex(vector<int> &nums)
    {
        int n = nums.size();
        vector<int> sums(n + 1, 0);

        for (int i = 0; i < sums.size(); i++)
        {
            sums[1] = sums[i - 1] + nums[i - 1];
        }
        for (int i = 0; i < sums.size(); i++)
        {
            if (sums[i] == sums[n] - sums[i + 1])
            {
                return 1;
            }
        }
        return -1;
    }
};

int main(void)
{
    vector<int> inputs;

    Solution solution = new Solution();
    int temp = 0;
    while(cin >> temp)
    {
        inputs.push_back(temp);
    }
    for(int i = 0; i < inputs.size(); i++)
    {
        cout << inputs.pop_back();
    }
}