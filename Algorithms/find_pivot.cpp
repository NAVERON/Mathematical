

#include <iostream>
#include <algorithm>
#include <numeric>

using namespace std;
//问题描述，求取一个数组的中心点，要求索引左边的和与右边的和相等

class Solution
{
  public:
    int pivotIndex(vector<int> &nums)  //nums表示输入的数组
    {
        int sum = accumulate(nums.begin(), nums.end(), 0);
        int curSum = 0, n = nums.size();
        for (int i = 0; i < n; ++i) {
            if (sum - nums[i] == 2 * curSum) return i;
            curSum += nums[i];
        }
        return -1;
    }
};

int main(void)
{
    vector<int> inputs;

    Solution solution;
    Solution *solution2 = new Solution();
    int temp = 0;
    while(cin >> temp)
    {
        inputs.push_back(temp);
    }

    int index = solution.pivotIndex(inputs);
    if(index != -1)
    {
        cout << " has pivot index" << index;
    }
    else
    {
        cout << " not has pivot";
    }
    cout << endl;

    for(int i = 0; i < inputs.size(); i++)
    {
        inputs.pop_back();
    }
}



//officle solution for above problem
/*
void trimLeftTrailingSpaces(string &input) {
    input.erase(input.begin(), find_if(input.begin(), input.end(), [](int ch) {
        return !isspace(ch);
    }));
}

void trimRightTrailingSpaces(string &input) {
    input.erase(find_if(input.rbegin(), input.rend(), [](int ch) {
        return !isspace(ch);
    }).base(), input.end());
}

vector<int> stringToIntegerVector(string input) {
    vector<int> output;
    trimLeftTrailingSpaces(input);
    trimRightTrailingSpaces(input);
    input = input.substr(1, input.length() - 2);
    stringstream ss;
    ss.str(input);
    string item;
    char delim = ',';
    while (getline(ss, item, delim)) {
        output.push_back(stoi(item));
    }
    return output;
}

int main() {
    string line;
    while (getline(cin, line)) {
        vector<int> nums = stringToIntegerVector(line);
        
        int ret = Solution().pivotIndex(nums);

        string out = to_string(ret);
        cout << out << endl;
    }
    return 0;
}
*/






