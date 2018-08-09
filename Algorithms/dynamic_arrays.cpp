

#include <iostream>
#include <algorithm>
#include <vector>

using namespace std;

int main(void)
{
	vector<int> v0;
	vector<int> v1(5, 0);

	vector<int> v2(v1.begin(), v1.end());
	vector<int> v3(v2);
	int a[5] = {0, 1, 2, 3, 4};
	vector<int> v4(a, *(&a + 1));  //这里不太懂是什么意思
	cout << "the size of v4 is : " << v4.size() << endl;
	for(int i = 0; i < v4.size(); i++)
	{
		cout << " " << v4[i];
	}
	for(int &item : v4)
	{
		cout << " " << item;
	}
	for(auto item = v4.begin(); item != v4.end(); item++)
	{
		cout << " " << *item;
	}
	cout << endl;

	v4[0] = 20;
	sort(v4.begin(), v4.end());

	v4.push_back(-1);
	v4.pop_back();
}



