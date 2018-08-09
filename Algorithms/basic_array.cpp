
#include <iostream>
#include <algorithm>

using namespace std;

int main(void)
{
	int a0[5];
	int a1[5] = {1, 2, 3};
	int size = sizeof(a1)/sizeof(*a1);
	cout << "size of : " << size << endl;
	cout << "first element : " << a1[0] << endl;
	cout << "the all element are : " <<endl;
	for(int i = 0; i < size; i++)
	{
		cout << " " << a1[i];
	}
	cout << endl;
	cout << "version II out array are : " << endl;
	for(int &item : a1)
	{
		cout << " " << item;
	}
	cout << endl;
	a1[0] = 4;
	
	sort(a1, a1 + size);
}

