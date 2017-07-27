# Graham Scan(graham算法)

> graham 凸包扫描算法
> [WikiPedia Reference![](https://upload.wikimedia.org/wikipedia/commons/thumb/7/71/GrahamScanDemo.gif/200px-GrahamScanDemo.gif)](https://en.wikipedia.org/wiki/Graham_scan)

# 算法描述

## 算法描述：

1. 第一步是找出点集合y坐标点最小的，如果包含多个，就选择点x坐标最小的作为原点P
2. 第二步，将点集合按照与x轴的夹角递增排序
3. 计算每个点相对于原点P的倾斜角度（或者称之为极角）
4. 使用stack栈结构进行遍历计算：
  按照排序偶的顺序进行扫描，对于每个点，计算相对前两点是“向左”还是“向右”，来决定新的点是在凸包的内部还是外部
  判断方向的计算方法：使用**叉乘**的计算结果。假设三个点$ P_{1} = (x_{1}, y_{1}), P_{2} = (x_{2}, y_{2}), P_{3} = (x_{3}, y_{3}) $，这里有两个向量$ P_{1} and \ P_{2} \rightarrow \quad \overrightarrow{P_{1}P_{2}} \quad and \quad \overrightarrow {P_{1}P_{3}}$，计算两个向量的叉乘，$ (x_{2} - x_{1}) (y_{3} -  y_{1}) - (y_{2} - y_{1}) (x_{3} - x_{1})$，根据正负号来判断左转还是右转
5. 遍历所有点集，栈结构中剩余的就是组成凸包的点集合


## 代码演示

```
# Three points are a counter-clockwise turn if ccw > 0, clockwise if
# ccw < 0, and collinear if ccw = 0 because ccw is a determinant that
# gives twice the signed  area of the triangle formed by p1, p2 and p3.
function ccw(p1, p2, p3):
    return (p2.x - p1.x)*(p3.y - p1.y) - (p2.y - p1.y)*(p3.x - p1.x)
```

```
let N           = number of points
let points[N+1] = the array of points
swap points[1] with the point with the lowest y-coordinate
sort points by polar angle with points[1]

# We want points[0] to be a sentinel point that will stop the loop.
let points[0] = points[N]

# M will denote the number of points on the convex hull.
let M = 1
for i = 2 to N:
    # Find next valid point on convex hull.
    while ccw(points[M-1], points[M], points[i]) <= 0:
        if M > 1:
            M -= 1
            continue
        # All points are collinear
        else if i == N:
            break
        else
            i += 1

    # Update M and swap points[i] to the correct place.
    M += 1
    swap points[M] with points[i]
```

# 参考地址整理

[博客园参考资料](http://www.cnblogs.com/Booble/archive/2011/02/28/1967179.html)
[凸包扫描算法的原理](https://wz.cnblogs.com/my/?tag=%E5%87%B8%E5%8C%85)

> 尽管少写那么多代码，但省下来的时间又在哪里呢？

[CSDN参考地址](http://blog.csdn.net/cumtwyc/article/details/49387333)




