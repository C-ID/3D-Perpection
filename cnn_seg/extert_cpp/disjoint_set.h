
#ifndef DISJOINT_SET_H_
#define DISJOINT_SET_H_

namespace apollo {
namespace common {
namespace util {

template <class T>
void DisjointSetMakeSet(T *x) {
  x->parent = x;
  x->node_rank = 0;
}

template <class T>
T *DisjointSetFindRecursive(T *x) {
  if (x->parent != x) {
    x->parent = DisjointSetFindRecursive(x->parent);
  }
  return x->parent;
}

template <class T>
T *DisjointSetFind(T *x) {
  T *y = x->parent;
  if (y == x || y->parent == y) {
    return y;
  }
  T *root = DisjointSetFindRecursive(y->parent);
  x->parent = root;
  y->parent = root;
  return root;
}

template <class T>
void DisjointSetMerge(T *x, const T *y) {}

template <class T>
void DisjointSetUnion(T *x, T *y) {
  x = DisjointSetFind(x);
  y = DisjointSetFind(y);
  if (x == y) {
    return;
  }
  if (x->node_rank < y->node_rank) {
    x->parent = y;
    DisjointSetMerge(y, x);
  } else if (y->node_rank < x->node_rank) {
    y->parent = x;
    DisjointSetMerge(x, y);
  } else {
    y->parent = x;
    x->node_rank++;
    DisjointSetMerge(x, y);
  }
}

}  // namespace util
}  // namespace common
}  // namespace apollo

#endif  // MODULES_PERCEPTION_COMMON_UTIL_DISJOINT_SET_H_
