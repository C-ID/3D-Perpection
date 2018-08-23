from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp cimport bool
from libcpp cimport float
from libcpp cimport int
from libc.stdlib cimport *
cimport cython


cdef extern from "cluster2d.cpp" namespace "apollo":
    cdef cppclass Cluster2D:
        Cluster2D() except +

        bool Init(int rows, int cols, float range)

        void Cluster(const float* category_pt_data,
               const float* instance_pt_x_data,
               const float* instance_pt_y_data,
               const float* pc_ptr,
               const int* valid_indices,
               const int cloud_size,
               float objectness_thresh, bool use_all_grids_for_clustering) except +MemoryError
        void Filter(const float* confidence_pt_data, const float* height_pt_data) except +MemoryError

        void Classify(const float* classify_pt_data) except +MemoryError

        void GetObjects(const float confidence_thresh, const float height_thresh, const int min_pts_num, const string path) except +MemoryError


cdef class PyCluster:
    cdef Cluster2D c_cluster
    def __cinit__(self):
        self.c_cluster = Cluster2D()

    def __dealloc__(self):
        pass

    def init(self, int rows, int cols, float range):
       return self.c_cluster.Init(rows, cols, range)

    def Cluster2Py(self,
               cat_p,
               ins_x_p,
               ins_y_p,
               pc_p,
               va_p,
               const int cloud_st,
               float objectness_tht, bool use_t):
        cdef:
            float* cat_pt = <float *> malloc(len(cat_p)*cython.sizeof(float))
            float* ins_x_pt = <float *> malloc(len(ins_x_p)*cython.sizeof(float))
            float* ins_y_pt = <float *> malloc(len(ins_y_p)*cython.sizeof(float))
            float* pc_pt = <float *> malloc(len(pc_p)*cython.sizeof(float))
            int* va_pt = <int *> malloc(len(va_p)*cython.sizeof(int))

        for i in range(len(cat_p)): cat_pt[i] = cat_p[i]
        for i in range(len(ins_x_p)): ins_x_pt[i] = ins_x_p[i]
        for i in range(len(ins_y_p)): ins_y_pt[i] = ins_y_p[i]
        for i in range(len(pc_p)): pc_pt[i] = pc_p[i]
        for i in range(len(va_p)): va_pt[i] = va_p[i]
        self.c_cluster.Cluster(cat_pt, ins_x_pt, ins_y_pt, pc_pt, va_pt,
                               cloud_st, objectness_tht, use_t)
        free(cat_pt)
        free(ins_x_pt)
        free(ins_y_pt)
        free(pc_pt)
        free(va_pt)

    def Filter2Py(self,
                  confidence_data,
                  height_data):

        cdef:
            float* confidence_p_data = <float *> malloc(len(confidence_data)*cython.sizeof(float))
            float* height_p_data = <float *> malloc(len(height_data)*cython.sizeof(float))

        for i in range(len(confidence_data)): confidence_p_data[i] = confidence_data[i]
        for i in range(len(height_data)): height_p_data[i] = height_data[i]
        self.c_cluster.Filter(confidence_p_data, height_p_data)
        free(confidence_p_data)
        free(height_p_data)

    def Classify2Py(self, classify_data):
        cdef:
            float* classify_p_data = <float *> malloc(len(classify_data)*cython.sizeof(float))
        for i in range(len(classify_data)): classify_p_data[i] = classify_data[i]
        self.c_cluster.Classify(classify_p_data)
        free(classify_p_data)

    def GetObjects2Py(self, float confidence_thresh, float height_thresh, int min_pts_num, string result_path):

        self.c_cluster.GetObjects(confidence_thresh, height_thresh, min_pts_num, result_path)


