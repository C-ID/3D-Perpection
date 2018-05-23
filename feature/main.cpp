#include <iostream>
#include <fstream>
#include <string>
#include "feature_generator.h"
#include <memory>


using std::string;
using std::shared_ptr;
using apollo::perception::cnnseg::FeatureGenerator;



bool GetPointCloudFromBin(const string &bin_file, float* cloud, int& num) {
    //pcl::PointCloud<PointXYZIT> pre_ori_cloud;
    //cloud->points.reserve(pre_ori_cloud.points.size());

    std::ifstream in;

    in.open(bin_file, std::ios::in | std::ios::binary);
    string line;
    struct p{
        float x;
        float y;
        float z;
        float intensity;
    };
    p po;
    int count=0;
    num = 0;
    if(in.is_open()) {
        while (in.read((char *) &po, sizeof(po)))
        {
            *(cloud++) = po.x;
            *(cloud++) = po.y;
            *(cloud++) = po.z;
            *(cloud++) = po.intensity;
            count++;
//            std::cout << "x: "<<  point.x << " y: " << point.y << " z: " << point.z << " intensity: " << point.intensity << std::endl;
        }
        num = count*4;
        std::cout << "num of points: " << num << std::endl;
    }
    return true;
}


void start(const string &path)
{
    float* points = new float[500000];
    float* feature = new float[640*640*8];
    int num=0;
    if(GetPointCloudFromBin(path, points, num)) printf("all points loaded!!!");
    else printf("error occured!!!");
    shared_ptr<FeatureGenerator> featuregenerator;
    featuregenerator.reset(new FeatureGenerator());
    featuregenerator->Init();
    featuregenerator->Generate(points, num, feature);
    delete[] points;
    delete[] feature;
}

int main()
{
    string path = "/home/bai/Project/3D-Perpection/feature/test/0000000010.bin";
    start(path);
    return 0;
}