
#include <iostream>
#include <string>
#include "cluster2d.h"
//#include "disjoint_set.h"
#include "pcl_types.h"
#include <pcl/io/pcd_io.h>
#include <pcl/io/io.h>
#include "cnn_segmentation.h"
#include <pcl/visualization/cloud_viewer.h>
//#include "types.h"

using apollo::perception::CNNSegmentation;
using apollo::perception::ObjectPtr;
using apollo::perception::pcl_util::PointCloud;
using apollo::perception::pcl_util::PointCloudPtr;
using apollo::perception::pcl_util::PointIndices;
using apollo::perception::pcl_util::PointXYZIT;
//using apollo::perception::SegmentationOptions;
using std::shared_ptr;
using std::string;
//using std::unordered_set;
using std::vector;
//using pcl::visualization::CloudViewer viewer("Cloud Viewer")
//uint8_t r(255), g(128), b(128);
int user_data;
const string pcd_file = "/home/bai/Project/perception/dataset/uscar_12_1470770225_1470770492_1349.pcd";



void viewerOneOff (pcl::visualization::PCLVisualizer& viewer)
{
    viewer.setBackgroundColor (0, 0, 0);  
}

void viewerPsycho (pcl::visualization::PCLVisualizer& viewer)
{
    static unsigned count = 0;
    std::stringstream ss;
    ss << "Once per viewer loop: " << count++;
    viewer.removeShape ("text", 0);
    viewer.addText (ss.str(), 200, 300, "text", 0);
    
    //FIXME: possible race condition here:
    user_data++;
}

bool GetPointCloudFromFile(const string &pcd_file, PointCloudPtr cloud) {
	pcl::PointCloud<PointXYZIT> pre_ori_cloud;
	if (pcl::io::loadPCDFile(pcd_file, pre_ori_cloud) < 0) {
		AERROR << "Failed to load pcd file: " << pcd_file;
		return false;
	}
	cloud->points.reserve(pre_ori_cloud.points.size());
	for (size_t i = 0; i < pre_ori_cloud.points.size(); ++i) {
		apollo::perception::pcl_util::Point point;
		point.x = pre_ori_cloud.points[i].x;
		point.y = pre_ori_cloud.points[i].y;
		point.z = pre_ori_cloud.points[i].z;
		point.intensity = pre_ori_cloud.points[i].intensity;
		if (std::isnan(pre_ori_cloud.points[i].x)) {
			continue;
		}
		cloud->push_back(point);
	}

	return true;
}


void show_result(const string &pcd_file)
{
	pcl::PointCloud<pcl::PointXYZ>::Ptr ori_cloud(new pcl::PointCloud<pcl::PointXYZ>);
	if(pcl::io::loadPCDFile(pcd_file, *ori_cloud)<0)
	{
		printf("well done!");
	}
	else
	{
	printf("ok\n");
	}
	pcl::visualization::CloudViewer viewer("Cloud Viewer");
	viewer.showCloud(ori_cloud);
	viewer.runOnVisualizationThreadOnce (viewerOneOff);
	while (!viewer.wasStopped ())
    {
    	
    }
}
/*
uint8_t GetTypeColor(ObjectType type) {

  switch (type) {
    case ObjectType::PEDESTRIAN:
      return uint8_t r(255), g(128), b(128);  // pink
    case ObjectType::BICYCLE:
      return uint8_t r(0), g(0), b(255);  // blue
    case ObjectType::VEHICLE:
      return uint8_t r(0), g(255), b(0);  // green
    default:
      return uint8_t r(0), g(255), b(255);  // yellow
  }
}
*/

void start()
{
	PointCloudPtr in_pc;
	in_pc.reset(new PointCloud());
	if(GetPointCloudFromFile(pcd_file, in_pc)) printf("load pcd file successed!!!\n");
	else printf("failed to load pcd file!!!\n");
	
	PointIndices valid_idx;
	auto &indices = valid_idx.indices;
	indices.resize(in_pc->size());
	std::iota(indices.begin(), indices.end(), 0);
	std::vector<ObjectPtr> out_objects;

	
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr show_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
	shared_ptr<CNNSegmentation> cnn_segmentor_;
	cnn_segmentor_.reset(new CNNSegmentation());
	cnn_segmentor_->Init();
	for (int i = 0; i < 10; ++i)
		cnn_segmentor_->Segment(in_pc, valid_idx, &out_objects);
	cnn_segmentor_->Write2Json(out_objects);
	/*
	uint8_t r(255), g(128), b(128);
	for(int i = 0; i<out_objects.size(); ++i)
	{
		const ObjectPtr &obj = out_objects[i];
		for(int j = 0; j<obj->cloud->size(); j++)
		{
			const auto &po = obj->cloud->points[j];
			pcl::PointXYZRGB point;
			point.x = po.x;
			point.y = po.y;
			point.z = po.z;
			uint32_t rgb = (static_cast<uint32_t>(r) << 16 | static_cast<uint32_t>(g) << 8 | static_cast<uint32_t>(b));
			point.rgb = *reinterpret_cast<float*>(&rgb);
			show_cloud->points.push_back(point);
		}
	}
	pcl::visualization::CloudViewer viewer2("Cloud Viewer");
	viewer2.showCloud(show_cloud);
	viewer2.runOnVisualizationThreadOnce (viewerOneOff);
	while (!viewer2.wasStopped ())
	{
		
	}
	*/
	printf("well done! all process completed...\n");
}

int main()
{
	//show_result(pcd_file);
	start();

	return 0;
}