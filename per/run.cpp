
#include <iostream>
#include <string>
#include "cluster2d.h"
#include "pcl_types.h"
#include <pcl/io/pcd_io.h>
#include <pcl/io/io.h>
#include "cnn_segmentation.h"
#include <fstream>
#include <unordered_set>
#include <opencv2/opencv.hpp>



using apollo::perception::CNNSegmentation;
using apollo::perception::ObjectPtr;
using apollo::perception::pcl_util::PointCloud;
using apollo::perception::pcl_util::PointCloudPtr;
using apollo::perception::pcl_util::PointIndices;
using apollo::perception::pcl_util::PointXYZIT;
using apollo::perception::ObjectType;
using std::shared_ptr;
using std::string;
using std::vector;



bool GetPointCloudFromFile(const string &pcd_file, PointCloudPtr cloud) {
	pcl::PointCloud<PointXYZIT> pre_ori_cloud;
	if (pcl::io::loadPCDFile(pcd_file, pre_ori_cloud) < 0) {
		AERROR << "Failed to load pcd file: " << pcd_file;
		return false;
	}


	cloud->points.reserve(pre_ori_cloud.points.size());
    cloud->header.frame_id = "abc";
    cloud->header.stamp = 100;
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
	std::cout << "pcl read: " << cloud->size() << std::endl;
	return true;
}

bool GetPointCloudFromBin(const string &bin_file, PointCloudPtr cloud) {

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
    int count = 0;
    if(in.is_open()) {
        while (in.read((char *) &po, sizeof(po))) {
            apollo::perception::pcl_util::Point point;

            point.x = po.x;

            point.y = po.y;

            point.z = po.z;

            point.intensity = po.intensity;

            count++;
            std::cout << "x: "<<  point.x << " y: " << point.y << " z: " << point.z << " intensity: " << point.intensity << std::endl;

            cloud->push_back(point);

        }
        std::cout << "num of points: " << count << std::endl;
    }
    return true;
}


cv::Vec3b GetTypeColor(ObjectType type) {
    switch (type) {
        case ObjectType::PEDESTRIAN:
            return cv::Vec3b(255, 128, 128);  // pink
        case ObjectType::BICYCLE:
            return cv::Vec3b(0, 0, 255);  // blue
        case ObjectType::VEHICLE:
                return cv::Vec3b(0, 255, 0);  // green
        default:
            return cv::Vec3b(0, 255, 255);  // yellow
    }
}

struct CellStat {
    CellStat() : point_num(0), valid_point_num(0) {}
    int point_num;
    int valid_point_num;
};

int F2I(float val, float ori, float scale) {
    return static_cast<int>(std::floor((ori - val) * scale));
}

bool IsValidRowCol(int row, int rows, int col, int cols) {
    return row >= 0 && row < rows && col >= 0 && col < cols;
}

int RowCol2Grid(int row, int col, int cols) {
    return row * cols + col;
}

void DrawDetection(const PointCloudPtr &pc_ptr, const PointIndices &valid_idx,
                   int rows, int cols, float range,
                   const vector<ObjectPtr> &objects,
                   const string &result_file) {
    // create a new image for visualization
    cv::Mat img(rows, cols, CV_8UC3, cv::Scalar(0.0));

    // map points into bird-view grids
    float inv_res_x = 0.5 * static_cast<float>(cols) / range;
    float inv_res_y = 0.5 * static_cast<float>(rows) / range;
    int grids = rows * cols;
    vector<CellStat> view(grids);

    const std::vector<int> *valid_indices_in_pc = &(valid_idx.indices);
    CHECK_LE(valid_indices_in_pc->size(), pc_ptr->size());
    std::unordered_set<int> unique_indices;
    for (size_t i = 0; i < valid_indices_in_pc->size(); ++i) {
        int point_id = valid_indices_in_pc->at(i);
        CHECK(unique_indices.find(point_id) == unique_indices.end());
        unique_indices.insert(point_id);
    }

    for (size_t i = 0; i < pc_ptr->size(); ++i) {
        const auto &point = pc_ptr->points[i];
        // * the coordinates of x and y have been exchanged in feature generation
        // step,
        // so they should be swapped back here.
        int col = F2I(point.y, range, inv_res_x);  // col
        int row = F2I(point.x, range, inv_res_y);  // row
        if (IsValidRowCol(row, rows, col, cols)) {
            // get grid index and count point number for corresponding node
            int grid = RowCol2Grid(row, col, cols);
            view[grid].point_num++;
            if (unique_indices.find(i) != unique_indices.end()) {
                view[grid].valid_point_num++;
            }
        }
    }

    // show grids with grey color
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            int grid = RowCol2Grid(row, col, cols);
            if (view[grid].valid_point_num > 0) {
                img.at<cv::Vec3b>(row, col) = cv::Vec3b(127, 127, 127);
            } else if (view[grid].point_num > 0) {
                img.at<cv::Vec3b>(row, col) = cv::Vec3b(63, 63, 63);
            }
        }
    }

    // show segment grids with tight bounding box
    const cv::Vec3b segm_color(0, 0, 255);  // red

    for (size_t i = 0; i < objects.size(); ++i) {
        const ObjectPtr &obj = objects[i];
        CHECK_GT(obj->cloud->size(), 0);

        int x_min = INT_MAX;
        int y_min = INT_MAX;
        int x_max = INT_MIN;
        int y_max = INT_MIN;
        float score = obj->score;
        CHECK_GE(score, 0.0);
        CHECK_LE(score, 1.0);
        for (size_t j = 0; j < obj->cloud->size(); ++j) {
            const auto &point = obj->cloud->points[j];
            int col = F2I(point.y, range, inv_res_x);  // col
            int row = F2I(point.x, range, inv_res_y);  // row
            CHECK(IsValidRowCol(row, rows, col, cols));
            img.at<cv::Vec3b>(row, col) = segm_color * score;
            x_min = std::min(col, x_min);
            y_min = std::min(row, y_min);
            x_max = std::max(col, x_max);
            y_max = std::max(row, y_max);
        }

        // fillConvexPoly(img, list.data(), list.size(), cv::Scalar(positive_prob *
        // segm_color));
        cv::Vec3b bbox_color = GetTypeColor(obj->type);
        cv::rectangle(img, cv::Point(x_min, y_min), cv::Point(x_max, y_max),
                  cv::Scalar(bbox_color));

    }
    if (!cv::imwrite(result_file, img)) {
        return;
    }
    std::cout << "save done " << std::endl;
    /*
    // write image intensity values into file
    FILE *f_res;
    f_res = fopen(result_file.c_str(), "w");
    fprintf(f_res, "%d %d\n", rows, cols);
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            fprintf(f_res, "%u %u %u\n", img.at<cv::Vec3b>(row, col)[0],
                    img.at<cv::Vec3b>(row, col)[1], img.at<cv::Vec3b>(row, col)[2]);
        }
    }
    fclose(f_res);
    */
}


void start(const string &pcd_file, const string& json_path, const string& png_path)
{
	PointCloudPtr in_pc;
	in_pc.reset(new PointCloud());
	if(GetPointCloudFromBin(pcd_file, in_pc)) printf("load pcd file successed!!!\n");
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
	//for (int i = 0; i < 10; ++i)
	cnn_segmentor_->Segment(in_pc, valid_idx, &out_objects);
    DrawDetection(in_pc, valid_idx, cnn_segmentor_->height(),
                  cnn_segmentor_->width(), cnn_segmentor_->range(), out_objects,
                  png_path);
	//cnn_segmentor_->Write2Json(out_objects, json_path);
}

int main(int argc, char* argv[])
{
    string pcd_path = "/home/bai/hobot_pcd/1524038206.297796000.pcd";
    PointCloudPtr in_pc;
    in_pc.reset(new PointCloud());
    GetPointCloudFromFile(pcd_path, in_pc);
	//start(argv[1], argv[2], argv[3]);
	return 0;
}


