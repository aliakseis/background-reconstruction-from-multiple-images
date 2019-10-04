#pragma once

# ifndef MAJOR_STEPS
# define MAJOR_STEPS

# include <opencv2/opencv.hpp>
# include <vector>

void valueOrder(std::vector <float > &source, std::vector <int > & image_order);
size_t searchVecElemNum(size_t i, size_t j, size_t n_file);
float calculateVari(cv::Mat & matchan3);
void findMedian(cv::Mat &cloud, cv::Mat_ <cv::Vec3f >::iterator img_out);
size_t ReadImagesFromFolder(char * folder_name, std::vector <cv::Mat > & Input_Image);
void PixelDistance(std::vector <cv::Mat > & Image, size_t n_file, std::vector <cv::Mat > &
    Image_dist);
void getDistOrder(std::vector <float > &Member, size_t n_file, cv::Mat & order_mat);
void getCandiCloud(cv::Mat &order, float vari_thershold, std::vector <cv::Mat_ <cv::Vec3f >::
    iterator > & Pt_img, cv::Mat_ <cv::Vec3f >::iterator img_out);

# endif
