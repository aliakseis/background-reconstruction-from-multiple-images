#include <sys/types.h>
//# include <dirent.h>
#include <string>
#include <vector>
#include <algorithm>
#include <math.h>
#include <opencv2/opencv.hpp>
#include <filesystem>

namespace fs = std::filesystem;

size_t searchVecElemNum(size_t i, size_t j, size_t n_file)
{
    return j - 1 + i * (n_file - 1) - i * (i + 1) / 2;
}

void valueOrder(std::vector <float > &source, std::vector <int > & image_order)
{
    std::vector <float > temporary(1, INFINITY);
    int accumulator = 0;
    for (std::vector <float >::size_type src_num = 0; src_num != source.size(); ++src_num)
    {
        std::vector <int >::iterator iter_ord = image_order.begin();
        for (std::vector <float >::iterator iter_tem = temporary.begin(); iter_tem != temporary.end
        (); ++iter_tem)
        {
            if (source[src_num] >= *iter_tem)
                ++iter_ord;
            else
            {
                temporary.insert(iter_tem, source[src_num]);
                break;
            }
        }
        image_order.insert(iter_ord, accumulator);
        ++accumulator;
    }
}

float calculateVari(cv::Mat & matchan3)
{
    cv::Mat matrix;
    matchan3.copyTo(matrix);
    float px_num = matrix.rows;
    cv::Scalar mean = cv::mean(matrix);
    matrix -= mean;
    cv::pow(matrix, 2, matrix);
    cv::Scalar somme = cv::sum(matrix);
    float all = somme[0] + somme[1] + somme[2];
    all /= px_num;
    return sqrt(all);
}

/**
* @brief find the median of a cloud of RGB vectors
* @param cloud , Mat :: CV_32FC3 , a column of RGB vectors of the best cloud
* @param img_out , Mat_ <Vec3f >:: iterator , save the median RGB vector of the cloud
* @return none
*/
void findMedian(cv::Mat &cloud, cv::Mat_ <cv::Vec3f >::iterator img_out)
{
    float dist_min = INFINITY;
    for (int i = 0; i != cloud.rows; ++i)
    {
        cv::Mat member;
        cloud.copyTo(member);
        cv::Mat center(member.rows, 1, CV_32FC3, cv::Scalar(member.at <cv::Vec3f >(i, 0)[0],
            member.at <cv::Vec3f >(i, 0)[1], member.at <cv::Vec3f >(i, 0)[2]));
        member -= center;
        cv::pow(member, 2, member);
        std::vector < cv::Mat > channel(3);
        cv::split(member, channel);
        cv::Mat somme = channel[0] + channel[1] + channel[2];
        cv::pow(somme, 0.5, somme);
        cv::Scalar dist = cv::sum(somme);
        if (dist[0] < dist_min)
        {
            dist_min = dist[0];
            *img_out = cloud.at <cv::Vec3f >(i, 0);
        }
    }
}

/**
* @brief read several image files in a folder , save them into a vector of Mat
* @param Folder_name , char *, image folder
* @param Input_Image , vector of Mat :: CV_32FC3
* @return number of files
*/
size_t ReadImagesFromFolder(char * folder_name, std::vector <cv::Mat > & Input_Image)
{
    //DIR* folder_dir = opendir(folder_name);
    //struct dirent * file_info;
    //while ((file_info = readdir(folder_dir)) != NULL)
    for (const auto & entry : fs::directory_iterator(folder_name))
    {
        //if (strcmp(file_info->d_name, ".") != 0 && strcmp(file_info->d_name, "..") != 0)
        {
            //std::string whole_dir = folder_name;
            //whole_dir = whole_dir + "/" + file_info->d_name;
            std::string whole_dir = entry.path().string();
            cv::Mat Image = cv::imread(whole_dir);
            Image.convertTo(Image, CV_32FC3);
            cv::Mat border_mask;
            cv::inRange(Image, cv::Scalar(255, 255, 255), cv::Scalar(255, 255, 255), border_mask);
            Image.setTo(cv::Scalar(NAN, NAN, NAN), border_mask);
            Input_Image.push_back(Image);
        }
    }
    //closedir(folder_dir);
    return Input_Image.size();
}

/**
* @brief calculate matrix of euclidean distance between RGB vectors at the same position
of two images
* @param Image , vector of Mat :: CV_32FC3 , Input images , already defined
* @param n_file , unsigned , number of images
* @param Image_dist , vector of Mat :: CV_32FC1 , matrix of euclidean distance , defined here
such matrix between
* image i and image j (i<j) is saved at position j -1+i*( n_file -1) -i*(i +1) /2. the search of
position in
* Image_dist is achieved by searchVecElemNum in detail_preparation . cpp
* @return none
*/
void PixelDistance(std::vector <cv::Mat > &Image, size_t n_file, std::vector <cv::Mat > &
    Image_dist)
{
    for (size_t i = 0; i < n_file - 1; ++i)
    {
        for (size_t j = i + 1; j < n_file; ++j)
        {
            std::vector <cv::Mat > Image_channel;
            cv::Mat image_diff = Image[i] - Image[j];
            cv::pow(image_diff, 2, image_diff);
            cv::split(image_diff, Image_channel);
            cv::Mat dist_ij = Image_channel[0] + Image_channel[1] + Image_channel[2];
            cv::pow(dist_ij, 0.5, dist_ij);
            Image_dist.push_back(dist_ij);
        }
    }
}

/**
* @brief sort the distance of vector of images to the vector of a certain image , save
orderly
* the image numbers as a row of matrix
* @param Member , vector of float , euclidean distance of two RGB vectors of every two
images at a certain pixel
* position
* @param n_file , unsigned , number of images
* @param order_mat , n_file * n_file Mat of CV_32FC1 , map of distance order whose row save
the image numbers on
* ascending order , already defined
* @return none
*/
void getDistOrder(std::vector <float > &Member, size_t n_file, cv::Mat & order_mat)
{
    for (size_t i = 0; i < n_file; ++i)
    {
        /* pixel distances to pixel i */
        std::vector <float > dist_to_i;
        for (size_t j = 0; j < n_file; ++j)
        {
            if (j == i)
                dist_to_i.push_back(0);
            else if (i < j)
                dist_to_i.push_back(Member[searchVecElemNum(i, j, n_file)]);
            else
                dist_to_i.push_back(Member[searchVecElemNum(j, i, n_file)]);
        }
        /* sort members in ascending order , return a map of image numbers */
        std::vector <int > vec_order;
        valueOrder(dist_to_i, vec_order);
        for (size_t k = 0; k < n_file; ++k)
            order_mat.at <float >(i, k) = vec_order[k];
    }
}

/**
* @brief get one or several clouds of RGB vectors sharing the similar values . for each
cloud of pixels , save
* the number of their belonging images as a vector
Appendix Source Code 143
* @param order , Mat :: CV_32FC1 , map of pixel distances
* @param storage_cloud , vector of float vector , save image numbers of each cloud
* @return none
*/
void getCandiCloud(cv::Mat &order, float vari_thershold, std::vector <cv::Mat_ <cv::Vec3f >::
    iterator > & Pt_img, cv::Mat_ <cv::Vec3f >::iterator img_out)
{
    const size_t size_mini(2), size_maxi(order.rows);
    std::vector < std::vector <float > > storage_cloud;
    /* increase the size of cloud by iteration */
    for (size_t size_current = size_mini; size_current <= size_maxi; ++size_current)
    {
        std::vector < std::vector <float > > actual_cloud;
        /* select each time a pixel to find possible dense cloud */
        for (size_t center = 0; center != size_maxi; ++center)
        {
            /* select the members most approche to the center */
            cv::Mat center_mat;
            order(cv::Rect(0, center, size_current, 1)).copyTo(center_mat);
            std::set <float > center_set(center_mat.begin <float >(), center_mat.end <float >());
            /* initialize the first member set as center set */
            std::set <float > member_set1(center_set.begin(), center_set.end());
            /* calculate the intersection of two member sets */
            bool isFinish = 0;
            for (size_t mem_num = 1; mem_num != size_current; ++mem_num)
            {
                size_t row_num = center_mat.at <float >(0, mem_num);
                cv::Mat member_mat;
                order(cv::Rect(0, row_num, size_current, 1)).copyTo(member_mat);
                std::set <float > member_set2(member_mat.begin <float >(), member_mat.end <float
                >());
                std::set <float > intersect;
                std::set_intersection(member_set1.begin(), member_set1.end(), member_set2 .
                    begin(), member_set2.end(), inserter(intersect, intersect.begin()));
                /* analyze the result , prepare for the next iteration */
                if (intersect.size() != size_current)
                    break;
                else
                {
                    member_set1.clear();
                    member_set1.insert(intersect.begin(), intersect.end());
                }
                /* ensure the iteration has been accomplished */
                if (mem_num == (size_current - 1))
                    isFinish = 1;
            }
            /* save a candidate result */
            if (isFinish)
            {
                std::vector <float > candidate_cloud;
                candidate_cloud.insert(candidate_cloud.begin(), member_set1.begin(),
                    member_set1.end());
                std::vector < std::vector <float > >::iterator repetation = std::find(
                    actual_cloud.begin(), actual_cloud.end(), candidate_cloud);
                if (repetation == actual_cloud.end())
                    actual_cloud.push_back(candidate_cloud);
            }
        }
        /* examine the number clouds in storage */
        size_t cloud_size = actual_cloud.size();
        if (cloud_size > 1) // >1: continue searching ; update storage
        {
            storage_cloud.clear();
            storage_cloud = actual_cloud;
        }
        else if (cloud_size == 1) // 1: maybe find target cloud ; examine variance
        {
            cv::Mat single_cloud(size_current, 1, CV_32FC3);
            for (size_t candi_num = 0; candi_num != size_current; ++candi_num)
            {
                size_t img_num = (actual_cloud[0])[candi_num];
                single_cloud.at <cv::Vec3f >(candi_num, 0) = *(Pt_img[img_num]);
            }
            float vari = calculateVari(single_cloud);
            if (vari < vari_thershold || size_current == 2)
            {
                findMedian(single_cloud, img_out);
                break;
            }
            else
                cloud_size = 0;
        }
        if (cloud_size == 0) // 0: select a cloud among previous result
        {
            /* search the values of pixels in different clouds */
            cv::Mat cloud_vari_mini;
            float vari_mini(INFINITY);
            size_t size_previous = size_current - 1;
            for (size_t cloud_num = 0; cloud_num != storage_cloud.size(); ++cloud_num)
            {
                cv::Mat single_cloud(size_previous, 1, CV_32FC3);
                for (size_t ele_num = 0; ele_num != size_previous; ++ele_num)
                {
                    size_t img_num = (storage_cloud[cloud_num])[ele_num];
                    single_cloud.at <cv::Vec3f >(ele_num, 0) = *(Pt_img[img_num]);
                }
                float vari = calculateVari(single_cloud);
                if (vari < vari_mini)
                {
                    vari_mini = vari;
                    cloud_vari_mini = cv::Mat();
                    single_cloud.copyTo(cloud_vari_mini);
                }
            }
            findMedian(cloud_vari_mini, img_out);
            break;
        }
    }
}
