#include <opencv2/opencv.hpp>
//#include " opencv2 / nonfree / nonfree .hpp"
#include <vector>
#include <sys/types.h>
//#include <dirent.h>
#include <stdlib.h>
#include <filesystem>

namespace fs = std::filesystem;

///////////////////////////////////////////////////////////////////////////////
/// jitter blur corrector : replace the fusion area by patches in aligned images
/// @argv [1] , directory of aligned image folder
/// @argv [2] , name of the fusion image
/// @argv [3] , sigma of Gaussian distribution
/// @argv [4] , threshold of color distances
///////////////////////////////////////////////////////////////////////////////

int main(int, char * argv[])
{
    int sigma = atoi(argv[3]);
    int wsize = sigma * 4 + 1; // gaussian window size
    float thresh = atof(argv[4]);
    // read in images
    cv::Mat fusion = cv::imread(argv[2]);
    fusion.convertTo(fusion, CV_32FC3);
    cv::Mat fusion_blur;
    cv::GaussianBlur(fusion, fusion_blur, cv::Size(wsize, wsize), sigma, sigma, cv ::
        BORDER_REFLECT);
    std::vector <cv::Mat > aligned;
    //DIR* folder_dir = opendir(argv[1]);
    //struct dirent * file_info;
    std::vector <cv::Mat > similar; //’ difference ’ is marked as 0, ’similarity ’ is marked as 1.
        //while ((file_info = readdir(folder_dir)) != NULL)
    for (const auto & entry : fs::directory_iterator(argv[1]))
    {
        //if (strcmp(file_info->d_name, ".") != 0 && strcmp(file_info->d_name, "..") != 0)
        {
            //std::string dir_whole = argv[1];
            //dir_whole = dir_whole + "/" + file_info->d_name;
            std::string dir_whole = entry.path().string();
            cv::Mat img_ori = cv::imread(dir_whole);
            img_ori.convertTo(img_ori, CV_32FC3);
            aligned.push_back(img_ori);
            //** - - Step1 : blur images - -**//
            cv::Mat img_blur;
            cv::GaussianBlur(img_ori, img_blur, cv::Size(wsize, wsize), sigma, sigma, cv ::
                BORDER_REFLECT);
            //** - - Step2 : create distance cards - -**//
            cv::Mat vdist = img_blur - fusion_blur;
            cv::pow(vdist, 2, vdist);
            std::vector < cv::Mat > cdist(3);
            cv::split(vdist, cdist);
            cv::Mat ndist = cdist[0] + cdist[1] + cdist[2];
            cv::pow(ndist, 0.5, ndist);
            ndist.setTo(-1, ndist >= thresh);
            ndist.setTo(0, ndist != -1);
            ndist = ndist + 1;
            similar.push_back(ndist);
        }
    }
    cv::Mat index = cv::Mat::ones(aligned.size(), 1, CV_32FC1); //0: the image has been used
    cv::Mat left = cv::Mat::ones(fusion.size(), CV_32FC1);//1: area left to be corrected
    cv::Mat result(fusion.size(), CV_32FC3, cv::Scalar(0, 0, 0));
    std::vector < cv::Mat > cresult(3);
    cv::split(result, cresult);
    while (cv::sum(index)[0] != 0 && cv::sum(left)[0] != 0)
    {
        //** - - Step3 : find candidate aligned image - -**//
        int max_area = 0, best_choice = -1;
        for (int i = 0; i != similar.size(); ++i)
        {
            if (index.at <float >(i, 0) == 0)
                continue;
            int actual_area = cv::sum(similar[i].mul(left))[0];
            if (actual_area > max_area)
            {
                max_area = actual_area;
                best_choice = i;
            }
        }
        //** - - Step4 : fill in the final image - -**//
        cv::Mat mask = similar[best_choice].mul(left);
        std::vector < cv::Mat > caligned(3);
        cv::split(aligned[best_choice], caligned);
        cresult[0] = cresult[0] + caligned[0].mul(mask);
        cresult[1] = cresult[1] + caligned[1].mul(mask);
        cresult[2] = cresult[2] + caligned[2].mul(mask);
        //** - - Step5 : update index and area left to be corrected - -**//
        index.at <float >(best_choice, 0) = 0;
        left = left.mul(1 - mask);
    }
    cv::merge(cresult, result);
    result.convertTo(result, CV_8UC3);
    cv::imshow("result", result);
    cv::waitKey(0);
    return 0;
}
