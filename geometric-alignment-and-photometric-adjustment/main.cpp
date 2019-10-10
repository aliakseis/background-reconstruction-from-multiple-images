#include <opencv2/opencv.hpp>
//#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <vector>
#include <sys/types.h>
//#include <dirent.h>
#include <algorithm>
#include <random>
#include <filesystem>

namespace fs = std::filesystem;

///////////////////////////////////////////////////////////////////////////////
/// Geometric and photometric adjustment : align images in the sequence
/// @argv [1] , directory of the folder of original images
///////////////////////////////////////////////////////////////////////////////

int main(int, char * argv[])
{
    try
    {
        std::vector < cv::Mat > img_res, img_geo, homo_inliers;
        std::vector < std::vector < cv::Point2f > > match_posi;
        std::vector<fs::path> filenames;

        {
            auto sift = cv::xfeatures2d::SIFT::create();
            cv::Mat img_scene, descriptors_scene;
            std::vector < cv::KeyPoint > keypoints_scene;

            cv::BFMatcher matcher;
            const float ratio = 0.8;

            //** - - Step1 : Geometric alignment - -**//
            //while ((file_info = readdir(folder_dir)) != NULL)
            for (const auto & entry : fs::directory_iterator(argv[1]))
            {
                //if (strcmp(file_info->d_name, ".") != 0 && strcmp(file_info->d_name, "..") != 0)
                {
                    //std::string dir_whole = argv[1];
                    //dir_whole = dir_whole + "/" + file_info->d_name;

                    filenames.push_back(entry.path().filename());

                    std::string dir_whole = entry.path().string();
                    if (img_res.empty())
                    {
                        img_res.push_back(cv::imread(dir_whole));
                        cv::cvtColor(img_res[0], img_scene, cv::COLOR_RGB2GRAY);
                        sift->detectAndCompute(img_scene, cv::noArray(), keypoints_scene, descriptors_scene);
                        img_res[0].convertTo(img_res[0], CV_32FC3);
                    }
                    else
                    {
                        cv::Mat img_object, descriptors_object;
                        cv::Mat img_ori = cv::imread(dir_whole);
                        cv::cvtColor(img_ori, img_object, cv::COLOR_RGB2GRAY);
                        img_ori.convertTo(img_ori, CV_32FC3);
                        // SIFT keypoints and descriptors //
                        std::vector < cv::KeyPoint > keypoints_object;
                        sift->detectAndCompute(img_object, cv::noArray(), keypoints_object, descriptors_object);
                        // Match descriptors using FLANN matcher //
                        std::vector < std::vector < cv::DMatch > > matches;
                        matcher.knnMatch(descriptors_object, descriptors_scene, matches, 2);
                        std::vector < cv::DMatch > good_matches;
                        for (size_t i = 0; i < matches.size(); ++i)
                        {
                            if (matches[i][0].distance < ratio * matches[i][1].distance)
                                good_matches.push_back(matches[i][0]);
                        }
                        // Get coordinates of matched pixels //
                        std::vector < cv::Point2f > obj, scene;
                        for (size_t i = 0; i < good_matches.size(); ++i)
                        {
                            obj.push_back(keypoints_object[good_matches[i].queryIdx].pt);
                            scene.push_back(keypoints_scene[good_matches[i].trainIdx].pt);
                        }
                        match_posi.push_back(scene);
                        // Compute homography //
                        cv::Mat inlier_mask;
                        cv::Mat H = findHomography(obj, scene, CV_RANSAC, 1, inlier_mask);
                        homo_inliers.push_back(inlier_mask);
                        // Bilinear interpolation //
                        std::vector < cv::Mat > img_ch, img_interp(3);
                        cv::split(img_ori, img_ch);
                        cv::warpPerspective(img_ch[0], img_interp[0], H, img_scene.size(), cv::INTER_LINEAR);
                        cv::warpPerspective(img_ch[1], img_interp[1], H, img_scene.size(), cv::INTER_LINEAR);
                        cv::warpPerspective(img_ch[2], img_interp[2], H, img_scene.size(), cv::INTER_LINEAR);
                        cv::Mat img_al;
                        cv::merge(img_interp, img_al);
                        img_geo.push_back(img_al);
                    }
                }
            }
        }


        //** - - Step2 : Chromatic adjustment - -**//
        std::vector < cv::Mat > colchannel_scene, colchannel_obj;
        cv::Mat pow_two, obj_all;
        for (size_t img_num = 0; img_num != img_geo.size(); ++img_num)
        {
            // Prepare intensity matrix //
            cv::Mat obj_col, scene_col;
            std::vector < int > rand_posi;
            int match_num = 0;
            for (int i = 0; i != homo_inliers[img_num].rows; ++i)
            {
                if (homo_inliers[img_num].at < uchar >(i) == 1)
                {
                    obj_col.push_back(img_geo[img_num].at < cv::Vec3f >((match_posi[img_num])[i]))
                        ;
                    scene_col.push_back(img_res[0].at < cv::Vec3f >((match_posi[img_num])[i]));
                    rand_posi.push_back(match_num);
                    ++match_num;
                }
            }
            std::random_device rd;
            std::mt19937 g(rd());
            std::shuffle(rand_posi.begin(), rand_posi.end(), g);
            cv::split(scene_col, colchannel_scene);
            cv::Mat all_one(obj_col.size(), CV_32FC3, cv::Scalar(1, 1, 1));
            cv::pow(obj_col, 2, pow_two);
            cv::hconcat(all_one, obj_col, obj_all);
            cv::hconcat(obj_all, pow_two, obj_all);
            cv::split(obj_all, colchannel_obj);
            // Calculate transfer model with RANSAC //
            int inliers_max = 0;
            std::vector < cv::Mat > final_model;
            for (size_t i = 0; i != (rand_posi.size() / 3); ++i)
            {
                cv::Mat inliers = cv::Mat::ones(colchannel_scene[0].size(), CV_8UC1);
                std::vector < cv::Mat > model(3);
                for (int j = 0; j != 3; ++j)
                {
                    cv::Mat ref = (cv::Mat_ < float >(3, 1)
                        << colchannel_scene[j].at < float >(rand_posi[3 * i]),
                        colchannel_scene[j].at < float >(rand_posi[3 * i + 1]),
                        colchannel_scene[j].at < float >(rand_posi[3 * i + 2]));
                    cv::Mat tar(3, 3, CV_32FC1);
                    colchannel_obj[j].row(rand_posi[3 * i]).copyTo(tar.row(0));
                    colchannel_obj[j].row(rand_posi[3 * i + 1]).copyTo(tar.row(1));
                    colchannel_obj[j].row(rand_posi[3 * i + 2]).copyTo(tar.row(2));
                    model[j] = tar.inv(cv::DECOMP_SVD)* ref;
                    // Search inliers //
                    cv::Mat check = (cv::abs(colchannel_obj[j] * model[j] - colchannel_scene[j])
                        <= 15) / 255;
                    cv::multiply(inliers, check, inliers);
                }
                int inliers_actual = cv::sum(inliers)[0];
                if (inliers_actual > inliers_max)
                {
                    final_model.assign(model.begin(), model.end());
                    inliers_max = inliers_actual;
                }
            }
            // Adjust image colors //
            std::vector < cv::Mat > color_ch(3), geo_channel(3);
            cv::split(img_geo[img_num], geo_channel);
            for (int j = 0; j != 3; ++j)
            {
                color_ch[j] = final_model[j].at < float >(0, 0) * cv::Mat::ones(geo_channel[j].size()
                    , CV_32FC1) +
                    final_model[j].at < float >(1, 0) * geo_channel[j];
                cv::Mat pow_interp;
                cv::pow(geo_channel[j], 2, pow_interp);
                color_ch[j] = color_ch[j] + final_model[j].at < float >(2, 0) * pow_interp;
            }
            cv::Mat img_colal;
            cv::merge(color_ch, img_colal);
            img_res.push_back(img_colal);
        }

        // save results
        for (size_t i = 0; i < img_res.size(); ++i)
        {
            std::string name_out = (fs::path(argv[2]) / filenames[i]).string();
            imwrite(name_out, img_res[i]);
        }

        //closedir(folder_dir);
        return 0;
    }
    catch (const std::exception& ex)
    {
        std::cerr << "Fatal: " << ex.what() << '\n';
        return 1;
    }
}
