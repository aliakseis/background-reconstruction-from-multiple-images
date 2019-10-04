#include <opencv2/opencv.hpp>
#include <vector>
#include <math.h>
#include "detail.h"

//////////////////////////////////////////////////////////////////////////////
/// meaningful clique : select the pixels belonging to the most dense clouds
/// @argv [1] , directory of image folder
/// @argv [2] , thershold of variance
//////////////////////////////////////////////////////////////////////////////

int main(int, char * argv[])
{
    /* input images , convert them into float format */
    std::vector <cv::Mat > Input_image;
    size_t n_file = ReadImagesFromFolder(argv[1], Input_image);
    /* create pointers for original images */
    std::vector <cv::Mat_ <cv::Vec3f >::iterator > Pt_img;
    for (size_t i = 0; i != n_file; ++i)
        Pt_img.push_back(Input_image[i].begin <cv::Vec3f >());
    cv::Mat_ <cv::Vec3f >::iterator pt_end = Input_image[0].end <cv::Vec3f >();
    /* give an ascending order to the distance between every images */
    std::vector <cv::Mat > Image_dist;
    PixelDistance(Input_image, n_file, Image_dist);
    /* create pointers for distance array */
    std::vector <cv::Mat_ <float >::iterator > Pt_dist;
    for (size_t i = 0; i != Image_dist.size(); ++i)
        Pt_dist.push_back(Image_dist[i].begin <float >());
    /* create pointer for output image */
    cv::Mat Output(Input_image[0].rows, Input_image[0].cols, CV_32FC3);
    cv::Mat_ <cv::Vec3f >::iterator pt_out = Output.begin <cv::Vec3f >();
    /* traverse every pixel positions , select the cloud of pixels */
    while (Pt_img[0] != pt_end)
    {
        /* examine if there are more than 2 pixels */
        int num_valable = 0;
        for (size_t n_im = 0; n_im != n_file; ++n_im)
        {
            if (!isnan((*Pt_img[n_im])[0]))
            {
                *pt_out = *Pt_img[n_im];
                ++num_valable;
            }
        }
        if (num_valable > 1)
        {
            /* distance between every two images */
            std::vector <float > Member;
            for (size_t i = 0; i != Image_dist.size(); ++i)
                Member.push_back(*(Pt_dist[i]));
            /* get the distance order */
            cv::Mat matrix_order(n_file, n_file, CV_32FC1);
            getDistOrder(Member, n_file, matrix_order);
            /* get the candidate dense clouds */
            getCandiCloud(matrix_order, atof(argv[2]), Pt_img, pt_out);
        }
        else if (num_valable == 0)
            * pt_out = cv::Vec3f(255, 255, 255);
        /* pass to the next pixel position */
        for (size_t i = 0; i != n_file; ++i)
            ++Pt_img[i];
        for (size_t i = 0; i != Image_dist.size(); ++i)
            ++Pt_dist[i];
        ++pt_out;
    }
    /* save new image */
    std::string name_out = argv[1];
    name_out = name_out + "_clique.png";
    imwrite(name_out, Output);
    return 0;
}
