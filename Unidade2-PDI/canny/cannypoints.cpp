#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <opencv2/opencv.hpp>
#include <vector>

#define STEP 7
#define JITTER 3
#define RAIO 6

int main(int argc, char** argv) {
    std::vector<int> yrange;
    std::vector<int> xrange;

    cv::Mat image, frame, points;

    int width, height;
    int x, y;

    image = cv::imread(argv[1], cv::IMREAD_COLOR);

    std::srand(std::time(0));

    if (image.empty()) {
        std::cout << "Não foi encontrada a imagem" << std::endl;
        return -1;
    }

    width = image.cols;
    height = image.rows;

    xrange.resize(height / STEP);
    yrange.resize(width / STEP);

    std::iota(xrange.begin(), xrange.end(), 0);
    std::iota(yrange.begin(), yrange.end(), 0);

    for (uint i = 0; i < xrange.size(); i++) {
        xrange[i] = xrange[i] * STEP + STEP / 2;
    }

    for (uint i = 0; i < yrange.size(); i++) {
        yrange[i] = yrange[i] * STEP + STEP / 2;
    }

    points = cv::Mat(height, width, CV_8UC3, cv::Vec3b(255,255,255));

    std::random_shuffle(xrange.begin(), xrange.end());

    for (auto i : xrange) {
        std::random_shuffle(yrange.begin(), yrange.end());
        for (auto j : yrange) {
            x = i + std::rand() % (2 * JITTER) - JITTER + 1;
            y = j + std::rand() % (2 * JITTER) - JITTER + 1;
            cv::circle(points, cv::Point(y, x), RAIO, image.at<cv::Vec3b>(x, y),
                       cv::FILLED, cv::LINE_AA);
        }
    }

    cv::Mat cny;
    cv::Canny(image, cny, 67, 133);

    for (int i = 0; i < image.rows; i++) {
        for (int j = 0; j < image.cols; j++) {
            if (cny.at<uchar>(i, j) == 255) {
                cv::circle(points, cv::Point(j, i), RAIO/4, cv::Scalar(0),
                       cv::FILLED, cv::LINE_AA);
            }
        }
    }

    cv::imwrite("pontos.jpg", points);
    return 0;
}