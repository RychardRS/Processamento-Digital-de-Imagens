#include <iostream>
#include <opencv2/opencv.hpp>

int main(int, char**){
  cv::Mat image, image1, image2;
  cv::Vec3b val;

  image= cv::imread("biel.png",cv::IMREAD_GRAYSCALE);
  if(!image.data)
    std::cout << "nao abriu biel.png" << std::endl;

  cv::imshow("Original", image);

// -----------PRIMEIRA CÓPIA---------
  cv::Rect regiao1 = cv::Rect(0, 0,
  image.cols/2, image.rows/2);

  image(regiao1).copyTo(image1);

// --Movendo uma cópia do quarto quadrante para o primeiro
  image(cv::Rect(image.cols/2, image.rows/2,
  image.cols/2, image.rows/2)).copyTo(image(regiao1));

// ----------SEGUNDA CÓPIA--------
  cv::Rect regiao2 = cv::Rect(image.cols/2, 0,
   image.cols/2, image.rows/2);

  image(regiao2).copyTo(image2);

// --Movendo uma cópia do terceiro quadrante para o segundo
  image(cv::Rect(0, image.rows/2,
   image.cols/2, image.rows/2)).copyTo(image(regiao2));

// --Movendo a cópia do primeiro quadrante para o quarto
  image1.copyTo(image(cv::Rect(image.cols/2, image.rows/2,
  image.cols/2, image.rows/2)));

// --Movendo a cópia do segundo quadrante para o terceiro
  image2.copyTo(image(cv::Rect(0, image.rows/2,
   image.cols/2, image.rows/2)));

  cv::imshow("Troca regioes", image);
  cv::waitKey();
}