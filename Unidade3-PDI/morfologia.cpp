#include <iostream>
#include <opencv2/opencv.hpp>

int main(int argc, char** argv) {
  cv::Mat digitos1, dilatacao1, erosao1;
  cv::Mat digitos2, dilatacao2, erosao2;
  cv::Mat digitos3, dilatacao3, erosao3;
  cv::Mat digitos4, dilatacao4, erosao4;
  cv::Mat digitos5, dilatacao5, erosao5;
  cv::Mat str, image;

  digitos1 = cv::imread("../digitos-1.png", cv::IMREAD_UNCHANGED);
  digitos2 = cv::imread("../digitos-2.png", cv::IMREAD_UNCHANGED);
  digitos3 = cv::imread("../digitos-3.png", cv::IMREAD_UNCHANGED);
  digitos4 = cv::imread("../digitos-4.png", cv::IMREAD_UNCHANGED);
  digitos5 = cv::imread("../digitos-5.png", cv::IMREAD_UNCHANGED);

    if(digitos1.empty()){
        std::cout << "Não foi possível carregar a imagem digito 1" << std::endl;
        return -1;
    }

    if(digitos2.empty()){
        std::cout << "Não foi possível carregar a imagem digito 2" << std::endl;
        return -1;
    }

    if(digitos3.empty()){
        std::cout << "Não foi possível carregar a imagem digito 3" << std::endl;
        return -1;
    }

    if(digitos4.empty()){
        std::cout << "Não foi possível carregar a imagem digito 4" << std::endl;
        return -1;
    }

  cv::bitwise_not(digitos1, digitos1);
  cv::bitwise_not(digitos2, digitos2);
  cv::bitwise_not(digitos3, digitos3);
  cv::bitwise_not(digitos4, digitos4);
  cv::bitwise_not(digitos5, digitos5);

  str = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 15));

  cv::dilate(digitos1, dilatacao1, str);
  cv::erode(dilatacao1, erosao1, str);

  cv::dilate(digitos2, dilatacao2, str);
  cv::erode(dilatacao2, erosao2, str);

  cv::dilate(digitos3, dilatacao3, str);
  cv::erode(dilatacao3, erosao3, str);

  cv::dilate(digitos4, dilatacao4, str);
  cv::erode(dilatacao4, erosao4, str);

  cv::dilate(digitos5, dilatacao5, str);
  cv::erode(dilatacao5, erosao5, str);

  cv::bitwise_not(erosao1, erosao1);
  cv::bitwise_not(erosao2, erosao2);
  cv::bitwise_not(erosao3, erosao3);
  cv::bitwise_not(erosao4, erosao4);
  cv::bitwise_not(erosao5, erosao5);

  cv::imwrite("morfologia-digito-1.png", erosao1);
  cv::imwrite("morfologia-digito-2.png", erosao2);
  cv::imwrite("morfologia-digito-3.png", erosao3);
  cv::imwrite("morfologia-digito-4.png", erosao4);
  cv::imwrite("morfologia-digito-5.png", erosao5);

  cv::waitKey();
  return 0;
}