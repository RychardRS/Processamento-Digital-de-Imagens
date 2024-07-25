#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>

int main(int argc, char **argv){
  cv::Mat image;
  std::ofstream file;

  image = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);

  if (!image.data){
    std::cout << "Não foi possível abrir" << argv[1] << std::endl;
    return 0;
  }

  //aplicando threshold
  cv::threshold(image, image, 1, 255, cv::THRESH_BINARY + cv::THRESH_OTSU);

  std::vector<std::vector<cv::Point>> contornos;
  std::vector<cv::Vec4i> hierarchy;

  // abaixo temos 2 tipos de aproximação, onde uma gera um CSV extenso (NONE) e outra gera um CSV reduzido

  cv::findContours(image, contornos, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
  //cv::findContours(image, contornos, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

  cv::cvtColor(image, image, cv::COLOR_GRAY2BGR);

  file.open("contornos.svg");
  if (!file.is_open()){
    std::cout << "Não foi possível abrir contornos.svg" << std::endl;
    return 0;
  }

  file << "<svg height=\"" << image.rows << "\" width=\"" << image.cols
       << "\" xmlns=\"http://www.w3.org/2000/svg\">" << std::endl;

  //para realizar contagem de pontos
  int count = 0;

  for (size_t i = 0; i < contornos.size(); i++){
    file << "<path d=\"M " << contornos[i][0].x << " " << contornos[i][0].y << " ";
    for (size_t j = 1; j < contornos[i].size(); j++){
      file << "L" << contornos[i][j].x << " " << contornos[i][j].y << " ";
      count++;
    }
    file << "Z\" fill=\"#cccccc\" stroke=\"red\" stroke-width=\"1\" />" << std::endl;
    cv::drawContours(image, contornos, static_cast<int>(i), cv::Scalar(255, 0, 0));
  }

  std::cout << "Total de pontos de contornos: " << count << std::endl;

  file << "</svg>" << std::endl;
  file.close();

  cv::imshow("Imagem", image);
  cv::waitKey();

  return 0;
}