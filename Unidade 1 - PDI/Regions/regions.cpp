#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cout << "quantidade de argumentos invalida.\n";
        return 0;
    }

    cv::Mat image = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        std::cout << "Abertura de imagem falhou.\n";
        return 0;
    }

    int p1[2], p2[2];
    std::string p1char, p2char;

    std::cout << "Insira dois pontos no formato x y representando as coordenadas do retangulo a ser invertido.\nInsira o ponto 1: ";
    std::cin >> p1[0] >> p1[1];

    if (p1[0] < 0 || p1[0] >= image.rows || p1[1] < 0 || p1[1] >= image.cols) {
        std::cout << "os pontos especificados estao fora do escopo da imagem.\n";
        return 0;
    }

    std::cout << "Insira o ponto 2: ";
    std::cin >> p2[0] >> p2[1];

    if (p2[0] < 0 || p2[0] >= image.rows || p2[1] < 0 || p2[1] >= image.cols) {
        std::cout << "os pontos especificados estao fora do escopo da imagem.\n";
        return 0;
    }

    for (int i = std::min(p1[0], p2[0]); i < std::max(p1[0], p2[0]); i++) {
        for (int j = std::min(p1[1], p2[1]); j < std::max(p1[1], p2[1]); j++) {
            image.at<uchar>(i, j) = 255 - image.at<uchar>(i, j);
        }
    }

    cv::namedWindow("imagem processada", cv::WINDOW_NORMAL);
    cv::imshow("imagem processada", image);
    cv::waitKey();

    return 0;
}
