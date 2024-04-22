#include <iostream>
#include <opencv2/opencv.hpp>

// Função para carregar a imagem
cv::Mat loadImage(const std::string& path) {
    cv::Mat image = cv::imread(path, cv::IMREAD_GRAYSCALE);
    if (!image.data) {
        std::cout << "Imagem não carregou corretamente\n";
        exit(-1);
    }
    return image;
}

// Função para eliminar bolhas que tocam as bordas
void eliminateBorders(cv::Mat& image) {
    cv::Point p;
    int width = image.cols;
    int height = image.rows;

    // Elimina bolhas que tocam bordas esquerda e direita
    for (int i = 0; i < height; i++) {
        if (image.at<uchar>(i, 0) == 255) {
            p.x = 0;
            p.y = i;
            cv::floodFill(image, p, 0);
        }
        if (image.at<uchar>(i, width - 1) == 255) {
            p.x = width - 1;
            p.y = i;
            cv::floodFill(image, p, 0);
        }
    }

    // Elimina bolhas que tocam bordas superior e inferior
    for (int j = 0; j < width; j++) {
        if (image.at<uchar>(0, j) == 255) {
            p.x = j;
            p.y = 0;
            cv::floodFill(image, p, 0);
        }
        if (image.at<uchar>(height - 1, j) == 255) {
            p.x = j;
            p.y = height - 1;
            cv::floodFill(image, p, 0);
        }
    }
}

// Função para contar bolhas com e sem buracos
void countBolhas(cv::Mat& image, int& nBolhasComBuracos, int& nBolhasSemBuracos) {
    cv::Point p;
    int width = image.cols;
    int height = image.rows;

    // Trocando cor de fundo da imagem
    p.x = 0;
    p.y = 0;
    cv::floodFill(image, p, 10);

    // Busca bolhas com buracos
    nBolhasComBuracos = 0;
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            if (image.at<uchar>(i, j) == 0) {
                if (image.at<uchar>(i - 1, j - 1) == 255) {
                    nBolhasComBuracos++;
                    p.x = j - 1;
                    p.y = i - 1;
                    cv::floodFill(image, p, 100);
                }
                p.x = j;
                p.y = i;
                cv::floodFill(image, p, 100);
            }
        }
    }

    // Busca bolhas restantes
    nBolhasSemBuracos = 0;
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            if (image.at<uchar>(i, j) == 255) {
                nBolhasSemBuracos++;
                p.x = j;
                p.y = i;
                cv::floodFill(image, p, 200);
            }
        }
    }
}

int main(int argc, char** argv) {
    cv::Mat image = loadImage("./bolhas.png");
    int width = image.cols;
    int height = image.rows;
    std::cout << width << "x" << height << std::endl;

    eliminateBorders(image);
    cv::imwrite("nova-img-bolhas.png", image);

    int nBolhasComBuracos = 0;
    int nBolhasSemBuracos = 0;
    countBolhas(image, nBolhasComBuracos, nBolhasSemBuracos);

    std::cout << "A figura tem " << nBolhasComBuracos + nBolhasSemBuracos << " bolhas na imagem, sendo elas divididas da seguinte forma: \n";
    std::cout << nBolhasComBuracos << " bolhas com buracos\n";
    std::cout << nBolhasSemBuracos << " bolhas sem buracos\n";

    cv::imshow("image", image);
    cv::imwrite("labelling.png", image);
    cv::waitKey();
    return 0;
}
