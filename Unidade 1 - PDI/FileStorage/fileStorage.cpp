#include <iostream>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <string>

constexpr int SIDE = 256;
constexpr int PERIODOS[2] = {8, 4};

// Função para criar e salvar uma imagem senoidal em formato YAML
void createAndSaveSineImage(const std::string& filename, cv::Mat& image, int period) {
    image = cv::Mat::zeros(SIDE, SIDE, CV_32FC1);

    for (int i = 0; i < SIDE; i++) {
        for (int j = 0; j < SIDE; j++) {
            image.at<float>(i, j) = 127 * sin(2 * M_PI * period * j / SIDE) + 128;
        }
    }

    cv::FileStorage fs(filename, cv::FileStorage::WRITE);
    fs << "mat" << image;
    fs.release();
}

// Função para carregar uma imagem de um arquivo YAML
void loadImageFromYAML(const std::string& filename, cv::Mat& image) {
    cv::FileStorage fs(filename, cv::FileStorage::READ);
    fs["mat"] >> image;
}

// Função para normalizar uma imagem e salvá-la como PNG
void normalizeAndSaveImage(const cv::Mat& image, const std::string& filename) {
    cv::Mat normalized;
    cv::normalize(image, normalized, 0, 255, cv::NORM_MINMAX);
    normalized.convertTo(normalized, CV_8U);
    cv::imwrite(filename, normalized);
}

// Função para calcular a diferença absoluta entre duas imagens
void calculateAbsoluteDifference(const cv::Mat& image1, const cv::Mat& image2, cv::Mat& result) {
    result = cv::Mat::zeros(SIDE, SIDE, CV_8U);
  
    for (int i = 0; i < SIDE; i++) {
        for (int j = 0; j < SIDE; j++) {
            result.at<uchar>(i, j) = std::abs(image1.at<uchar>(i, j) - image2.at<uchar>(i, j));
        }
    }
}

int main() {
    std::stringstream ss_yml1, ss_yml2;
    std::stringstream ss_img1, ss_img2;

    cv::Mat image1, image2, result;

    // Cria e salva as imagens senoidais
    ss_yml1 << "senoide1-" << SIDE << ".yml";
    createAndSaveSineImage(ss_yml1.str(), image1, PERIODOS[0]);

    ss_yml2 << "senoide2-" << SIDE << ".yml";
    createAndSaveSineImage(ss_yml2.str(), image2, PERIODOS[1]);

    // Normaliza e salva as imagens como PNG
    normalizeAndSaveImage(image1, "senoide1-" + std::to_string(SIDE) + ".png");
    normalizeAndSaveImage(image2, "senoide2-" + std::to_string(SIDE) + ".png");

    // Carrega as imagens do arquivo YAML
    loadImageFromYAML(ss_yml1.str(), image1);
    loadImageFromYAML(ss_yml2.str(), image2);

    // Normaliza novamente as imagens
    normalizeAndSaveImage(image1, "senoide1-" + std::to_string(SIDE) + ".png");
    normalizeAndSaveImage(image2, "senoide2-" + std::to_string(SIDE) + ".png");

    // Calcula a diferença absoluta entre as imagens
    calculateAbsoluteDifference(image1, image2, result);

    // Salva a diferença absoluta como resultado
    cv::imwrite("result.png", result);

    // Exibe as imagens e espera pela tecla de pressionada
    cv::imshow("image1", image1);
    cv::imshow("image2", image2);
    cv::imshow("result", result);
    cv::waitKey(0);

    return 0;
}
