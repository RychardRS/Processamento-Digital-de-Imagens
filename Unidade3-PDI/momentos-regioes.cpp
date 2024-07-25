#include <iostream>
#include <vector>
#include <cmath>
#include <opencv2/opencv.hpp>

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <base_image> <target_image1> <target_image2> ..." << std::endl;
        return -1;
    }

    // Carrega a imagem base
    cv::Mat baseImageGray = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
    cv::Mat baseImageColor = cv::imread(argv[1], cv::IMREAD_COLOR);

    if (baseImageGray.empty() || baseImageColor.empty()) {
        std::cerr << "Could not open or find the base image!" << std::endl;
        return -1;
    }

    // Fator de escala para redimensionamento
    double scaleFactor = 0.5;

    // Loop através de cada imagem alvo
    for (int i = 2; i < argc; ++i) {
        // Carrega a imagem alvo
        cv::Mat targetImageGray = cv::imread(argv[i], cv::IMREAD_GRAYSCALE);
        if (targetImageGray.empty()) {
            std::cerr << "Could not open or find target image " << argv[i] << "!" << std::endl;
            continue;
        }

        // Calcula os momentos Hu para a imagem alvo
        double targetHuMoments[7];
        cv::Moments targetMoments = cv::moments(targetImageGray, false);
        cv::HuMoments(targetMoments, targetHuMoments);

        for (double& moment : targetHuMoments) {
            moment = -1 * std::copysign(1.0, moment) * std::log10(std::abs(moment));
        }

        // Abordagem de janela deslizante para encontrar a imagem alvo na imagem base
        int windowHeight = targetImageGray.rows;
        int windowWidth = targetImageGray.cols;
        double minDiff = std::numeric_limits<double>::max();
        cv::Point bestMatch;

        std::cout << "🔍 Searching for target " << i - 1 << " in the base image" << std::endl;
        for (int y = 0; y <= baseImageGray.rows - windowHeight; ++y) {
            for (int x = 0; x <= baseImageGray.cols - windowWidth; ++x) {
                cv::Rect window(x, y, windowWidth, windowHeight);
                cv::Mat subImage = baseImageGray(window);

                // Calcula os momentos Hu para a janela atual
                double subImageHuMoments[7];
                cv::Moments subImageMoments = cv::moments(subImage, false);
                cv::HuMoments(subImageMoments, subImageHuMoments);

                for (double& moment : subImageHuMoments) {
                    moment = -1 * std::copysign(1.0, moment) * std::log10(std::abs(moment));
                }

                // Calcula a diferença entre os momentos
                double diff = 0.0;
                for (int j = 0; j < 7; ++j) {
                    diff += std::abs(targetHuMoments[j] - subImageHuMoments[j]);
                }

                // Atualiza a melhor correspondência se a diferença atual for menor
                if (diff < minDiff) {
                    minDiff = diff;
                    bestMatch = cv::Point(x, y);
                }
            }
        }

        // Desenha um retângulo na imagem base
        cv::rectangle(baseImageColor, bestMatch, cv::Point(bestMatch.x + windowWidth, bestMatch.y + windowHeight), cv::Scalar(0, 0, 255), 4); // Retângulo vermelho, espessura 4

        std::cout << "Target " << i - 1 << " best match found at: " << bestMatch << std::endl;
        std::cout << "Difference: " << minDiff << std::endl;
        std::cout << "Target " << i - 1 << " Hu Moments: ";
        for (const double& moment : targetHuMoments) {
            std::cout << moment << " ";
        }
        std::cout << std::endl;

        // Exibe a imagem base com o retângulo
        cv::imshow("Base Image with Rectangle", baseImageColor);
        cv::waitKey(0);
    }

    return 0;
}
