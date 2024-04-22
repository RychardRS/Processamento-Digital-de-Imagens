#include <iostream>
#include <opencv2/opencv.hpp>

#define DECODE 0
#define ENCODE 1
#define SHIFT_LEFT 2

int main(int argc, char** argv) {
    
    std::string e_img_path; // Caminho da imagem de entrada para codificação
    std::string h_img_path; // Caminho da imagem hospedeira
    uchar mode; // Modo de operação (codificação, decodificação, shift_left)

    // APÓS COMPILAR, UTILIZE O COMANDO ABAIXO PARA EXECUTAR:
    //      ./Esteganografia decode ./desafio-esteganografia.png

    if (argc == 1) {
        std::cout << "Nenhum argumento passado.\n";
        return 0;
    } else if (argc == 2) {
        if (std::strcmp(argv[1], "encode") == 0) {
            std::cout << "Sintaxe correta para codificação é \"stenography encode <caminho-para-imagem-de-codificacao> <caminho-para-imagem-hospedeira>\".\n";
            return 0;
        } else if (std::strcmp(argv[1], "decode") == 0) {
            std::cout << "Sintaxe correta para decodificação é \"stenography decode <caminho-para-imagem>\".\n";
            return 0;
        } else if (std::strcmp(argv[1], "shift_left") == 0) {
            std::cout << "Sintaxe correta para mostrar planos é \"stenography shift_left <caminho-para-imagem>\".\n";
            return 0;
        } else {
            std::cout << "Opções válidas são \"encode\", \"decode\" ou \"shift_left\".\n";
            return 0;
        }
    } else if (argc == 3) {
        if (std::strcmp(argv[1], "decode") == 0) {
            h_img_path = argv[2];
            mode = DECODE;
        } else if (std::strcmp(argv[1], "shift_left") == 0){
            h_img_path = argv[2];
            mode = SHIFT_LEFT;
        } else if (std::strcmp(argv[1], "encode") == 0) {
            std::cout << "Sintaxe correta para codificação é \"stenography encode <caminho-para-imagem-de-codificacao> <caminho-para-imagem-hospedeira>\".\n";
            return 0;
        } else {
            std::cout << "Opções válidas são \"encode\", \"decode\" e \"shift_left\".\n";
            return 0;
        }
    } else if (argc == 4) {
        if (std::strcmp(argv[1], "encode") == 0) {
            e_img_path = argv[2];
            h_img_path = argv[3];
            mode = ENCODE;
        } else if (std::strcmp(argv[1], "decode") == 0) {
            std::cout << "Sintaxe correta para decodificação é \"stenography decode <caminho-para-imagem>\".\n";
            return 0;
        } else if (std::strcmp(argv[1], "shift_left") == 0) {
            std::cout << "Sintaxe correta para mostrar planos é \"stenography shift_left <caminho-para-imagem>\".\n";
            return 0;
        } else {
            std::cout << "Opções válidas são \"encode\" e \"decode\".\n";
            return 0;
        }
    } else {
        std::cout << "Número inválido de argumentos.\n";
        return 0;
    }

    if (mode == ENCODE) {
        cv::Mat h_img, e_img;
        h_img = cv::imread(h_img_path, cv::IMREAD_COLOR);
        e_img = cv::imread(e_img_path, cv::IMREAD_COLOR);

        if (h_img.rows != e_img.rows || h_img.cols != e_img.cols) {
            std::cout << "Ambas as imagens devem ter as mesmas dimensões.\n";
            return 0;
        }

        for (int i = 0; i < h_img.rows; i++) {
            for (int j = 0; j < h_img.cols; j++) {
                h_img.at<cv::Vec3b>(i, j)[0] = (h_img.at<cv::Vec3b>(i, j)[0] & 248) | (e_img.at<cv::Vec3b>(i, j)[0] >> 5);
                h_img.at<cv::Vec3b>(i, j)[1] = (h_img.at<cv::Vec3b>(i, j)[1] & 248) | (e_img.at<cv::Vec3b>(i, j)[1] >> 5);
                h_img.at<cv::Vec3b>(i, j)[2] = (h_img.at<cv::Vec3b>(i, j)[2] & 248) | (e_img.at<cv::Vec3b>(i, j)[2] >> 5);
            }
        }

        cv::imwrite("encoded_output.png", h_img);

    } else if (mode == DECODE) {
        
        cv::Mat h_img;
        h_img = cv::imread(h_img_path, cv::IMREAD_COLOR);

        if (h_img.data == NULL) {
            std::cout << "Não foi possível abrir a imagem.\n";
            exit(-1);
        }

        for (int i = 0; i < h_img.rows; i++) {
            for (int j = 0; j < h_img.cols; j++) {
                h_img.at<cv::Vec3b>(i, j)[0] = h_img.at<cv::Vec3b>(i, j)[0] << 5;
                h_img.at<cv::Vec3b>(i, j)[1] = h_img.at<cv::Vec3b>(i, j)[1] << 5;
                h_img.at<cv::Vec3b>(i, j)[2] = h_img.at<cv::Vec3b>(i, j)[2] << 5;
            }
        }

        cv::imwrite("decoded_output.png", h_img);

    } else {
        cv::Mat h_img;
        h_img = cv::imread(h_img_path, cv::IMREAD_COLOR);
        cv::Mat plane_h_img(2*h_img.rows, 4*h_img.cols, CV_8UC3);

        for (uchar k = 0; k < 8; k++) {
            int dispx = k > 3 ? h_img.rows : 0;
            int dispy = k > 3 ? (k - 4)*h_img.cols : k*h_img.cols;

            for (int i = 0; i < h_img.rows; i++) {
                for (int j = 0; j < h_img.cols; j++) {
                    plane_h_img.at<cv::Vec3b>(i + dispx,j + dispy)[0] = h_img.at<cv::Vec3b>(i,j)[0] << k;
                    plane_h_img.at<cv::Vec3b>(i + dispx,j + dispy)[1] = h_img.at<cv::Vec3b>(i,j)[1] << k;
                    plane_h_img.at<cv::Vec3b>(i + dispx,j + dispy)[2] = h_img.at<cv::Vec3b>(i,j)[2] << k;
                }
            }
        }

        cv::imwrite("planes_output.png", plane_h_img);
    }

    return 0;
}
