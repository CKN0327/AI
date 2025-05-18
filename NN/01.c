#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <time.h>

#define MAX_SAMPLES 60000
#define FEATURE_SIZE 784
#define CLASS_NUM 10

typedef struct {
    int label;
    unsigned char features[FEATURE_SIZE];
} Sample;

int load_csv(const char *filename, Sample *data, int max_samples) {
    FILE *fp = fopen(filename, "r");
    if (!fp) {
        perror("錯誤");
        return -1;
    }

    char line[8192];
    int count = 0;

    while (fgets(line, sizeof(line), fp) && count < max_samples) {
        char *tokens[FEATURE_SIZE + 1];
        int num_tokens = 0;

        char *token = strtok(line, ",");
        while (token != NULL && num_tokens < FEATURE_SIZE + 1) {
            tokens[num_tokens++] = token;
            token = strtok(NULL, ",");
        }

        if (num_tokens < 1) {
            data[count].label = -1;
            memset(data[count].features, 0, FEATURE_SIZE);
            count++;
            continue;
        }

        data[count].label = atoi(tokens[num_tokens - 1]);

        int i = 0;
        for (; i < num_tokens - 1 && i < FEATURE_SIZE; ++i) {
            data[count].features[i] = (unsigned char)atoi(tokens[i]);
        }
        for (; i < FEATURE_SIZE; ++i) {
            data[count].features[i] = 0;
        }

        count++;
    }

    fclose(fp);
    return count;
}

double euclidean_distance(unsigned char *a, unsigned char *b) {
    double sum = 0.0;
    for (int i = 0; i < FEATURE_SIZE; ++i) {
        double diff = (double)a[i] - (double)b[i];
        sum += diff * diff;
    }
    return sqrt(sum);
}

int predict_knn(Sample *train_data, int train_size, Sample *test_sample) {
    int best_label = -1;
    double best_dist = DBL_MAX;

    for (int i = 0; i < train_size; ++i) {
        double dist = euclidean_distance(train_data[i].features, test_sample->features);
        if (dist < best_dist) {
            best_dist = dist;
            best_label = train_data[i].label;
        }
    }
    return best_label;
}

double evaluate_macro_f1(int *true_labels, int *pred_labels, int size) {
    int tp[CLASS_NUM] = {0}, fp[CLASS_NUM] = {0}, fn[CLASS_NUM] = {0};

    for (int i = 0; i < size; ++i) {
        int t = true_labels[i];
        int p = pred_labels[i];
        if (t == p)
            tp[t]++;
        else {
            fp[p]++;
            fn[t]++;
        }
    }

    double macro_f1 = 0.0;
    for (int i = 0; i < CLASS_NUM; ++i) {
        double precision = (tp[i] + fp[i] == 0) ? 0 : (double)tp[i] / (tp[i] + fp[i]);
        double recall = (tp[i] + fn[i] == 0) ? 0 : (double)tp[i] / (tp[i] + fn[i]);
        double f1 = (precision + recall == 0) ? 0 : 2 * precision * recall / (precision + recall);
        macro_f1 += f1;
    }

    return macro_f1 / CLASS_NUM;
}

void write_predictions(const char *filename, int *labels, int size) {
    FILE *fp = fopen(filename, "w");
    if (!fp) {
        perror("無法寫入結果檔案");
        return;
    }

    for (int i = 0; i < size; ++i) {
        fprintf(fp, "%d\n", labels[i]);
    }

    fclose(fp);
}

int main() {
    clock_t start_time = clock();

    Sample *train_data = malloc(sizeof(Sample) * MAX_SAMPLES);
    Sample *test_data = malloc(sizeof(Sample) * MAX_SAMPLES);

    printf("載入訓練資料中...\n");
    int train_size = load_csv("mnist_train.csv", train_data, MAX_SAMPLES);
    if (train_size <= 0) {
        printf("沒找到訓練資料\n");
        return 1;
    }

    printf("載入測試資料中...\n");
    int test_size = load_csv("mnist_test.csv", test_data, 10000);
    if (test_size <= 0) {
        printf("沒找到測試資料\n");
        return 1;
    }

    int *true_labels = malloc(sizeof(int) * test_size);
    int *pred_labels = malloc(sizeof(int) * test_size);
    int *train_pred_labels = malloc(sizeof(int) * train_size);
    if (!true_labels || !pred_labels || !train_pred_labels) {
        printf("記憶體配置錯誤\n");
        return 1;
    }

    printf("預測訓練資料中...\n");
    for (int i = 0; i < train_size; ++i) {
        train_pred_labels[i] = predict_knn(train_data, train_size, &train_data[i]);
        if (i % 100 == 0) {
            printf("預測 %d/%d\n", i, test_size);
        }
    }
    write_predictions("result_train.csv", train_pred_labels, train_size);

    printf("預測測試資料中...\n");
    for (int i = 0; i < test_size; ++i) {
        true_labels[i] = test_data[i].label;
        pred_labels[i] = predict_knn(train_data, train_size, &test_data[i]);

        if (i % 100 == 0) {
            printf("預測 %d/%d\n", i, test_size);
        }
    }
    write_predictions("result_test.csv", pred_labels, test_size);

    double macro_f1 = evaluate_macro_f1(true_labels, pred_labels, test_size);

    clock_t end_time = clock();
    double elapsed_sec = (double)(end_time - start_time) / CLOCKS_PER_SEC;

    printf("F1-score: %.4f\n", macro_f1);
    printf("執行時間: %.2f 秒\n", elapsed_sec);

    FILE *result_fp = fopen("result.txt", "w");
    if (result_fp) {
        fprintf(result_fp, "Macro F1-score: %.4f\n", macro_f1);
        fprintf(result_fp, "運行: %.2f 秒\n", elapsed_sec);
        fclose(result_fp);
    }

    free(train_data);
    free(test_data);
    free(true_labels);
    free(pred_labels);
    free(train_pred_labels);
    return 0;
}
