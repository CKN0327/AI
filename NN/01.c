#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

#define MAX_SAMPLES 60000
#define FEATURE_SIZE 784
#define CLASS_NUM 10

// ��Ƶ��c
typedef struct {
    int label;
    unsigned char features[FEATURE_SIZE];
} Sample;

int load_csv(const char *filename, Sample *data, int max_samples) {
    FILE *fp = fopen(filename, "r");
    if (!fp) {
        perror("���~");
        return -1;
    }

    char line[8192];
    int count = 0;

    while (fgets(line, sizeof(line), fp) && count < max_samples) {

        char *tokens[FEATURE_SIZE + 1];  // �̦h FEATURE_SIZE + 1 �����]�t label�^
        int num_tokens = 0;

        // �����ξ��
        char *token = strtok(line, ",");
        while (token != NULL && num_tokens < FEATURE_SIZE + 1) {
            tokens[num_tokens++] = token;
            token = strtok(NULL, ",");
        }

        // �ܤ֭n���@�� label �~�B�z
        if (num_tokens < 1) {
            data[count].label = -1;
            memset(data[count].features, 0, FEATURE_SIZE);
            count++;
            continue;
        }

        // label ���̫�@��
        data[count].label = atoi(tokens[num_tokens - 1]);

        // �e�������� features�]������ 0�^
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

// �p��ڴX���o�Z��
double euclidean_distance(unsigned char *a, unsigned char *b) {
    double sum = 0.0;
    for (int i = 0; i < FEATURE_SIZE; ++i) {
        double diff = (double)a[i] - (double)b[i];
        sum += diff * diff;
    }


    return sqrt(sum);
}

// �̪�F�w���]K=1�^
int predict_knn(Sample *train_data, int train_size, Sample *test_sample) {
    int best_label = -1;
    double best_dist = DBL_MAX;

    for (int i = 0; i < train_size; ++i) {
        double dist = euclidean_distance(train_data[i].features, test_sample->features);
        //printf("Dist to train[%d]: %.4f\n", i, dist);
        if (dist < best_dist) {
            best_dist = dist;
            best_label = train_data[i].label;
        }
    }
    return best_label;
}

// Macro F1-score �p��
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

// �D�{��
int main() {

    Sample *train_data = malloc(sizeof(Sample) * MAX_SAMPLES);
    Sample *test_data = malloc(sizeof(Sample) * MAX_SAMPLES);


    printf("���J��...\n");
    int train_size = load_csv("mnist_train.csv", train_data, MAX_SAMPLES);
    if (train_size <= 0) {
        printf("�S���\n");
        return 1;
    }

    printf("���J��...\n");
    int test_size = load_csv("mnist_test.csv", test_data, 10000);
    if (test_size <= 0) {
        printf("�S���\n");
        return 1;
    }

    /*for (int i = 0; i < 10; ++i) {
        printf("Test[%d] label: %d\n", i, test_data[i].label);
    }*/

    int *true_labels = malloc(sizeof(int) * test_size);
    int *pred_labels = malloc(sizeof(int) * test_size);
    if (!true_labels || !pred_labels) {
        printf("�O����t�m���~\n");
        return 1;
    }

    printf("�w��%d samples...\n", test_size);
    for (int i = 0; i < test_size; ++i) {
        true_labels[i] = test_data[i].label;
        pred_labels[i] = predict_knn(train_data, train_size, &test_data[i]);

        if (i % 100 == 0) {
            printf("�w�� %d/%d\n", i, test_size);
        }
    }

    /*for (int i = 0; i < 10; i++) {
        printf("Train label[%d]: %d\n", i, train_data[i].label);
    }*/


    double macro_f1 = evaluate_macro_f1(true_labels, pred_labels, test_size);
    printf("Macro F1-score: %.4f\n", macro_f1);

    free(train_data);
    free(test_data);
    free(true_labels);
    free(pred_labels);
    return 0;
}
